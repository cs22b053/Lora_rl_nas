"""Training and evaluation utilities for the corrected baseline."""
import os
import struct
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from retrain_model.reference_model import MODEL_VERSION, ReferenceAutoDeepLab


def compatible_cuda_rng_state(state, expected_size):
    """Preserve Philox seed/offset across the known 816 -> 16 byte format change.

    PyTorch 2.0.1 CUDAGeneratorImpl stored 800 bytes of 0xff legacy padding,
    then uint64 seed + int64 offset. PyTorch 2.3.1 removed only that padding.
    Refuse unknown layouts instead of silently reseeding or truncating them.
    """
    if (not isinstance(state, torch.Tensor) or state.dtype != torch.uint8
            or state.device.type != 'cpu' or state.ndim != 1):
        raise ValueError('CUDA RNG state must be a one-dimensional CPU byte tensor')
    if state.numel() == expected_size:
        return state.contiguous()
    if state.numel() == 816 and expected_size == 16:
        if not bool((state[:800] == 255).all()):
            raise ValueError('Unrecognized legacy CUDA RNG padding; refusing conversion')
        _, offset = struct.unpack('=Qq', bytes(state[800:].tolist()))
        if offset < 0 or offset % 4:
            raise ValueError('Invalid legacy Philox offset; refusing conversion')
        return state[800:].clone().contiguous()
    raise ValueError('Unsupported CUDA RNG state size: saved {}, current {}. '
                     'Checkpoint was not changed.'.format(state.numel(), expected_size))


def restore_cuda_rng_states(saved_states):
    current_states = torch.cuda.get_rng_state_all()
    if len(saved_states) != len(current_states):
        raise ValueError('CUDA RNG device count changed: saved {}, current {}'.format(
            len(saved_states), len(current_states)))
    converted = [compatible_cuda_rng_state(saved, current.numel())
                 for saved, current in zip(saved_states, current_states)]
    migrations = [dict(device=i, saved_bytes=saved.numel(), restored_bytes=new.numel())
                  for i, (saved, new) in enumerate(zip(saved_states, converted))
                  if saved.numel() != new.numel()]
    torch.cuda.set_rng_state_all(converted)
    for event in migrations:
        print('Converted CUDA RNG state for device {device}: {saved_bytes} -> '
              '{restored_bytes} bytes; preserved Philox seed and offset.'.format(**event), flush=True)
    return migrations


def resolve_training_steps(train_images, batch_size, accumulation_steps, epochs=None, max_steps=None):
    if min(train_images, batch_size, accumulation_steps) < 1:
        raise ValueError('Dataset size and batch settings must be positive')
    batches = train_images // batch_size  # DataLoader drop_last=True
    if not batches:
        raise ValueError('Training loader is empty')
    if epochs is not None:
        if epochs < 1 or max_steps is not None:
            raise ValueError('Choose positive --epochs OR --max-steps')
        if batches % accumulation_steps:
            raise ValueError('Epoch schedules require accumulation_steps to divide batches per epoch')
        return epochs * (batches // accumulation_steps)
    if max_steps is not None and max_steps < 1:
        raise ValueError('--max-steps must be positive')
    return 1500000 if max_steps is None else max_steps


def train_update(model, optimizer, scaler, samples, device, amp=True, max_attempts=16):
    """Retry AMP overflows on the same data and restore buffers before retrying."""
    if not samples:
        raise ValueError('An update requires at least one microbatch')
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    for attempt in range(max_attempts):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for sample in samples:
            image = sample['image'].to(device, non_blocking=True)
            target = sample['label'].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
                logits = model(image)
                losses = F.cross_entropy(logits, target, ignore_index=255, reduction='none')
                loss = losses.sum() / (target != 255).sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            scaler.scale(loss / len(samples)).backward()
            total_loss += loss.item() / len(samples)
            del logits, losses, loss, image, target
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            return total_loss
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        print('AMP overflow: retry {}/{} on the same batch; scale {} -> {}'.format(
            attempt + 1, max_attempts, old_scale, scaler.get_scale()), flush=True)
    raise FloatingPointError('Repeated AMP overflows; no optimizer update completed')


@contextmanager
def calibrated_batch_norm(model, loader, device, batches):
    """Temporarily estimate BN statistics from training images in FP32.

    Dropout/drop path stay disabled. No gradients, validation images or labels
    are used for calibration. Restore buffers and modes even if evaluation fails.
    """
    if batches < 1:
        raise ValueError('Calibration batches must be positive')
    norms = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    modes = {m: m.training for m in model.modules()}
    momenta = {m: m.momentum for m in norms}
    try:
        model.eval()
        for m in norms:
            m.reset_running_stats()
            m.momentum = None
            m.train()
        iterator = iter(loader)
        with torch.no_grad():
            for index in range(batches):
                try:
                    sample = next(iterator)
                except StopIteration:
                    raise ValueError('Not enough complete training batches for calibration') from None
                model(sample['image'].to(device, non_blocking=True).float())
                if (index + 1) % 32 == 0:
                    print('BN calibration {}/{}'.format(index + 1, batches), flush=True)
        model.eval()
        for name, value in model.named_buffers():
            if not torch.isfinite(value).all():
                raise FloatingPointError('Non-finite calibrated buffer: ' + name)
        yield
    finally:
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        for module, training in modes.items():
            module.training = training
        for module, momentum in momenta.items():
            module.momentum = momentum


def parameter_groups(model, weight_decay):
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim > 1 else no_decay).append(parameter)
    if not decay or not no_decay:
        raise ValueError('Unexpected empty optimizer group')
    return [{'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}]


def learning_rate(step, total_steps, base_lr, warmup_steps=1000, power=0.9):
    if warmup_steps and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    fraction = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * max(0.0, 1.0 - fraction) ** power


def set_learning_rate(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def atomic_save(checkpoint, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('model_version') != MODEL_VERSION:
        raise ValueError('This is a legacy/incompatible checkpoint. Evaluate it with the legacy '
                         'evaluate.py arguments; do not resume corrected training from it.')
    return checkpoint


def model_from_checkpoint(checkpoint):
    model = ReferenceAutoDeepLab(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model


@torch.inference_mode()
def validate(model, loader, device, scales=(1.0,), flip=False, amp=True, cvar_alpha=.2):
    from search_policy import lower_tail_mean
    if not scales or any(not 0 < s < float('inf') for s in scales):
        raise ValueError('Evaluation scales must be finite and positive')
    if not 0 < cvar_alpha <= 1:
        raise ValueError('CVaR alpha must be in (0, 1]')
    model.eval()
    confusion = torch.zeros((19, 19), dtype=torch.int64, device=device)
    images = 0
    image_scores = []
    for index, sample in enumerate(loader):
        image = sample['image'].to(device, non_blocking=True)
        target = sample['label'].to(device, non_blocking=True)
        height, width = target.shape[-2:]
        scores = torch.zeros((image.shape[0], 19, height, width), device=device)
        for scale in scales:
            size = (max(1, int((height - 1) * scale + 1)),
                    max(1, int((width - 1) * scale + 1)))
            scaled = F.interpolate(image, size=size, mode='bilinear', align_corners=True)
            for mirrored in ([False, True] if flip else [False]):
                with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
                    logits = model(scaled.flip(-1) if mirrored else scaled)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Non-finite validation logits; do not report this as mIoU')
                if mirrored:
                    logits = logits.flip(-1)
                logits = F.interpolate(logits.float(), size=(height, width),
                                       mode='bilinear', align_corners=True)
                scores.add_(logits.softmax(dim=1))
        prediction = scores.argmax(dim=1)
        for gt, pred in zip(target, prediction):
            valid = (gt >= 0) & (gt < 19)
            counts = torch.bincount(19 * gt[valid] + pred[valid], minlength=19 ** 2).reshape(19, 19)
            confusion += counts
            present = counts.sum(1) > 0
            if present.any():
                counts = counts.double()
                union = counts.sum(0) + counts.sum(1) - counts.diag()
                image_scores.append((counts.diag()[present] / union[present].clamp_min(1)).mean().item())
        images += image.shape[0]
        if (index + 1) % 50 == 0:
            print('Validation {}/{}'.format(index + 1, len(loader)), flush=True)
    if not images:
        raise ValueError('Validation dataset is empty')
    counts = confusion.double()
    intersection = counts.diag()
    union = counts.sum(0) + counts.sum(1) - intersection
    iou = intersection / union.clamp_min(1)
    tail = (lower_tail_mean(torch.tensor(image_scores, dtype=torch.float64), cvar_alpha).item() * 100
            if image_scores else None)
    return {'miou': iou.mean().item() * 100, 'class_iou': (iou * 100).cpu().tolist(),
            'images': images, 'confusion': confusion.cpu().tolist(),
            'image_cvar': tail, 'image_cvar_alpha': cvar_alpha,
            'image_iou_definition': 'mean IoU over ground-truth-present classes, excluding void pixels',
            'image_miou_percent': [s * 100 for s in image_scores]}
