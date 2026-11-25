"""A search minibatch completes only after an actual optimizer update."""
import torch
from torch import nn
from torch.nn import functional as F
from search_supernet import InputLoRAConv


def inactive_adapter_state(adapters, optimizer):
    """Snapshot inactive slices: a zero dense gradient does not stop SGD momentum."""
    snapshots = []
    for module in adapters:
        if not module.active_sources:
            continue
        if len(module.active_sources) == module.lora_A.shape[0]:
            continue
        inactive = torch.tensor([i not in module.active_sources for i in range(module.lora_A.shape[0])],
                                device=module.lora_A.device, dtype=torch.bool)
        for parameter in (module.lora_A, module.lora_B):
            if parameter.grad is None:
                continue
            momentum = optimizer.state.get(parameter, {}).get('momentum_buffer')
            snapshots.append((parameter, inactive, parameter.detach()[inactive].clone(),
                              None if momentum is None else momentum[inactive].clone()))
    return snapshots


@torch.no_grad()
def restore_inactive_adapter_state(snapshots, optimizer):
    for parameter, inactive, values, old_momentum in snapshots:
        parameter[inactive] = values
        momentum = optimizer.state.get(parameter, {}).get('momentum_buffer')
        if momentum is not None:
            momentum[inactive] = 0 if old_momentum is None else old_momentum


def update_weights(model, optimizer, scaler, image, target, amp=True, max_attempts=16):
    adapters = [m for m in model.modules() if isinstance(m, InputLoRAConv)]
    if adapters and not isinstance(optimizer, torch.optim.SGD):
        raise ValueError('Input-specific adapter update isolation currently supports SGD only')
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    for attempt in range(max_attempts):
        for module in adapters:
            module.active_sources.clear()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=image.device.type, enabled=amp and image.is_cuda):
            logits = model(image)
            loss = F.cross_entropy(logits, target, ignore_index=255, reduction='none').sum()
            loss = loss / (target != 255).sum().clamp_min(1)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite supernet loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=not scaler.is_enabled())
        old_scale = scaler.get_scale()
        inactive = inactive_adapter_state(adapters, optimizer)
        scaler.step(optimizer)
        restore_inactive_adapter_state(inactive, optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            return dict(loss=loss.item(), retries=attempt, amp_scale=scaler.get_scale())
        # Retry this same architecture/minibatch. Do not advance the epoch,
        # schedule, data cursor or controller after a skipped AMP update.
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        print('AMP overflow: retry {}/{}; scale {} -> {}'.format(
            attempt+1, max_attempts, old_scale, scaler.get_scale()), flush=True)
    raise FloatingPointError('No successful weight update after {} attempts; inspect numerical stability '
                             'or run a fresh --no-amp smoke test.'.format(max_attempts))
