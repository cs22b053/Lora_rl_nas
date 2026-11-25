"""Measured rewards: mean segmentation IoU and lower-tail image performance."""
import torch
from torch import nn
from search_policy import lower_tail_mean


@torch.no_grad()
def score_candidate(model, cell, calibration, panel, device, alpha=.2, risk_weight=.2, amp=True):
    """Retry a failed AMP score from clean BN state on the SAME data in FP32.

    No invalid score enters PPO. A second non-finite result remains a hard
    failure; finite AMP evaluations retain their existing numerical protocol.
    """
    # Callers normally supply cached lists. Materialize iterators as well so a
    # retry cannot silently use different augmentations or an exhausted panel.
    calibration, panel = tuple(calibration), tuple(panel)
    use_amp = amp and device.type == 'cuda'
    try:
        result = _score_candidate_once(model, cell, calibration, panel, device,
                                       alpha, risk_weight, amp=use_amp)
        result.update(scoring_precision='amp' if use_amp else 'fp32', amp_fallback=False)
    except FloatingPointError as error:
        if not use_amp:
            raise
        print('Candidate AMP scoring failed ({}); resetting BN and retrying the same '
              'calibration/evaluation images in FP32.'.format(error), flush=True)
        try:
            result = _score_candidate_once(model, cell, calibration, panel, device,
                                           alpha, risk_weight, amp=False)
        except FloatingPointError as retry_error:
            raise FloatingPointError('Candidate scoring is non-finite even after fresh FP32 '
                                     'BN calibration; no reward was produced: {}'.format(retry_error)) from retry_error
        result.update(scoring_precision='fp32', amp_fallback=True, retry_reason=str(error))
    result['scoring_protocol'] = 'finite_retry_v1'
    return result


@torch.no_grad()
def _score_candidate_once(model, cell, calibration, panel, device, alpha, risk_weight, amp):
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    modes = {name: m.training for name, m in model.named_modules()}
    momenta = {name: m.momentum for name, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)}
    try:
        model.set_architecture(cell)
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = None
                m.train()
        for sample in calibration:
            image = sample['image'].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
                logits = model(image)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite logits during candidate BN calibration')
            del logits
        for name, value in model.named_buffers():
            if not torch.isfinite(value).all():
                raise FloatingPointError('Non-finite candidate calibration buffer: ' + name)
        model.eval()
        confusion = torch.zeros(19, 19, dtype=torch.int64, device=device)
        image_scores = []
        for sample in panel:
            image = sample['image'].to(device, non_blocking=True)
            target = sample['label'].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
                logits = model(image)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite candidate logits; refusing to rank this evaluation')
            prediction = logits.argmax(1)
            del logits
            for gt, pred in zip(target, prediction):
                valid = (gt >= 0) & (gt < 19)
                if not valid.any():
                    continue
                counts = torch.bincount(19*gt[valid]+pred[valid], minlength=361).reshape(19, 19)
                confusion += counts
                union = counts.sum(0) + counts.sum(1) - counts.diag()
                present = counts.sum(1) > 0
                image_scores.append((counts.diag()[present] / union[present].clamp_min(1)).mean())
        if len(image_scores) * alpha < 8:
            raise ValueError('Too few labeled images for CVaR: require at least 8 images in the tail')
        union = confusion.sum(0) + confusion.sum(1) - confusion.diag()
        mean = (confusion.diag() / union.clamp_min(1)).mean()
        tail = lower_tail_mean(torch.stack(image_scores), alpha)
        reward = (1-risk_weight) * mean + risk_weight * tail
        return dict(reward=reward.item(), miou=mean.item(), image_cvar=tail.item(),
                    images=len(image_scores), image_scores=torch.stack(image_scores).cpu().tolist())
    finally:
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
        for name, module in model.named_modules():
            module.training = modes[name]
            if name in momenta:
                module.momentum = momenta[name]
