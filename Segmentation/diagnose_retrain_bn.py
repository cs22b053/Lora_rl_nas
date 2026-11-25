"""Compare stored and train-only recalibrated BN statistics in a saved pilot.

No optimization, checkpoint modification, or validation-set calibration occurs.
Both measurements use FP32, full-resolution, single-scale validation.
"""
import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from reference_training import calibrated_batch_norm, model_from_checkpoint, read_checkpoint, validate
from retrain_data import CityscapesFine, seed_worker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--calibration-batches', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.batch_size < 2 or args.calibration_batches < 1 or args.workers < 0:
        p.error('Require batch size >=2, calibration batches >=1 and workers >=0')
    if not torch.cuda.is_available():
        raise RuntimeError('Submit run_diagnose_pilots.sh on the GPU')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    checkpoint = read_checkpoint(args.checkpoint)
    if checkpoint.get('data_protocol'):
        raise ValueError('Independent probes already use train-only BN calibration on their saved split')
    config = checkpoint['train_config']
    train = CityscapesFine(config['dataset_root'], 'train', config['crop_size'])
    val = CityscapesFine(config['dataset_root'], 'val')
    calibration = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                             num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker,
                             generator=torch.Generator().manual_seed(args.seed))
    if args.calibration_batches > len(calibration):
        p.error('Calibration count exceeds available complete training batches')
    validation = DataLoader(val, batch_size=1, shuffle=False, num_workers=args.workers,
                            pin_memory=True)
    model = model_from_checkpoint(checkpoint).to(device)
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all():
            raise FloatingPointError('Non-finite checkpoint tensor: ' + name)
    result = dict(checkpoint=str(Path(args.checkpoint).resolve()), step=checkpoint['step'],
                  model_config=checkpoint['model_config'], seed=args.seed,
                  calibration_split='train', evaluation_split='val',
                  calibration_batches=args.calibration_batches,
                  calibration_images=args.calibration_batches * args.batch_size,
                  crop_size=config['crop_size'], scales=[1.0], flip=False, amp=False)
    del checkpoint
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_suffix(output.suffix + '.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(output)

    print('Stored BN statistics: full validation in FP32', flush=True)
    result['stored_bn'] = validate(model, validation, device, amp=False)
    print('Stored BN mIoU: {:.3f}%'.format(result['stored_bn']['miou']), flush=True)
    save()
    print('Recalibrating BN using training images only; all weights frozen', flush=True)
    with calibrated_batch_norm(model, calibration, device, args.calibration_batches):
        result['calibrated_bn'] = validate(model, validation, device, amp=False)
    result['delta_miou_points'] = result['calibrated_bn']['miou'] - result['stored_bn']['miou']
    result['complete'] = True
    save()
    print('Calibrated BN mIoU: {:.3f}%; change {:+.3f} percentage points'.format(
        result['calibrated_bn']['miou'], result['delta_miou_points']), flush=True)
    print('Report: {}. Original checkpoint unchanged.'.format(output), flush=True)


if __name__ == '__main__':
    main()
