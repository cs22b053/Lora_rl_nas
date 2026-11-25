"""Evaluate a corrected checkpoint using its saved architecture and preprocessing."""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mypath import Path as DatasetPath
from reference_training import model_from_checkpoint, read_checkpoint, validate
from retrain_data import CityscapesFine, CLASS_NAMES


def run(args):
    if args.train or args.dataset != 'cityscapes':
        raise ValueError('Corrected evaluation requires Cityscapes validation and no --train')
    if args.net_arch or args.cell_arch:
        raise ValueError('The corrected checkpoint stores its architecture; omit --net_arch/--cell_arch')
    if args.eval_batch_size < 1 or args.workers < 0:
        raise ValueError('Invalid evaluation batch size/workers')
    checkpoint = read_checkpoint(args.checkpoint)
    if checkpoint.get('data_protocol'):
        raise ValueError('This is an architecture-selection probe. Use its held-out TRAIN metrics; '
                         'evaluate the fresh full-data retraining checkpoint on official validation.')
    if not torch.cuda.is_available():
        raise RuntimeError('Submit run_test_baseline.sh on the GPU')
    device = torch.device('cuda', args.gpu)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root = args.dataset_root or checkpoint['train_config'].get('dataset_root') or DatasetPath.db_root_dir('cityscapes')
    dataset = CityscapesFine(root, 'val')
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True, drop_last=False)
    model = model_from_checkpoint(checkpoint).to(device)
    result = validate(model, loader, device, args.eval_scales, args.eval_flip, amp=not args.eval_fp32,
                      cvar_alpha=args.eval_cvar_alpha)
    result.update(checkpoint=str(Path(args.checkpoint).resolve()), step=checkpoint['step'],
                  model_config=checkpoint['model_config'], model_version=checkpoint['model_version'],
                  scales=args.eval_scales, flip=args.eval_flip, amp=not args.eval_fp32,
                  dataset_root=str(Path(root).resolve()), split='val')
    protocol = 'scales_{}_flip_{}_amp_{}'.format('_'.join(format(s, 'g') for s in args.eval_scales),
                                               int(args.eval_flip), int(not args.eval_fp32))
    protocol += '_cvar_{}'.format(format(args.eval_cvar_alpha, 'g'))
    destination = Path(args.results_dir or 'log_reference_eval') / protocol
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / '{}_step_{}.json'.format(Path(args.checkpoint).stem, checkpoint['step'])
    path.write_text(json.dumps(result, indent=2) + '\n')
    for name, value in zip(CLASS_NAMES, result['class_iou']):
        print('{}: {:.2f}'.format(name, value))
    print('Update {}: Mean IoU {:.3f}% ({} validation images)'.format(
        checkpoint['step'], result['miou'], result['images']), flush=True)
    if result['image_cvar'] is not None:
        print('Worst {:.1f}% image IoU mean (CVaR): {:.3f}%'.format(
            args.eval_cvar_alpha * 100, result['image_cvar']), flush=True)
    print('Results: {}'.format(path.resolve()), flush=True)
