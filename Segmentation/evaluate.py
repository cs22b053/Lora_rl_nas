import os
from pathlib import Path as FilePath

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.utils.data import DataLoader

from mypath import Path
from utils.utils import inter_and_union
from config_utils.evaluate_args import obtain_evaluate_args


def select_checkpoints(args):
    prefix = 'deeplab_{}_{}_v3_{}_epoch'.format(args.backbone, args.dataset, args.exp)
    data_dir = FilePath(args.data_dir)
    if args.checkpoint_epochs:
        checkpoints = [(epoch, data_dir / '{}{}.pth'.format(prefix, epoch))
                       for epoch in sorted(set(args.checkpoint_epochs))]
        for _, path in checkpoints:
            if not path.is_file():
                raise FileNotFoundError('Checkpoint not found: {}'.format(path))
    else:
        checkpoints = []
        for path in data_dir.glob('{}*.pth'.format(prefix)):
            suffix = path.stem[len(prefix):]
            if suffix.isdigit():
                checkpoints.append((int(suffix), path))
        checkpoints.sort()
    if not checkpoints:
        raise FileNotFoundError('No saved checkpoints matching {}*.pth in {}'.format(prefix, data_dir))
    return checkpoints


def load_checkpoint(model, path):
    # Load optimizer tensors onto CPU too, avoiding a transient GPU memory spike.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state_dict = checkpoint['state_dict']
    consume_prefix_in_state_dict_if_present(state_dict, 'module.')
    # Preserve all BN buffers and reject mismatched models.
    model.load_state_dict(state_dict, strict=True)
    model.eval()


@torch.inference_mode()
def evaluate(model, dataloader, num_classes, device, scales, flip):
    # inference_mode alone does not disable batch statistics or dropout.
    model.eval()
    intersection = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)
    num_images = 0
    for i, sample in enumerate(dataloader):
        inputs = sample['image'].to(device, non_blocking=True)
        target = sample['label']
        n, height, width = target.shape
        total_outputs = torch.zeros((n, num_classes, height, width), device=device)
        for scale in scales:
            size = (max(1, int(height * scale)), max(1, int(width * scale)))
            scaled_inputs = F.interpolate(inputs, size=size, mode='bilinear', align_corners=True)
            outputs = model(scaled_inputs)
            total_outputs += F.interpolate(outputs, size=(height, width),
                                           mode='bilinear', align_corners=True)
            if flip:
                outputs = model(torch.flip(scaled_inputs, dims=[3]))
                outputs = torch.flip(outputs, dims=[3])
                total_outputs += F.interpolate(outputs, size=(height, width),
                                               mode='bilinear', align_corners=True)
        # Keep the batch dimension even for a final batch containing one image.
        pred = total_outputs.argmax(dim=1).cpu().numpy().astype(np.uint8)
        mask = target.numpy().astype(np.uint8)
        inter, batch_union = inter_and_union(pred, mask, num_classes)
        intersection += inter
        union += batch_union
        num_images += n
        if (i + 1) % 10 == 0 or i + 1 == len(dataloader):
            print('eval: {}/{} ({} images)'.format(i + 1, len(dataloader), num_images), flush=True)
    if not num_images:
        raise ValueError('Validation loader is empty')
    # Aggregate dataset-wide counts before calculating class IoU.
    return intersection / (union + 1e-10), num_images


def main():
    args = obtain_evaluate_args()
    if args.checkpoint:
        from evaluate_baseline import run
        return run(args)
    from retrain_model.build_autodeeplab import Retrain_Autodeeplab
    if args.train:
        raise ValueError('evaluate.py only supports evaluation; omit --train')
    if args.dataset != 'cityscapes' or args.backbone != 'autodeeplab':
        raise ValueError('This evaluator supports Auto-DeepLab on Cityscapes')
    if args.eval_batch_size < 1 or args.workers < 0:
        raise ValueError('Evaluation batch size must be positive and workers nonnegative')
    if any(not np.isfinite(scale) or scale <= 0 for scale in args.eval_scales):
        raise ValueError('Evaluation scales must be finite and positive')
    checkpoints = select_checkpoints(args)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; submit run_test_ppo.sh to a GPU node')
    device = torch.device('cuda', args.gpu)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    from dataloaders.datasets.cityscapes import CityscapesSegmentation

    dataset = CityscapesSegmentation(args=args, root=Path.db_root_dir(args.dataset), split='reval')
    args.num_classes = len(dataset.CLASSES)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False,
                            persistent_workers=args.workers > 0)
    model = Retrain_Autodeeplab(args).to(device)

    # Separate corrected results from old logs and other inference protocols.
    protocol = 'scales_{}_flip_{}'.format('_'.join(format(s, 'g') for s in args.eval_scales),
                                         int(args.eval_flip))
    results_root = args.results_dir or 'log_{}_eval'.format(args.exp)
    results_dir = FilePath(results_root) / protocol
    results_dir.mkdir(parents=True, exist_ok=True)
    print('Evaluating {} checkpoints; scales={}, flip={}, batch_size={}'.format(
        len(checkpoints), args.eval_scales, args.eval_flip, args.eval_batch_size), flush=True)
    print('Results: {}'.format(results_dir.resolve()), flush=True)
    best = None
    for epoch, checkpoint_path in checkpoints:
        print('\nEvaluating epoch {}: {}'.format(epoch, checkpoint_path), flush=True)
        load_checkpoint(model, checkpoint_path)
        iou, num_images = evaluate(model, dataloader, args.num_classes, device,
                                   args.eval_scales, args.eval_flip)
        miou = float(iou.mean() * 100)
        lines = ['Checkpoint: {}'.format(checkpoint_path.resolve()),
                 'Network architecture: {}'.format(args.net_arch),
                 'Cell architecture: {}'.format(args.cell_arch),
                 'Scales: {}; flip: {}; batch size: {}; images: {}'.format(
                     args.eval_scales, args.eval_flip, args.eval_batch_size, num_images)]
        lines.extend('IoU {}: {:.2f}'.format(name, value * 100)
                     for name, value in zip(dataset.CLASSES, iou))
        lines.append('epoch: {} Mean IoU: {:.2f}'.format(epoch, miou))
        # Replace one result per checkpoint instead of appending duplicate scores.
        result_path = results_dir / 'result_{}.txt'.format(epoch)
        temporary_path = result_path.with_suffix('.txt.tmp.{}'.format(os.getpid()))
        temporary_path.write_text('\n'.join(lines) + '\n')
        temporary_path.replace(result_path)
        print('\n'.join(lines), flush=True)
        if best is None or miou > best[1]:
            best = (epoch, miou)
    print('\nBest evaluated checkpoint: epoch {}, Mean IoU: {:.2f}'.format(*best), flush=True)


if __name__ == '__main__':
    main()
