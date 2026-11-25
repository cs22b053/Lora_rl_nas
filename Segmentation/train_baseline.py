"""Single-GPU, step-based training of the corrected Auto-DeepLab baseline."""
import argparse
import json
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mypath import Path as DatasetPath
from retrain_data import CityscapesFine, seed_worker
from retrain_model.reference_model import MODEL_VERSION, ReferenceAutoDeepLab
from reference_training import (atomic_save, learning_rate, parameter_groups, read_checkpoint,
                                 resolve_training_steps, restore_cuda_rng_states,
                                 set_learning_rate, train_update, validate, calibrated_batch_norm)
from probe_data import probe_datasets


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['cityscapes'], default='cityscapes')
    p.add_argument('--dataset-root', default=DatasetPath.db_root_dir('cityscapes'))
    p.add_argument('--data-dir', '--data_dir', dest='data_dir', default='data_reference_f32')
    budget = p.add_mutually_exclusive_group()
    budget.add_argument('--max-steps', type=int, default=None,
                        help='Successful optimizer updates; defaults to 1500000 if --epochs is omitted')
    budget.add_argument('--epochs', type=int, default=None,
                        help='Convert complete drop-last data epochs to an optimizer-update budget')
    p.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int, default=16)
    p.add_argument('--accumulation-steps', type=int, default=1)
    p.add_argument('--crop-size', '--crop_size', dest='crop_size', type=int, default=769)
    p.add_argument('--filter-multiplier', '--filter_multiplier', dest='filter_multiplier', type=int, default=32)
    p.add_argument('--architecture-json', default=None,
                   help='Architecture exported by corrected search; omit for published baseline')
    p.add_argument('--base-lr', '--base_lr', dest='base_lr', type=float, default=0.05)
    p.add_argument('--weight-decay', type=float, default=0.0001)
    p.add_argument('--warmup-steps', type=int, default=1000)
    p.add_argument('--power', type=float, default=0.9)
    p.add_argument('--drop-path-prob', type=float, default=0.1)
    p.add_argument('--bn-momentum', type=float, default=0.0003,
                   help='PyTorch running-statistics update rate; saved with the model')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--resume', default='latest', help='latest, none, or a corrected checkpoint path')
    p.add_argument('--val-every', type=int, default=5000)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--snapshot-every', type=int, default=50000)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--stop-after-steps', type=int, default=None,
                   help='Stop after this many updates in THIS invocation, without changing the LR horizon')
    p.add_argument('--stop-at-step', type=int, default=None,
                   help='Absolute stage boundary; resubmitting a finished stage does not advance it')
    p.add_argument('--selection-split-json', default=None,
                   help='Independent probe: train/score only on disjoint parts of Cityscapes TRAIN')
    p.add_argument('--selection-calibration-batches', type=int, default=0)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.batch_size < 2 or args.accumulation_steps < 1:
        raise ValueError('Physical batch must be >=2 and accumulation_steps >=1')
    train_set = CityscapesFine(args.dataset_root, 'train', args.crop_size)
    data_protocol = None
    if args.selection_split_json:
        train_set, val_set, data_protocol = probe_datasets(train_set, args.selection_split_json)
        if args.selection_calibration_batches < 1:
            raise ValueError('Independent probes require fixed train-only BN calibration')
    else:
        if args.selection_calibration_batches:
            raise ValueError('Selection calibration is only supported for train-only probes')
        val_set = CityscapesFine(args.dataset_root, 'val', args.crop_size)
    args.max_steps = resolve_training_steps(len(train_set), args.batch_size,
                                           args.accumulation_steps, args.epochs, args.max_steps)
    for name in ('max_steps', 'accumulation_steps', 'crop_size', 'val_every', 'save_every',
                 'snapshot_every', 'log_every'):
        if getattr(args, name) < 1:
            raise ValueError('{} must be positive'.format(name))
    if args.batch_size < 2:
        raise ValueError('Physical batch must be >=2: the global ASPP branch uses BatchNorm')
    if args.workers < 0 or not 0 <= args.warmup_steps < args.max_steps:
        raise ValueError('Invalid workers/warmup setting')
    if args.base_lr <= 0 or args.weight_decay < 0 or args.power <= 0:
        raise ValueError('Invalid optimizer setting')
    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        raise ValueError('--stop-after-steps must be positive')
    if args.stop_at_step is not None and not 0 < args.stop_at_step <= args.max_steps:
        raise ValueError('--stop-at-step must be in (0, max_steps]')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA could not initialize with PyTorch {} (CUDA build {}). '
                           'Check the CUDA warning above and the allocated GPU driver; '
                           'this can also fail inside a GPU allocation. '
                           'No checkpoint has been loaded or modified.'.format(
                               torch.__version__, torch.version.cuda))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    output = Path(args.data_dir)
    output.mkdir(parents=True, exist_ok=True)
    resume = output / 'latest.pth' if args.resume == 'latest' else Path(args.resume)
    checkpoint = None
    if args.resume != 'none' and resume.is_file():
        checkpoint = read_checkpoint(resume)
        if checkpoint.get('data_protocol') != data_protocol:
            raise ValueError('Training/selection data changed; never resume a probe as a full run')
        fixed = ('max_steps', 'batch_size', 'accumulation_steps', 'crop_size', 'filter_multiplier',
                 'base_lr', 'weight_decay', 'warmup_steps', 'power', 'drop_path_prob', 'seed', 'no_amp')
        changed = [key for key in fixed if vars(args)[key] != checkpoint['train_config'][key]]
        if args.bn_momentum != checkpoint['train_config'].get('bn_momentum', 0.0003):
            changed.append('bn_momentum')
        if args.selection_calibration_batches != checkpoint['train_config'].get('selection_calibration_batches', 0):
            changed.append('selection_calibration_batches')
        if changed:
            raise ValueError('Resume changes the experiment: {}. Use the saved settings.'.format(', '.join(changed)))
    elif args.resume not in ('none', 'latest'):
        raise FileNotFoundError(resume)
    elif (output / 'config.json').exists():
        raise ValueError('Output directory already has an experiment but no usable resume was selected. '
                         'Choose a fresh --data-dir to avoid mixing runs.')
    generator = torch.Generator()
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker,
                        generator=generator)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers,
                            pin_memory=True, worker_init_fn=seed_worker)
    if not len(loader):
        raise ValueError('Training loader is empty')
    calibration_generator = torch.Generator()
    calibration_loader = None
    if data_protocol:
        calibration_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                        num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker,
                                        generator=calibration_generator)
        if args.selection_calibration_batches > len(calibration_loader):
            raise ValueError('Calibration count exceeds the probe training split')
    architecture = {}
    if args.architecture_json:
        artifact = json.loads(Path(args.architecture_json).read_text())
        if artifact.get('format') != 'autodeeplab_cell_v1' or artifact.get('smoke_test', False):
            raise ValueError('Need a completed non-smoke corrected search architecture')
        architecture = dict(cell=artifact['cell'], backbone=artifact['backbone'])
        if not checkpoint:
            (output / 'search_architecture.json').write_text(json.dumps(artifact, indent=2) + '\n')
    elif checkpoint:
        architecture = {k: checkpoint['model_config'][k] for k in ('cell', 'backbone')
                        if k in checkpoint['model_config']}
    model = ReferenceAutoDeepLab(args.filter_multiplier, drop_path_prob=args.drop_path_prob,
                                 bn_momentum=args.bn_momentum, **architecture).to(device)
    if checkpoint and model.model_config != checkpoint['model_config']:
        raise ValueError('Architecture changed on resume')
    optimizer = torch.optim.SGD(parameter_groups(model, args.weight_decay), lr=args.base_lr,
                                momentum=0.9, nesterov=False)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp, init_scale=1024.)
    step, epoch, batch_in_epoch, best = 0, 0, 0, -1.0
    runtime = dict(torch_version=str(torch.__version__), cuda_build=torch.version.cuda,
                   cudnn_version=torch.backends.cudnn.version())
    resume_events = list(checkpoint.get('resume_events', [])) if checkpoint else []
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        step, epoch, batch_in_epoch = checkpoint['step'], checkpoint['epoch'], checkpoint['batch_in_epoch']
        best = checkpoint['best_miou']
        torch.set_rng_state(checkpoint['torch_rng'])
        migrations = restore_cuda_rng_states(checkpoint['cuda_rng'])
        if migrations:
            resume_events.append(dict(step=step, event='cuda_rng_format_conversion',
                                      devices=migrations, source_runtime=checkpoint.get('runtime'),
                                      destination_runtime=runtime))
        random.setstate(checkpoint['python_rng'])
        numpy_rng = checkpoint['numpy_rng']
        np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32),
                             numpy_rng[2], numpy_rng[3], numpy_rng[4]))
        print('Resumed optimizer, AMP, progress and weights from {} at update {}'.format(resume, step), flush=True)
    config = dict(vars(args), model_version=MODEL_VERSION, model_config=model.model_config,
                  data_protocol=data_protocol, evaluation_split='train_selection' if data_protocol else 'val',
                  parameters=sum(p.numel() for p in model.parameters()), train_images=len(train_set),
                  validation_images=len(val_set), effective_batch=args.batch_size * args.accumulation_steps,
                  batches_per_epoch=len(loader),
                  scheduled_data_epochs=args.max_steps * args.accumulation_steps / len(loader),
                  runtime=runtime, resume_events=resume_events)
    if not checkpoint:
        (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    printed_config = dict(config)
    if data_protocol:
        printed_config['data_protocol'] = dict(format=data_protocol['format'],
                                               fingerprint=data_protocol['fingerprint'],
                                               training_images=len(train_set), selection_images=len(val_set))
    print(json.dumps(printed_config, indent=2), flush=True)
    for i, group in enumerate(optimizer.param_groups):
        print('Optimizer group {}: {} parameters, weight_decay={}'.format(
            i, sum(p.numel() for p in group['params']), group['weight_decay']), flush=True)
    if args.accumulation_steps > 1:
        print('Gradient accumulation preserves the effective SGD batch, but BN uses the physical batch.', flush=True)
    requested_stop = [False]
    def stop_handler(signum, frame):
        requested_stop[0] = True
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, stop_handler)

    def save(path):
        numpy_rng = np.random.get_state()
        atomic_save(dict(model_version=MODEL_VERSION, model_config=model.model_config,
                         data_protocol=data_protocol,
                         runtime=runtime, resume_events=resume_events,
                         train_config=vars(args), state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                         scaler=scaler.state_dict(), step=step, epoch=epoch, batch_in_epoch=batch_in_epoch,
                         best_miou=best, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                         python_rng=random.getstate(), numpy_rng=(numpy_rng[0], numpy_rng[1].tolist(),
                                                                 numpy_rng[2], numpy_rng[3], numpy_rng[4])), path)

    def evaluate_and_record():
        nonlocal best
        save(output / 'latest.pth')
        if data_protocol:
            # A fixed calibration panel for every candidate/checkpoint, without
            # consuming the training RNG stream (including workers=0).
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    random.seed(10000)
                    np.random.seed(10000)
                    torch.manual_seed(10000)
                    calibration_generator.manual_seed(10000)
                    with calibrated_batch_norm(model, calibration_loader, device,
                                               args.selection_calibration_batches):
                        result = validate(model, val_loader, device, amp=not args.no_amp)
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
        else:
            result = validate(model, val_loader, device, amp=not args.no_amp)
        result.update(step=step, scales=[1.0], flip=False,
                      evaluation_split='train_selection' if data_protocol else 'val',
                      calibration_batches=args.selection_calibration_batches)
        with (output / 'validation.jsonl').open('a') as stream:
            stream.write(json.dumps(result) + '\n')
        print('{} update {}: mIoU {:.3f}% ({} images)'.format(
            'Held-out TRAIN selection' if data_protocol else 'Validation', step,
            result['miou'], result['images']), flush=True)
        if result['miou'] > best:
            best = result['miou']
            save(output / 'best.pth')
        model.train()
        save(output / 'latest.pth')

    # Complete validation if the job ended after the pre-validation checkpoint.
    if step and (step % args.val_every == 0 or step == args.max_steps or step == args.stop_at_step):
        log = output / 'validation.jsonl'
        recorded = {json.loads(line)['step'] for line in log.read_text().splitlines()} if log.exists() else set()
        if step not in recorded:
            evaluate_and_record()
    if args.stop_at_step and step >= args.stop_at_step:
        print('Stage already complete at update {}; no further optimization.'.format(step), flush=True)
        return

    # Reproduce epoch shuffling on restart. Worker augmentation sequences are recreated
    # by replaying consumed batches; bitwise equivalence is not promised (cuDNN/AMP).
    generator.manual_seed(args.seed + epoch)
    iterator = iter(loader)
    for _ in range(batch_in_epoch):
        next(iterator)
    initial_step, started = step, time.monotonic()
    model.train()
    while step < args.max_steps:
        if requested_stop[0]:
            save(output / 'latest.pth')
            print('Saved latest.pth at update {}; resubmit the same command to continue.'.format(step), flush=True)
            return
        lr = learning_rate(step, args.max_steps, args.base_lr, args.warmup_steps, args.power)
        set_learning_rate(optimizer, lr)
        model.set_progress(step, args.max_steps)
        samples = []
        for _ in range(args.accumulation_steps):
            try:
                sample = next(iterator)
            except StopIteration:
                epoch += 1
                batch_in_epoch = 0
                generator.manual_seed(args.seed + epoch)
                iterator = iter(loader)
                sample = next(iterator)
            batch_in_epoch += 1
            samples.append(sample)
        total_loss = train_update(model, optimizer, scaler, samples, device, amp=not args.no_amp)
        del samples, sample
        step += 1
        if step == initial_step + 1 or step % args.log_every == 0:
            seconds = (time.monotonic() - started) / (step - initial_step)
            print('update {}/{} epoch {:.2f} loss {:.4f} lr {:.7f} sec/update {:.2f} estimated days left {:.2f}'.format(
                step, args.max_steps, epoch + batch_in_epoch / len(loader), total_loss, lr, seconds,
                seconds * (args.max_steps - step) / 86400), flush=True)
        validate_now = step % args.val_every == 0 or step == args.max_steps or step == args.stop_at_step
        if validate_now and not requested_stop[0]:
            evaluate_and_record()
        if step % args.save_every == 0 or validate_now:
            save(output / 'latest.pth')
        if step % args.snapshot_every == 0:
            save(output / 'step_{}.pth'.format(step))
        if step == args.stop_at_step:
            print('Stage complete at update {}; LR horizon remains {}.'.format(step, args.max_steps), flush=True)
            return
        if args.stop_after_steps and step - initial_step >= args.stop_after_steps:
            save(output / 'latest.pth')
            print('Requested short run complete; resume with the same schedule.', flush=True)
            return
    save(output / 'latest.pth')
    print('Training complete. Best single-scale validation mIoU: {:.3f}%'.format(best), flush=True)


if __name__ == '__main__':
    main()
