"""From-scratch cell NAS: single-path weights, input LoRA, measured CVaR rewards."""
import argparse
import json
import math
import random
import signal
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from mypath import Path as DatasetPath
from reference_training import atomic_save, restore_cuda_rng_states
from retrain_data import CityscapesFine, seed_worker
from retrain_model.reference_model import BACKBONE, CELL
from search_data import SearchPanel, dataset_fingerprint, split_indices
from search_policy import CellPolicy, architecture_key, decode_action, ppo_update
from search_scoring import score_candidate
from search_optimization import update_weights
from search_supernet import SEARCH_VERSION, SearchSupernet, final_parameter_count, optimizer_groups
from local_search import LOCAL_VERSION, LocalCellPolicy, reference_neighborhood


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', default=DatasetPath.db_root_dir('cityscapes'))
    p.add_argument('--output-dir', default='search_lora_cvar_seed1')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--warmup-epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=6)
    p.add_argument('--crop-size', type=int, default=321)
    p.add_argument('--train-short-side', type=int, default=512,
                   help='Resize before search crops; 0 reproduces legacy full-image scale augmentation')
    p.add_argument('--filter-multiplier', type=int, default=8)
    p.add_argument('--lora-rank', type=int, default=4, help='0 disables LoRA for an ablation')
    p.add_argument('--lora-lr-multiplier', type=float, default=2.)
    p.add_argument('--lr', type=float, default=.025)
    p.add_argument('--min-lr', type=float, default=.001)
    p.add_argument('--weight-decay', type=float, default=.0003)
    p.add_argument('--policy-lr', type=float, default=.0003)
    p.add_argument('--max-skips', type=int, default=2)
    p.add_argument('--risk-weight', type=float, default=.2, help='0 is mean-IoU PPO ablation')
    p.add_argument('--cvar-alpha', type=float, default=.2, help='Lower tail fraction of image scores')
    p.add_argument('--rollout-size', type=int, default=16)
    p.add_argument('--ppo-epochs', type=int, default=4)
    p.add_argument('--reward-images', type=int, default=64)
    p.add_argument('--rank-images', type=int, default=128)
    p.add_argument('--calibration-batches', type=int, default=8)
    p.add_argument('--final-candidates', type=int, default=16)
    p.add_argument('--max-final-params', type=int, default=24000000, help='Parameter cap at --target-filters')
    p.add_argument('--target-filters', type=int, default=32)
    p.add_argument('--search-space', choices=['global', 'reference-local'], default='global')
    p.add_argument('--local-candidates', type=int, default=128)
    p.add_argument('--candidate-seed', type=int, default=0, help='Fixed library seed; keep across search ablations')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--resume', default='latest')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--random-search', action='store_true', help='Same supernet/scoring budget, no PPO updates')
    p.add_argument('--steps-per-epoch', type=int, default=None, help='For short hardware checks only')
    p.add_argument('--stop-after-steps', type=int, default=None, help='Pause this invocation after N weight updates')
    p.add_argument('--smoke-test', action='store_true', help='Tiny isolated GPU run exercising all stages')
    return p


def main():
    args = parser().parse_args()
    if not 0 < args.cvar_alpha <= 1:
        raise ValueError('CVaR alpha must be in (0, 1]')
    if args.smoke_test:
        args.output_dir += '_smoke'
        args.epochs, args.warmup_epochs, args.steps_per_epoch = 2, 0, 2
        args.batch_size, args.crop_size, args.filter_multiplier = 2, 65, 2
        args.rollout_size, args.final_candidates, args.calibration_batches = 4, 4, 2
        args.reward_images = args.rank_images = max(40, int(8 / args.cvar_alpha + 1))
    for key in ('epochs', 'batch_size', 'crop_size', 'filter_multiplier', 'rollout_size', 'ppo_epochs',
                'reward_images', 'rank_images', 'calibration_batches', 'final_candidates', 'save_every'):
        if getattr(args, key) < 1:
            raise ValueError('{} must be positive'.format(key))
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError('Warmup must finish before search ends: otherwise PPO never runs')
    if not 0 < args.cvar_alpha <= 1 or not 0 <= args.risk_weight <= 1:
        raise ValueError('Invalid CVaR settings')
    if min(args.reward_images, args.rank_images)*args.cvar_alpha < 8:
        raise ValueError('Use enough reward/ranking images to have >=8 images in the CVaR tail')
    if args.batch_size < 2 or args.rollout_size < 4 or args.final_candidates < 2:
        raise ValueError('Require batch>=2, rollout>=4 and final candidates>=2')
    if args.workers < 0 or args.lora_rank < 0 or not 0 <= args.max_skips <= 10:
        raise ValueError('Invalid workers/rank/skip cap')
    if args.train_short_side < 0 or (args.train_short_side and args.crop_size > args.train_short_side):
        raise ValueError('Search short side must be zero or at least the crop size')
    if not 0 < args.min_lr <= args.lr or args.lora_lr_multiplier <= 0 or args.policy_lr <= 0:
        raise ValueError('Invalid learning rate')
    if args.target_filters < 1 or args.max_final_params < final_parameter_count(CELL, args.target_filters):
        raise ValueError('Parameter budget must include the reference at the target width')
    local_cells = (reference_neighborhood(args.local_candidates, args.candidate_seed,
                                         args.target_filters, args.max_final_params)
                   if args.search_space == 'reference-local' else None)
    search_version = LOCAL_VERSION if local_cells else SEARCH_VERSION
    if any(v is not None and v < 1 for v in (args.steps_per_epoch, args.stop_after_steps)):
        raise ValueError('Step limits must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('Submit run_search_ppo.sh on a GPU node; no CPU search is launched')
    device = torch.device('cuda')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    resume = output / 'latest.pth' if args.resume == 'latest' else Path(args.resume)
    if args.resume != 'none' and resume.is_file():
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
        if checkpoint.get('search_version') != search_version:
            raise ValueError('Legacy search checkpoints are incompatible; start in a fresh directory')
        mutable = {'workers', 'resume', 'output_dir', 'save_every', 'stop_after_steps'}
        saved_args = dict(checkpoint['args'])
        saved_args.setdefault('train_short_side', 0)
        for key, value in dict(target_filters=32, search_space='global', local_candidates=128, candidate_seed=0).items():
            saved_args.setdefault(key, value)
        changed = [k for k, v in vars(args).items() if k not in mutable and saved_args.get(k) != v]
        if changed:
            raise ValueError('Resume configuration mismatch: {}'.format(changed))
    elif args.resume not in ('none', 'latest'):
        raise FileNotFoundError(resume)
    elif (output / 'config.json').exists():
        raise ValueError('Existing experiment without a selected checkpoint; choose a fresh output directory')
    training = CityscapesFine(args.dataset_root, 'train', args.crop_size,
                              base_short_side=args.train_short_side or None,
                              random_scale=args.train_short_side == 0)
    raw = CityscapesFine(args.dataset_root, 'train', training=False)
    fingerprint = dataset_fingerprint(raw)
    splits = split_indices(len(training), args.split_seed)
    if max(args.reward_images, args.rank_images) > min(len(splits['controller']), len(splits['selection'])):
        raise ValueError('Panel size exceeds the disjoint split size')
    if args.calibration_batches * args.batch_size > len(splits['weights']):
        raise ValueError('Not enough weight-training images for calibration')
    generator = torch.Generator()
    loader = DataLoader(Subset(training, splits['weights']), batch_size=args.batch_size,
                        shuffle=True, drop_last=True, num_workers=args.workers, pin_memory=True,
                        worker_init_fn=seed_worker, generator=generator)
    batches = min(len(loader), args.steps_per_epoch or len(loader))
    if not batches:
        raise ValueError('Empty weight-training loader')
    model = SearchSupernet(args.filter_multiplier, args.lora_rank).to(device)
    policy = (LocalCellPolicy(local_cells) if local_cells else CellPolicy(args.max_skips)).to(device)
    optimizer = torch.optim.SGD(optimizer_groups(model, args.weight_decay, args.lora_lr_multiplier),
                                lr=args.lr, momentum=.9)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.policy_lr, eps=1e-5)
    # The default 65536 overflowed throughout the original four-step GPU smoke
    # test. Start more conservatively and retry skipped updates explicitly.
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp, init_scale=1024.)
    epoch, position, step, policy_updates, last_loss = 0, 0, 0, 0, 3.0
    last_candidates = []
    if checkpoint:
        if local_cells != checkpoint.get('local_cells'):
            raise ValueError('Local candidate library changed on resume')
        if checkpoint['fingerprint'] != fingerprint or checkpoint['splits'] != splits:
            raise ValueError('Training image list/split changed on resume')
        model.load_state_dict(checkpoint['model'], strict=True)
        policy.load_state_dict(checkpoint['policy'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        epoch, position, step = checkpoint['epoch'], checkpoint['position'], checkpoint['step']
        if epoch > 0 and step == 0:
            raise ValueError('Checkpoint completed epochs without any weight updates. '
                             'Use a fresh output directory for the corrected smoke test.')
        policy_updates, last_loss = checkpoint['policy_updates'], checkpoint['last_loss']
        last_candidates = checkpoint['last_candidates']
        torch.set_rng_state(checkpoint['torch_rng'])
        restore_cuda_rng_states(checkpoint['cuda_rng'])
        random.setstate(checkpoint['python_rng'])
    else:
        (output / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
        (output / 'splits.json').write_text(json.dumps(dict(fingerprint=fingerprint, indices=splits,
             images=[str(p.relative_to(raw.root)) for p in raw.images]), indent=2) + '\n')
        if local_cells:
            (output / 'candidate_library.json').write_text(json.dumps(local_cells, indent=2) + '\n')
    print('Search config: {}'.format(json.dumps(vars(args))), flush=True)
    print('Disjoint TRAIN splits: {}; official validation is unused.'.format(
        {k: len(v) for k,v in splits.items()}), flush=True)
    print('Supernet parameters: {}; LoRA parameters: {}'.format(sum(p.numel() for p in model.parameters()),
          sum(p.numel() for n,p in model.named_parameters() if 'lora_' in n)), flush=True)
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))

    def save():
        atomic_save(dict(search_version=search_version, local_cells=local_cells, args=vars(args), fingerprint=fingerprint,
                         splits=splits, model=model.state_dict(), policy=policy.state_dict(),
                         optimizer=optimizer.state_dict(), policy_optimizer=policy_optimizer.state_dict(),
                         scaler=scaler.state_dict(), epoch=epoch, position=position, step=step,
                         policy_updates=policy_updates, last_loss=last_loss, last_candidates=last_candidates,
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                         python_rng=random.getstate()), output / 'latest.pth')

    def panel(split, count, panel_seed):
        ids = random.Random(panel_seed).sample(splits[split], count)
        view = SearchPanel(raw, ids, args.crop_size, panel_seed,
                           short_side=args.train_short_side or 513)
        data = DataLoader(view, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                          pin_memory=True, generator=torch.Generator().manual_seed(panel_seed))
        # Cache identical CPU tensors for every architecture in this comparison.
        return list(data)

    calibration = panel('weights', args.calibration_batches * args.batch_size, args.split_seed + 10000)
    def evaluate(cell, data):
        params = final_parameter_count(cell, args.target_filters)
        if params > args.max_final_params:
            return dict(reward=-1., miou=0., image_cvar=0., parameters=params, over_budget=True)
        result = score_candidate(model, cell, calibration, data, device, args.cvar_alpha,
                                  args.risk_weight, amp=not args.no_amp)
        result.update(parameters=params, over_budget=False)
        return result

    initial_step, started = step, time.monotonic()
    while epoch < args.epochs:
        generator.manual_seed(args.seed + epoch)
        iterator = iter(loader)
        for _ in range(position):
            next(iterator)
        model.train()
        while position < batches:
            if stop[0]:
                save()
                return
            sample = next(iterator)
            progress = (epoch*batches + position) / (args.epochs*batches)
            lr = args.min_lr + (args.lr-args.min_lr) * .5 * (1 + math.cos(progress*math.pi))
            for group in optimizer.param_groups:
                group['lr'] = lr * group['lr_multiplier']
            context = torch.tensor([[epoch/max(1,args.epochs-1), lr/args.lr, min(last_loss/3., 2.)]], device=device)
            with torch.no_grad():
                if (epoch*batches + position) % 10 == 0:
                    cell = CELL  # Keep the reference candidate trained as a comparison anchor.
                else:
                    uniform = epoch < args.warmup_epochs or args.random_search or random.random() < .5
                    cell = decode_action(policy(context, uniform=uniform)[0][0])
            model.set_architecture(cell)
            image, target = sample['image'].to(device, non_blocking=True), sample['label'].to(device, non_blocking=True)
            update = update_weights(model, optimizer, scaler, image, target, amp=not args.no_amp)
            position += 1
            step += 1
            last_loss = update['loss']
            del sample, image, target
            if position % 25 == 0 or position == 1 or position == batches:
                print('epoch {}/{} batch {}/{} updates {} CE {:.4f} base LR {:.6f} LoRA LR {:.6f} AMP scale {:.1f}'.format(
                    epoch+1, args.epochs, position, batches, step, last_loss, lr,
                    lr*args.lora_lr_multiplier, update['amp_scale']), flush=True)
            if position % args.save_every == 0:
                save()
            if args.stop_after_steps and step-initial_step >= args.stop_after_steps:
                save()
                print('Paused; resubmit without --stop-after-steps to continue.', flush=True)
                return
        save()  # A restart during a rollout repeats that whole, stationary rollout.
        if epoch >= args.warmup_epochs:
            if step == 0:
                raise RuntimeError('Refusing to run PPO before any successful weight update')
            # Reconstruct this even when resuming at the end of an epoch.
            progress = (epoch*batches + batches - 1) / (args.epochs*batches)
            lr = args.min_lr + (args.lr-args.min_lr)*.5*(1 + math.cos(progress*math.pi))
            context = torch.tensor([[epoch/max(1,args.epochs-1), lr/args.lr, min(last_loss/3., 2.)]], device=device)
            reward_panel = panel('controller', args.reward_images, args.split_seed + 20000 + epoch)
            contexts = context.repeat(args.rollout_size, 1)
            with torch.no_grad():
                actions, old_lp, _, old_values = policy(contexts, uniform=args.random_search)
            rewards, candidates = [], []
            for index, action in enumerate(actions):
                if stop[0]:
                    save()
                    return
                cell = decode_action(action)
                result = evaluate(cell, reward_panel)
                rewards.append(result['reward'])
                candidates.append(dict(cell=cell, **result))
                print('Controller epoch {} candidate {}/{} mean {:.3f}% tail {:.3f}%'.format(
                    epoch+1, index+1, len(actions), result['miou']*100, result['image_cvar']*100), flush=True)
            metrics = {}
            if not args.random_search:
                metrics = ppo_update(policy, policy_optimizer, contexts, actions, old_lp, old_values,
                                     torch.tensor(rewards, device=device), epochs=args.ppo_epochs)
                policy_updates += 1
            last_candidates = sorted(candidates, key=lambda c: c['reward'], reverse=True)
            with (output / 'controller.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(epoch=epoch, policy_updates=policy_updates,
                                             metrics=metrics, candidates=candidates)) + '\n')
            del reward_panel
            print('PPO updates completed: {}'.format(policy_updates), flush=True)
        epoch, position = epoch + 1, 0
        save()
    if not args.random_search and policy_updates == 0:
        raise RuntimeError('Search ended without a PPO update')
    if args.smoke_test and step != args.epochs*batches:
        raise RuntimeError('Smoke test did not complete all required weight updates')
    if stop[0]:
        save()
        return
    # Re-evaluate all finalists at the SAME final weights on the unused selection split.
    candidates = {architecture_key(CELL): [list(edge) for edge in CELL]}
    for previous in last_candidates[:max(1, args.final_candidates//2)]:
        if not previous['over_budget']:
            candidates[architecture_key(previous['cell'])] = previous['cell']
    context = torch.tensor([[1., args.min_lr/args.lr, min(last_loss/3., 2.)]], device=device)
    for _ in range(args.final_candidates * 50):
        if len(candidates) >= args.final_candidates:
            break
        with torch.no_grad():
            cell = decode_action(policy(context, uniform=args.random_search)[0][0])
        if final_parameter_count(cell, args.target_filters) <= args.max_final_params:
            candidates[architecture_key(cell)] = cell
    selection = panel('selection', args.rank_images, args.split_seed + 30000)
    ranked = []
    for cell in candidates.values():
        if stop[0]:
            save()
            return
        ranked.append(dict(cell=cell, **evaluate(cell, selection)))
    reference = next(r for r in ranked if architecture_key(r['cell']) == architecture_key(CELL))
    # A mean-IoU guard avoids choosing a lower-mean candidate solely for its tail.
    eligible = [r for r in ranked if r['miou'] >= reference['miou']]
    winner = max(eligible, key=lambda r: r['reward'])
    artifact = dict(format='autodeeplab_cell_v1', search_version=search_version,
                    cell=winner['cell'], backbone=list(BACKBONE), search_seed=args.seed,
                    split_seed=args.split_seed, fingerprint=fingerprint, smoke_test=args.smoke_test,
                    selection=winner, reference=reference, risk_weight=args.risk_weight,
                    cvar_alpha=args.cvar_alpha, selection_split='held-out subset of Cityscapes train',
                    is_reference=architecture_key(winner['cell']) == architecture_key(CELL))
    (output / 'ranking.json').write_text(json.dumps(ranked, indent=2) + '\n')
    temporary = output / 'architecture.json.tmp'
    temporary.write_text(json.dumps(artifact, indent=2) + '\n')
    temporary.replace(output / 'architecture.json')
    print('Successful weight updates: {}; PPO updates: {}'.format(step, policy_updates), flush=True)
    print('Search complete in {:.2f} hours. Export: {}. Proxy mean {:.3f}%, reference selected={}.'.format(
        (time.monotonic()-started)/3600, output / 'architecture.json', winner['miou']*100,
        artifact['is_reference']), flush=True)
    print('Proxy scores are NOT full-training Cityscapes validation mIoU. Retrain exported architecture from scratch.', flush=True)


if __name__ == '__main__':
    main()
