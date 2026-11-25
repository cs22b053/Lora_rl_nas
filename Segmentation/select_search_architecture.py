"""Re-rank finalists across independent supernets on one common held-out panel."""
import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from retrain_data import CityscapesFine
from retrain_model.reference_model import BACKBONE, CELL
from search_data import SearchPanel, dataset_fingerprint, split_indices
from search_policy import architecture_key
from search_scoring import score_candidate
from search_supernet import SUPPORTED_SEARCH_VERSIONS, SearchSupernet
from local_search import LOCAL_VERSION, reference_neighborhood


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--search-dirs', nargs='+', required=True)
    p.add_argument('--output', default='architecture_consensus.json')
    p.add_argument('--top-per-run', type=int, default=3)
    p.add_argument('--all-visited', action='store_true',
                   help='Re-rank every valid visited cell, plus all final candidates and the reference')
    p.add_argument('--rank-images', type=int, default=None, help='Override held-out panel size, up to the split size')
    p.add_argument('--calibration-batches', type=int, default=None)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if args.top_per_run < 1 or args.workers < 0:
        p.error('Invalid candidate count/workers')
    if any(v is not None and v < 1 for v in (args.rank_images, args.calibration_batches)):
        p.error('Panel/calibration sizes must be positive')
    configs, versions, candidates = [], [], {architecture_key(CELL): [list(e) for e in CELL]}
    for directory in map(Path, args.search_dirs):
        config = json.loads((directory / 'config.json').read_text())
        config.setdefault('train_short_side', 0)
        artifact = json.loads((directory / 'architecture.json').read_text())
        if artifact.get('search_version') not in (*SUPPORTED_SEARCH_VERSIONS, LOCAL_VERSION) or artifact.get('smoke_test'):
            raise ValueError('Only completed, non-smoke corrected searches can be combined')
        versions.append(artifact['search_version'])
        configs.append(config)
        candidates[architecture_key(artifact['cell'])] = artifact['cell']
        ranking = json.loads((directory / 'ranking.json').read_text())
        if config.get('search_space') == 'reference-local':
            cells = reference_neighborhood(config['local_candidates'], config['candidate_seed'],
                                           config['target_filters'], config['max_final_params'])
            saved = json.loads((directory / 'candidate_library.json').read_text())
            if saved != cells:
                raise ValueError('Local library differs from the search protocol')
            for cell in cells:
                candidates[architecture_key(cell)] = cell
        if args.all_visited:
            for line in (directory / 'controller.jsonl').read_text().splitlines():
                ranking.extend(result for result in json.loads(line)['candidates'] if not result['over_budget'])
        else:
            ranking = sorted(ranking, key=lambda r: r['reward'], reverse=True)[:args.top_per_run]
        for result in ranking:
            candidates[architecture_key(result['cell'])] = result['cell']
    config = configs[0]
    if len(set(versions)) != 1:
        raise ValueError('Do not combine different search protocol versions')
    search_version = versions[0]
    mutable = {'seed', 'output_dir', 'workers', 'resume', 'save_every', 'stop_after_steps'}
    fixed = set(config) - mutable
    if any(c[k] != config[k] for c in configs for k in fixed):
        raise ValueError('Combine independent seeds of the SAME experiment, not different ablations/recipes')
    if len({c['seed'] for c in configs}) != len(configs):
        raise ValueError('Repeated search seeds would duplicate evidence')
    if not torch.cuda.is_available():
        raise RuntimeError('Submit run_select_search.sh on a GPU')
    device = torch.device('cuda')
    raw = CityscapesFine(config['dataset_root'], 'train', training=False)
    splits = split_indices(len(raw), config['split_seed'])
    fingerprint = dataset_fingerprint(raw)
    rank_images = args.rank_images or config['rank_images']
    calibration_batches = args.calibration_batches or config['calibration_batches']
    if rank_images > len(splits['selection']) or calibration_batches*config['batch_size'] > len(splits['weights']):
        raise ValueError('Requested image count exceeds its disjoint split')
    if rank_images*config['cvar_alpha'] < 8:
        raise ValueError('Need at least eight images in the CVaR tail')
    def panel(split, count, seed):
        indices = random.Random(seed).sample(splits[split], count)
        return list(DataLoader(SearchPanel(raw, indices, config['crop_size'], seed,
                                           short_side=config['train_short_side'] or 513),
                               batch_size=config['batch_size'], num_workers=args.workers,
                               shuffle=False, pin_memory=True))
    calibration = panel('weights', calibration_batches*config['batch_size'], config['split_seed']+10000)
    selection = panel('selection', rank_images, config['split_seed']+30000)
    print('Re-ranking {} unique cells across {} supernet(s), using {} held-out images and {} calibration images.'.format(
        len(candidates), len(configs), rank_images, calibration_batches*config['batch_size']), flush=True)
    scores = {key: [] for key in candidates}
    for directory in map(Path, args.search_dirs):
        checkpoint = torch.load(directory / 'latest.pth', map_location='cpu', weights_only=True)
        if (checkpoint.get('search_version') != search_version or checkpoint['fingerprint'] != fingerprint
                or checkpoint['splits'] != splits or checkpoint['epoch'] != config['epochs']):
            raise ValueError('Checkpoint is incomplete or its split changed')
        if config.get('search_space') == 'reference-local' and checkpoint.get('local_cells') != cells:
            raise ValueError('Checkpoint local library differs from the ranking protocol')
        model = SearchSupernet(config['filter_multiplier'], config['lora_rank']).to(device)
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        for index, (key, cell) in enumerate(candidates.items()):
            result = score_candidate(model, cell, calibration, selection, device,
                                      config['cvar_alpha'], config['risk_weight'], amp=not config['no_amp'])
            scores[key].append(result)
            print('{} candidate {}/{} proxy mean {:.3f}%'.format(directory, index+1,
                  len(candidates), result['miou']*100), flush=True)
        del model
        torch.cuda.empty_cache()
    results = []
    for key, cell in candidates.items():
        values = scores[key]
        result = dict(cell=cell, per_seed=values)
        result.update({metric: sum(v[metric] for v in values)/len(values)
                       for metric in ('miou', 'reward', 'image_cvar')})
        results.append(result)
    reference = next(r for r in results if architecture_key(r['cell']) == architecture_key(CELL))
    winner = max((r for r in results if r['miou'] >= reference['miou']), key=lambda r: r['reward'])
    artifact = dict(format='autodeeplab_cell_v1', search_version=search_version, cell=winner['cell'],
                    backbone=list(BACKBONE), smoke_test=False, fingerprint=fingerprint,
                    search_dirs=[str(Path(d).resolve()) for d in args.search_dirs],
                    search_seeds=[c['seed'] for c in configs], selection=winner, reference=reference,
                    selection_split='common held-out subset of Cityscapes train',
                    ranking_protocol=dict(all_visited=args.all_visited, rank_images=rank_images,
                                          calibration_batches=calibration_batches,
                                          short_side=config['train_short_side'] or 513,
                                          crop_size=config['crop_size']),
                    risk_weight=config['risk_weight'], cvar_alpha=config['cvar_alpha'],
                    is_reference=architecture_key(winner['cell']) == architecture_key(CELL), ranking=results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(artifact, indent=2) + '\n')
    temporary.replace(output)
    print('Consensus architecture: {}; reference selected={}. Proxy ranking only; full retraining remains necessary.'.format(
        output, artifact['is_reference']), flush=True)


if __name__ == '__main__':
    main()
