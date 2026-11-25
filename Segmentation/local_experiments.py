"""Prepare local-search finalists and compare independent TRAIN-only probes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from local_search import LOCAL_VERSION, cell_distance
from retrain_model.reference_model import BACKBONE, CELL, validate_architecture
from search_policy import architecture_key
from search_supernet import final_parameter_count


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Refusing to replace an existing experiment: ' + str(path))
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def manifest_id(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('format') != 'adl_local_finalists_v1':
        raise ValueError('Expected a local independent-probe manifest')
    split = json.loads((path.parent / manifest['split']).read_text())
    if manifest_id(split) != manifest['split_id']:
        raise ValueError('Prepared probe partition changed')
    if [c['id'] for c in manifest['candidates']] != list(range(len(manifest['candidates']))):
        raise ValueError('Noncontiguous candidate IDs')
    if architecture_key(manifest['candidates'][0]['cell']) != architecture_key(CELL):
        raise ValueError('Candidate 0 must be the published reference')
    for candidate in manifest['candidates']:
        validate_architecture(candidate['cell'], BACKBONE)
        artifact = json.loads((path.parent / candidate['architecture']).read_text())
        if (artifact['cell'] != candidate['cell'] or artifact.get('smoke_test')
                or artifact.get('format') != 'autodeeplab_cell_v1' or artifact['backbone'] != list(BACKBONE)):
            raise ValueError('Candidate architecture changed')
    return path, manifest


def prepare(ranking_path, search_dir, output, challengers=4):
    if challengers < 1:
        raise ValueError('Need at least one challenger')
    ranking_path, search_dir, output = Path(ranking_path), Path(search_dir), Path(output)
    ranking = json.loads(ranking_path.read_text())
    config = json.loads((search_dir / 'config.json').read_text())
    split = json.loads((search_dir / 'splits.json').read_text())
    if ranking.get('search_version') != LOCAL_VERSION or ranking.get('smoke_test'):
        raise ValueError('Use completed, non-smoke local-search re-ranking')
    if ranking.get('search_dirs') != [str(search_dir.resolve())]:
        raise ValueError('Prepare each search seed separately; ranking directory mismatch')
    if ranking['fingerprint'] != split['fingerprint'] or config['target_filters'] != 20:
        raise ValueError('Expected the matching dataset and F20 target')
    unique = {architecture_key(r['cell']): r for r in ranking['ranking']}
    ref = unique.pop(architecture_key(CELL))
    pool = sorted(unique.values(), key=lambda r: r['miou'], reverse=True)
    if len(pool) < challengers:
        raise ValueError('Not enough unique challengers')
    chosen = [ref, pool.pop(0)]
    # Include the risk-reward leader as well as the mean leader if they differ.
    if len(chosen) < challengers + 1 and pool:
        leader = max(pool, key=lambda r: r['reward'])
        chosen.append(leader)
        pool.remove(leader)
    while len(chosen) < challengers + 1:
        diverse = [r for r in pool if all(cell_distance(r['cell'], c['cell']) >= 2 for c in chosen[1:])]
        candidate = (diverse or pool)[0]
        chosen.append(candidate)
        pool.remove(candidate)
    candidates = []
    for index, row in enumerate(chosen):
        name = 'candidate_{}.json'.format(index)
        artifact = dict(format='autodeeplab_cell_v1', search_version=LOCAL_VERSION,
                        cell=row['cell'], backbone=list(BACKBONE), smoke_test=False,
                        is_reference=index == 0, stage='independent_probe_candidate',
                        source_ranking=str(ranking_path.resolve()))
        write_once(output / name, artifact)
        candidates.append(dict(id=index, architecture=name, cell=row['cell'],
                               parameters=final_parameter_count(row['cell'], 20),
                               proxy={k:row[k] for k in ('miou', 'image_cvar', 'reward')}))
    indices = split['indices']
    probe_split = dict(format='adl_probe_split_v1', fingerprint=split['fingerprint'],
                       train_indices=sorted(indices['weights'] + indices['controller']),
                       selection_indices=indices['selection'])
    write_once(output / 'probe_split.json', probe_split)
    manifest = dict(format='adl_local_finalists_v1', search_version=LOCAL_VERSION,
                    source_search=str(search_dir.resolve()), dataset_root=config['dataset_root'],
                    split='probe_split.json', split_id=manifest_id(probe_split),
                    fingerprint=split['fingerprint'], candidates=candidates,
                    recipe=dict(filter_multiplier=20, max_steps=500000, batch_size=16,
                                crop_size=769, base_lr=.05, weight_decay=.0001, warmup_steps=1000,
                                power=.9, drop_path_prob=0., bn_momentum=.01, val_every=10000,
                                selection_calibration_batches=32))
    write_once(output / 'manifest.json', manifest)
    print('Prepared reference + {} challengers: {}'.format(challengers, output / 'manifest.json'))


def stage_metrics(rows, stage, interval=10000):
    by_step = {r['step']: r for r in rows}
    steps = [stage - 2*interval, stage - interval, stage]
    if min(steps) <= 0 or any(s not in by_step for s in steps):
        raise ValueError('Stage needs completed evaluations at {}'.format(steps))
    selected = [by_step[s] for s in steps]
    for row in selected:
        if row.get('evaluation_split') != 'train_selection' or row.get('calibration_batches') != 32:
            raise ValueError('Ranking requires train-only selection with 32-batch BN calibration')
        if not all(math.isfinite(row[k]) for k in ('miou', 'image_cvar')):
            raise ValueError('Nonfinite probe score')
    return {key: sum(r[key] for r in selected) / 3 for key in ('miou', 'image_cvar')}


def rank_probes(manifest_path, seeds, ids, stage, margin=.5):
    path, manifest = load_manifest(manifest_path)
    if (not seeds or len(set(seeds)) != len(seeds) or len(set(ids)) != len(ids) or 0 not in ids
            or any(i < 0 or i >= len(manifest['candidates']) for i in ids)):
        raise ValueError('Need unique seeds/candidates including reference 0')
    if margin < 0 or not math.isfinite(margin):
        raise ValueError('Margin must be finite and nonnegative')
    results, shared_protocol = [], None
    for index in ids:
        candidate = manifest['candidates'][index]
        metrics = []
        for seed in seeds:
            directory = path.parent / 'probes' / 'candidate_{}_seed{}'.format(index, seed)
            config = json.loads((directory / 'config.json').read_text())
            binding = json.loads((directory / 'probe_binding.json').read_text())
            if binding != dict(manifest_id=manifest_id(manifest), candidate=index, seed=seed):
                raise ValueError('Probe belongs to another manifest or seed')
            if any(config[k] != v for k, v in manifest['recipe'].items()) or config['no_amp']:
                raise ValueError('Probe recipe differs from the common protocol')
            if config['seed'] != seed or config['evaluation_split'] != 'train_selection':
                raise ValueError('Probe seed/split mismatch')
            protocol = config['data_protocol']
            if protocol['fingerprint'] != manifest['fingerprint']:
                raise ValueError('Probe dataset fingerprint differs')
            if shared_protocol is None:
                shared_protocol = protocol
            if protocol != shared_protocol:
                raise ValueError('Probes used different train/selection images')
            if architecture_key(config['model_config'].get('cell', CELL)) != architecture_key(candidate['cell']):
                raise ValueError('Probe trained a different cell')
            if list(config['model_config'].get('backbone', BACKBONE)) != list(BACKBONE):
                raise ValueError('Probe used a different macro path')
            rows = [json.loads(line) for line in (directory / 'validation.jsonl').read_text().splitlines()]
            metrics.append(dict(seed=seed, **stage_metrics(rows, stage)))
        results.append(dict(id=index, per_seed=metrics,
                            miou=sum(r['miou'] for r in metrics)/len(metrics),
                            image_cvar=sum(r['image_cvar'] for r in metrics)/len(metrics)))
    reference = next(r for r in results if r['id'] == 0)
    for row in results:
        row['delta_miou'] = row['miou'] - reference['miou']
        row['per_seed_delta'] = [r['miou']-b['miou'] for r,b in zip(row['per_seed'], reference['per_seed'])]
    results.sort(key=lambda r: r['miou'], reverse=True)
    eligible = [r for r in results if r['id'] != 0 and r['delta_miou'] >= margin
                and all(d > 0 for d in r['per_seed_delta'])]
    winner = eligible[0] if eligible else reference
    return dict(format='adl_probe_report_v1', manifest_id=manifest_id(manifest), stage_step=stage,
                seeds=seeds, candidate_ids=ids, margin_points=margin, ranking=results,
                suggested_next_ids=[0]+[r['id'] for r in results if r['id'] != 0][:2],
                export_ready=stage >= 150000 and len(seeds) >= 2,
                winner_id=winner['id'], challenger_qualified=winner['id'] != 0,
                interpretation='Screening evidence on held-out TRAIN, not final validation or statistical significance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--ranking', required=True)
    p.add_argument('--search-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--challengers', type=int, default=4)
    p = sub.add_parser('rank')
    p.add_argument('--manifest', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--candidate-ids', nargs='+', type=int, default=[0,1,2,3,4])
    p.add_argument('--stage-step', type=int, required=True)
    p.add_argument('--margin', type=float, default=.5)
    p.add_argument('--export', default=None)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.ranking, args.search_dir, args.output, args.challengers)
    else:
        report = rank_probes(args.manifest, args.seeds, args.candidate_ids, args.stage_step, args.margin)
        path, manifest = load_manifest(args.manifest)
        name = 'report_{}_seeds_{}_ids_{}.json'.format(args.stage_step, '_'.join(map(str,args.seeds)),
                                                       '_'.join(map(str,args.candidate_ids)))
        (path.parent / name).write_text(json.dumps(report, indent=2) + '\n')
        for row in report['ranking']:
            print('Candidate {}: mean {:.3f}%, tail {:.3f}%, delta {:+.3f} points; per-seed {}'.format(
                row['id'], row['miou'], row['image_cvar'], row['delta_miou'], row['per_seed_delta']))
        print('Suggested next indices:', ','.join(map(str, report['suggested_next_ids'])))
        if args.export:
            if not report['export_ready']:
                raise ValueError('Export requires >=150k updates and >=2 independent training seeds')
            cell = manifest['candidates'][report['winner_id']]['cell']
            artifact = dict(format='autodeeplab_cell_v1', search_version=LOCAL_VERSION,
                            cell=cell, backbone=list(BACKBONE), smoke_test=False,
                            dataset_root=manifest['dataset_root'],
                            is_reference=report['winner_id']==0, independent_probe_evidence=report)
            write_once(args.export, artifact)
            print('Export:', args.export, 'Reference retained:', artifact['is_reference'])


if __name__ == '__main__':
    main()
