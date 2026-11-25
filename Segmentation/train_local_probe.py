"""Run one independently initialized finalist with a fixed, resumable stage boundary."""
import argparse

from local_experiments import load_manifest, manifest_id, write_once
from train_baseline import main as train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--candidate', type=int, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--stop-at-step', type=int, default=50000)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    path, manifest = load_manifest(args.manifest)
    if not 0 <= args.candidate < len(manifest['candidates']):
        p.error('Candidate index is outside the prepared shortlist')
    if args.stop_at_step not in (50000, 100000, 150000):
        p.error('Supported probe stage boundaries: 50000, 100000, 150000')
    candidate = manifest['candidates'][args.candidate]
    output = path.parent / 'probes' / 'candidate_{}_seed{}'.format(args.candidate, args.seed)
    write_once(output / 'probe_binding.json', dict(manifest_id=manifest_id(manifest),
                                                 candidate=args.candidate, seed=args.seed))
    argv = ['--dataset-root', manifest['dataset_root'], '--data-dir', str(output),
            '--architecture-json', str(path.parent / candidate['architecture']),
            '--selection-split-json', str(path.parent / manifest['split']),
            '--seed', str(args.seed), '--stop-at-step', str(args.stop_at_step),
            '--workers', str(args.workers), '--resume', 'latest']
    for key, value in manifest['recipe'].items():
        argv.extend(['--'+key.replace('_', '-'), str(value)])
    train(argv)


if __name__ == '__main__':
    main()
