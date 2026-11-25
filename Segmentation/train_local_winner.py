"""Fresh full-data F20 training after independent-probe selection."""
import argparse
import json
from pathlib import Path

from train_baseline import main as train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--architecture-json', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    artifact = json.loads(Path(args.architecture_json).read_text())
    evidence = artifact.get('independent_probe_evidence', {})
    if (not evidence.get('export_ready') or evidence.get('stage_step', 0) < 150000
            or len(set(evidence.get('seeds', []))) < 2):
        raise ValueError('Complete the independent probe selection before full retraining')
    if artifact.get('is_reference') or not evidence.get('challenger_qualified'):
        raise ValueError('No challenger qualified. Reference retained; no automatic full retrain is justified.')
    train(['--data-dir', args.data_dir, '--architecture-json', args.architecture_json,
           '--dataset-root', artifact['dataset_root'],
           '--seed', str(args.seed), '--workers', str(args.workers), '--max-steps', '500000',
           '--filter-multiplier', '20', '--batch-size', '16', '--crop-size', '769',
           '--base-lr', '.05', '--weight-decay', '.0001', '--warmup-steps', '1000',
           '--power', '.9', '--drop-path-prob', '0', '--bn-momentum', '.01', '--resume', 'latest'])


if __name__ == '__main__':
    main()
