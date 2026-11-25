"""Small CPU checks of local proposals, PPO replay, partitions and staged selection."""
import json
import tempfile
import unittest
import contextlib
import io
from pathlib import Path
from unittest.mock import patch

import torch

from local_search import LOCAL_OPS, LOCAL_VERSION, LocalCellPolicy, cell_distance, reference_neighborhood
from local_experiments import (load_manifest, manifest_id, prepare, rank_probes, stage_metrics,
                               write_once)
from probe_data import validate_probe_split
from retrain_model.reference_model import BACKBONE, CELL, validate_architecture
from search_policy import architecture_key, decode_action, ppo_update
from search_supernet import final_parameter_count
import train_baseline


class LocalWorkflowChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_neighborhood_budget_distance_and_reproducibility(self):
        cells = reference_neighborhood()
        self.assertEqual(cells, reference_neighborhood())
        self.assertEqual(len(cells), 128)
        self.assertEqual(len({architecture_key(c) for c in cells}), 128)
        self.assertEqual(architecture_key(cells[0]), architecture_key(CELL))
        self.assertTrue(any(cell_distance(c, CELL) == 2 for c in cells))
        for cell in cells:
            validate_architecture(cell, BACKBONE)
            self.assertLessEqual(cell_distance(cell, CELL), 2)
            self.assertLessEqual(final_parameter_count(cell, 20), 10147427)
            self.assertTrue(all(op in LOCAL_OPS for _, op in cell))
        swapped = [list(e) for e in CELL]
        swapped[0], swapped[1] = swapped[1], swapped[0]
        self.assertEqual(cell_distance(swapped, CELL), 0)

    def test_local_policy_replay_and_ppo(self):
        cells = reference_neighborhood(16)
        policy = LocalCellPolicy(cells)
        contexts = torch.zeros(32, 3)
        with torch.no_grad():
            actions, logp, entropy, values = policy(contexts)
            torch.testing.assert_close(logp, policy(contexts, actions)[1])
            torch.testing.assert_close(logp.exp(), torch.full((32,), 1/16))
        self.assertTrue(torch.isfinite(entropy).all())
        for action in actions:
            self.assertIn(architecture_key(decode_action(action)), {architecture_key(c) for c in cells})
        before = policy.actor.weight.clone()
        ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=.001), contexts, actions,
                   logp, values, torch.linspace(0, 1, 32), epochs=2)
        self.assertFalse(torch.equal(before, policy.actor.weight))
        invalid = actions.clone()
        invalid[0, 0] = 99
        with self.assertRaisesRegex(ValueError, 'outside'):
            policy(contexts, invalid)

    def test_probe_partition_rejects_leakage_and_changed_data(self):
        split = dict(format='adl_probe_split_v1', fingerprint='images',
                     train_indices=list(range(8)), selection_indices=[8,9])
        validate_probe_split(split, 'images', 10)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            validate_probe_split(dict(split, selection_indices=[7,8,9]), 'images', 10)
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            validate_probe_split(split, 'changed', 10)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            validate_probe_split(dict(split, train_indices=list(range(7))), 'images', 10)

    def make_experiment(self, root):
        search, output = root/'search', root/'finalists'
        search.mkdir()
        write_once(search/'config.json', dict(target_filters=20, dataset_root='/unused'))
        write_once(search/'splits.json', dict(fingerprint='images',
                                            indices=dict(weights=list(range(6)), controller=[6,7], selection=[8,9])))
        cells = reference_neighborhood(8)
        ranking = [dict(cell=c, miou=.3+i/100, image_cvar=.2, reward=.28+i/100) for i,c in enumerate(cells)]
        rank_path = root/'ranking.json'
        write_once(rank_path, dict(search_version=LOCAL_VERSION, fingerprint='images',
                                  search_dirs=[str(search.resolve())], ranking=ranking))
        prepare(rank_path, search, output)
        path, manifest = load_manifest(output/'manifest.json')
        protocol = validate_probe_split(json.loads((output/'probe_split.json').read_text()), 'images', 10)
        for candidate in manifest['candidates']:
            for seed in (0,1):
                directory = output/'probes'/f"candidate_{candidate['id']}_seed{seed}"
                write_once(directory/'probe_binding.json', dict(manifest_id=manifest_id(manifest),
                                                               candidate=candidate['id'], seed=seed))
                write_once(directory/'config.json', dict(manifest['recipe'], seed=seed, no_amp=False,
                           evaluation_split='train_selection', data_protocol=protocol,
                           model_config=dict(cell=candidate['cell'])))
                rows = [dict(step=s, miou=60.+candidate['id'], image_cvar=40.,
                             evaluation_split='train_selection', calibration_batches=32)
                        for s in (30000,40000,50000,130000,140000,150000)]
                (directory/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return path

    def test_prepare_and_stage_ranking(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.make_experiment(Path(temp))
            report = rank_probes(path, [0], [0,1,2,3,4], 50000)
            self.assertFalse(report['export_ready'])
            self.assertEqual(report['suggested_next_ids'], [0,4,3])
            report = rank_probes(path, [0,1], [0,4,3], 150000)
            self.assertTrue(report['export_ready'])
            self.assertEqual(report['winner_id'], 4)
            self.assertEqual(report['ranking'][0]['delta_miou'], 4.)
            with self.assertRaises(ValueError):
                rank_probes(path, [0,0], [0,4], 150000)
            with self.assertRaisesRegex(ValueError, 'completed evaluations'):
                rank_probes(path, [0], [0,4], 100000)

    def test_reference_retained_when_seed_disagrees(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.make_experiment(Path(temp))
            log = path.parent/'probes/candidate_1_seed1/validation.jsonl'
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            for row in rows:
                row['miou'] = 59.
            log.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            report = rank_probes(path, [0,1], [0,1], 150000)
            self.assertFalse(report['challenger_qualified'])
            self.assertEqual(report['winner_id'], 0)

    def test_official_val_and_nonfinite_scores_cannot_select_winner(self):
        rows = [dict(step=s, miou=60., image_cvar=40., evaluation_split='val', calibration_batches=32)
                for s in (30000,40000,50000)]
        with self.assertRaisesRegex(ValueError, 'train-only'):
            stage_metrics(rows, 50000)
        for row in rows:
            row.update(evaluation_split='train_selection', miou=float('nan'))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            stage_metrics(rows, 50000)

    def test_cpu_probe_resume_stage_and_no_official_validation(self):
        # Real tiny optimizer/checkpoint/BN-calibration loop on CPU; mock only
        # the GPU environment and data/model size. No Cityscapes images are read.
        class ToyData(torch.utils.data.Dataset):
            def __len__(self):
                return 8
            def __getitem__(self, index):
                return dict(image=torch.full((3,4,4), index/8.), label=torch.ones(4,4,dtype=torch.long))
        class ToyModel(torch.nn.Module):
            def __init__(self, filters, drop_path_prob, bn_momentum, **kwargs):
                super().__init__()
                self.net = torch.nn.Sequential(torch.nn.Conv2d(3,19,1), torch.nn.BatchNorm2d(19))
                self.model_config = dict(filter_multiplier=filters, drop_path_prob=drop_path_prob,
                                         bn_momentum=bn_momentum)
            def forward(self, image):
                return self.net(image)
            def set_progress(self, *args):
                pass
        def dataset(root, split, *args, **kwargs):
            self.assertEqual(split, 'train', 'Probe must never open official validation')
            return ToyData()
        data_protocol = dict(format='adl_probe_split_v1', fingerprint='toy',
                             train_indices=list(range(6)), selection_indices=[6,7])
        real_device, real_loader, real_fork = torch.device, torch.utils.data.DataLoader, torch.random.fork_rng
        def cpu_device(kind, *args, **kwargs):
            return real_device('cpu') if kind == 'cuda' else real_device(kind, *args, **kwargs)
        class CpuTorch:
            device = staticmethod(cpu_device)
            def __getattr__(self, name):
                return getattr(torch, name)
        def loader(*args, **kwargs):
            kwargs['pin_memory'] = False
            return real_loader(*args, **kwargs)
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            replacements = {
                'CityscapesFine': dataset, 'ReferenceAutoDeepLab': ToyModel, 'DataLoader': loader,
                'probe_datasets': lambda data, path: (torch.utils.data.Subset(data, list(range(6))),
                                                     torch.utils.data.Subset(data, [6,7]), data_protocol),
            }
            for name, value in replacements.items():
                stack.enter_context(patch.object(train_baseline, name, value))
            stack.enter_context(patch.object(train_baseline, 'torch', CpuTorch()))
            stack.enter_context(patch('torch.cuda.is_available', return_value=True))
            stack.enter_context(patch('torch.cuda.manual_seed_all'))
            stack.enter_context(patch('torch.cuda.current_device', return_value=0))
            stack.enter_context(patch('torch.cuda.get_rng_state_all', return_value=[]))
            stack.enter_context(patch('torch.cuda.set_rng_state_all'))
            stack.enter_context(patch('torch.backends.cudnn.version', return_value=None))
            stack.enter_context(patch('torch.random.fork_rng', side_effect=lambda **kw: real_fork(devices=[])))
            stack.enter_context(patch('train_baseline.signal.signal'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            argv = ['--data-dir', temp, '--max-steps', '6', '--warmup-steps', '0', '--workers', '0',
                    '--batch-size', '2', '--crop-size', '4', '--val-every', '2', '--save-every', '1',
                    '--snapshot-every', '100', '--selection-split-json', 'unused.json',
                    '--selection-calibration-batches', '1', '--no-amp', '--stop-at-step', '2']
            train_baseline.main(argv)
            checkpoint = torch.load(Path(temp)/'latest.pth', weights_only=True)
            self.assertEqual(checkpoint['step'], 2)
            train_baseline.main(argv)
            same = torch.load(Path(temp)/'latest.pth', weights_only=True)
            self.assertEqual(same['step'], 2)
            for key in checkpoint['state_dict']:
                torch.testing.assert_close(checkpoint['state_dict'][key], same['state_dict'][key], rtol=0, atol=0)
            argv[-1] = '4'
            train_baseline.main(argv)
            resumed = torch.load(Path(temp)/'latest.pth', weights_only=True)
            self.assertEqual(resumed['step'], 4)
            self.assertEqual(resumed['train_config']['max_steps'], 6)
            rows = [json.loads(line) for line in (Path(temp)/'validation.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows], [2,4])
            self.assertTrue(all(r['evaluation_split']=='train_selection' for r in rows))


if __name__ == '__main__':
    unittest.main()
