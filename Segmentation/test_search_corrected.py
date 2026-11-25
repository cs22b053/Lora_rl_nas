"""Small CPU regressions for search semantics; no search/training job is run."""
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import torch
from torch import nn
from PIL import Image
from retrain_data import CityscapesFine

from reference_training import model_from_checkpoint
from retrain_model.reference_model import BACKBONE, CELL, MODEL_VERSION, ReferenceAutoDeepLab
from search_data import split_indices
from search_policy import CellPolicy, decode_action, lower_tail_mean, ppo_update
from search_scoring import score_candidate
from search_optimization import update_weights
from search_supernet import InputLoRAConv, SearchSupernet, final_parameter_count, optimizer_groups


class SearchChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_splits_and_cvar(self):
        splits = split_indices(2975, 0)
        sets = [set(x) for x in splits.values()]
        self.assertEqual(len(set.union(*sets)), 2975)
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(splits, split_indices(2975, 0))
        values = torch.tensor([0., 1., 2., 3., 4.])
        self.assertAlmostEqual(lower_tail_mean(values, .3).item(), 1/3)
        self.assertAlmostEqual(lower_tail_mean(values, 1).item(), 2)
        self.assertAlmostEqual(lower_tail_mean(values, .01).item(), 0)

    def test_search_resize_happens_before_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'leftImg8bit/train/city'
            labels = root / 'gtFine/train/city'
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new('RGB', (16, 8)).save(images / 'city_leftImg8bit.png')
            mask = Image.new('L', (16, 8), 7)
            mask.paste(Image.new('L', (10, 8), 8), (6, 0))
            mask.save(labels / 'city_gtFine_labelIds.png')
            resized = CityscapesFine(root, 'train', crop_size=4, base_short_side=4, random_scale=False)
            native = CityscapesFine(root, 'train', crop_size=4, random_scale=False)
            with patch('retrain_data.random.random', return_value=1.), patch('retrain_data.random.randint', return_value=0):
                self.assertEqual(resized[0]['label'][0].tolist(), [0, 0, 0, 1])
                self.assertEqual(native[0]['label'][0].tolist(), [0, 0, 0, 0])

    def test_amp_skips_are_retried_and_never_reported_as_updates(self):
        class FakeScaler:
            def __init__(self, skips):
                self.remaining = skips
                self.value = 8.
                self.optimizer_steps = 0
            def is_enabled(self):
                return True
            def scale(self, loss):
                return loss
            def unscale_(self, optimizer):
                pass
            def get_scale(self):
                return self.value
            def step(self, optimizer):
                if not self.remaining:
                    optimizer.step()
                    self.optimizer_steps += 1
            def update(self):
                if self.remaining:
                    self.remaining -= 1
                    self.value /= 2
        model = nn.Sequential(nn.Conv2d(3, 19, 1), nn.BatchNorm2d(19))
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        image, target = torch.randn(2,3,3,3), torch.zeros(2,3,3,dtype=torch.long)
        initial = model[0].weight.detach().clone()
        scaler = FakeScaler(skips=2)
        result = update_weights(model, optimizer, scaler, image, target, amp=False, max_attempts=3)
        self.assertEqual(result['retries'], 2)
        self.assertEqual(scaler.optimizer_steps, 1)
        self.assertFalse(torch.equal(initial, model[0].weight))
        self.assertEqual(model[1].num_batches_tracked.item(), 1)
        initial = model[0].weight.detach().clone()
        scaler = FakeScaler(skips=3)
        with self.assertRaisesRegex(FloatingPointError, 'No successful weight update'):
            update_weights(model, optimizer, scaler, image, target, amp=False, max_attempts=2)
        self.assertEqual(scaler.optimizer_steps, 0)
        torch.testing.assert_close(initial, model[0].weight)
        self.assertEqual(model[1].num_batches_tracked.item(), 1)

    def test_policy_probability_and_update(self):
        policy = CellPolicy(max_skips=2)
        context = torch.zeros(16, 3)
        with torch.no_grad():
            actions, logp, entropy, values = policy(context)
            replay_logp = policy(context, actions)[1]
        torch.testing.assert_close(logp, replay_logp)
        self.assertTrue(torch.isfinite(entropy).all())
        for action in actions:
            cell = decode_action(action)
            self.assertLessEqual(sum(op == 'skip' for _,op in cell), 2)
            for n in range(5):
                self.assertNotEqual(cell[2*n][0], cell[2*n+1][0])
        original = policy.actor.weight.clone()
        metrics = ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=.001), context,
                             actions, logp, values, torch.linspace(0, 1, 16), epochs=2)
        self.assertTrue(torch.isfinite(torch.tensor(list(metrics.values()))).all())
        self.assertFalse(torch.equal(original, policy.actor.weight))

    def test_lora_zero_start_and_input_isolation(self):
        base = nn.Conv2d(4, 4, 1, bias=False)
        adapted = InputLoRAConv(base, inputs=3, rank=2)
        x = torch.randn(2, 4, 7, 7)
        torch.testing.assert_close(adapted(x), base(x))
        adapted.source = 1
        adapted(x).square().mean().backward()
        self.assertGreater(adapted.lora_B.grad[1].abs().sum().item(), 0)
        self.assertEqual(adapted.lora_B.grad[0].abs().sum().item(), 0)
        with torch.no_grad():
            adapted.lora_B[1].fill_(.1)
        self.assertFalse(torch.equal(adapted(x), base(x)))
        adapted.source = 0
        torch.testing.assert_close(adapted(x), base(x))

    def test_inactive_lora_momentum_is_frozen(self):
        model = InputLoRAConv(nn.Conv2d(2, 19, 1, bias=False), inputs=3, rank=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        image = torch.randn(2, 2, 3, 3)
        target = torch.zeros(2, 3, 3, dtype=torch.long)
        model.source = 0
        update_weights(model, optimizer, scaler, image, target, amp=False)
        before = [(p[0].detach().clone(), optimizer.state[p]['momentum_buffer'][0].clone())
                  for p in (model.lora_A, model.lora_B)]
        model.source = 1
        update_weights(model, optimizer, scaler, image, target, amp=False)
        for parameter, (value, momentum) in zip((model.lora_A, model.lora_B), before):
            torch.testing.assert_close(parameter[0], value, rtol=0, atol=0)
            torch.testing.assert_close(optimizer.state[parameter]['momentum_buffer'][0], momentum,
                                       rtol=0, atol=0)
        self.assertGreater(model.lora_B[1].abs().sum().item(), 0)
        self.assertEqual(model.lora_B[2].abs().sum().item(), 0)

    def test_nonfinite_candidate_is_not_ranked(self):
        class Nonfinite(nn.Module):
            def set_architecture(self, cell):
                pass
            def forward(self, image):
                return torch.full((image.shape[0], 19, *image.shape[-2:]), float('nan'))
        model = Nonfinite().train()
        sample = dict(image=torch.zeros(40, 3, 2, 2), label=torch.zeros(40, 2, 2, dtype=torch.long))
        with self.assertRaises(FloatingPointError):
            score_candidate(model, CELL, [sample], [sample], torch.device('cpu'), amp=False)
        self.assertTrue(model.training)

    def test_amp_candidate_retry_replays_data_and_reports_precision(self):
        sample = dict(image=torch.zeros(2,3,2,2), label=torch.zeros(2,2,2,dtype=torch.long))
        result = dict(miou=.5, reward=.4, image_cvar=0., images=40, image_scores=[0.]*40)
        # Simulate AMP failure without CUDA execution; BN restoration is tested
        # separately against the real scoring helper below.
        with patch('search_scoring._score_candidate_once',
                   side_effect=[FloatingPointError('overflow'), result]) as scorer:
            actual = score_candidate(nn.Identity(), CELL, iter([sample]), iter([sample]), torch.device('cuda'))
        self.assertEqual(actual['scoring_precision'], 'fp32')
        self.assertTrue(actual['amp_fallback'])
        first, retry = scorer.call_args_list
        self.assertTrue(first.kwargs['amp'])
        self.assertFalse(retry.kwargs['amp'])
        self.assertIs(first.args[2], retry.args[2])
        self.assertIs(first.args[3], retry.args[3])
        self.assertIs(retry.args[2][0], sample)
        with patch('search_scoring._score_candidate_once', side_effect=FloatingPointError('bad')) as scorer:
            with self.assertRaisesRegex(FloatingPointError, 'even after fresh FP32'):
                score_candidate(nn.Identity(), CELL, [sample], [sample], torch.device('cuda'))
            self.assertEqual(scorer.call_count, 2)
        with patch('search_scoring._score_candidate_once', side_effect=FloatingPointError('bad')) as scorer:
            with self.assertRaises(FloatingPointError):
                score_candidate(nn.Identity(), CELL, [sample], [sample], torch.device('cpu'), amp=False)
            self.assertEqual(scorer.call_count, 1)

    def test_nonfinite_calibration_restores_bn_before_retry(self):
        class BrokenCalibration(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(3, momentum=.03)
            def set_architecture(self, cell):
                pass
            def forward(self, image):
                return self.bn(image * float('nan'))
        model = BrokenCalibration().train()
        before = {name:v.clone() for name,v in model.named_buffers()}
        sample = dict(image=torch.ones(2,3,2,2), label=torch.zeros(2,2,2,dtype=torch.long))
        with self.assertRaisesRegex(FloatingPointError, 'during candidate BN calibration'):
            score_candidate(model, CELL, [sample], [sample], torch.device('cpu'), amp=False)
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(model.training)
        self.assertTrue(model.bn.training)
        self.assertEqual(model.bn.momentum, .03)

    def test_sampled_network_and_export(self):
        model = SearchSupernet(2, lora_rank=2)
        cell = decode_action(CellPolicy()(torch.zeros(1,3))[0][0])
        model.set_architecture(cell)
        y = model(torch.randn(2,3,33,33))
        self.assertEqual(tuple(y.shape), (2,19,33,33))
        y.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n,p in model.named_parameters() if 'lora_B' in n))
        groups = optimizer_groups(model, .0003, 2.)
        ids = [id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        self.assertEqual(next(g for g in groups if g['name']=='lora')['lr_multiplier'], 2.)
        child = ReferenceAutoDeepLab(2, cell=cell)
        self.assertEqual(sum(p.numel() for p in child.parameters()), final_parameter_count(cell, 2))
        self.assertEqual(final_parameter_count(CELL), 21622451)
        restored = model_from_checkpoint(dict(model_version=MODEL_VERSION, model_config=child.model_config,
                                              state_dict=child.state_dict()))
        self.assertEqual(restored.model_config, child.model_config)
        restored.eval()
        with torch.no_grad():
            self.assertEqual(tuple(restored(torch.zeros(1,3,33,49)).shape), (1,19,33,49))

    def test_candidate_bn_isolation_and_true_tail(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(3)
            def set_architecture(self, cell):
                pass
            def forward(self, x):
                x = self.bn(x)
                output = torch.zeros(x.shape[0],19,*x.shape[-2:])
                output[:,1] = 10
                return output
        model = Toy().train()
        before = {k:v.clone() for k,v in model.named_buffers()}
        sample = dict(image=torch.randn(40,3,2,2), label=torch.ones(40,2,2,dtype=torch.long))
        sample['label'][:8] = 0
        result = score_candidate(model, CELL, [sample], [sample], torch.device('cpu'), .2, .2)
        self.assertAlmostEqual(result['image_cvar'], 0)
        self.assertAlmostEqual(result['reward'], .8*result['miou'])
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, before[name])
        self.assertTrue(model.training)
        self.assertTrue(model.bn.training)
        self.assertEqual(model.bn.momentum, .1)


if __name__ == '__main__':
    unittest.main()
