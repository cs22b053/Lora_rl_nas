"""Small CPU regression checks. No dataset download or training run required."""
import random
import copy
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from reference_training import (atomic_save, learning_rate, model_from_checkpoint,
                                 parameter_groups, read_checkpoint, set_learning_rate, validate,
                                 calibrated_batch_norm, resolve_training_steps, train_update)
from reference_training import compatible_cuda_rng_state, restore_cuda_rng_states
from retrain_data import CityscapesFine
from retrain_model.reference_model import MODEL_VERSION, ReferenceAutoDeepLab


class BaselineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_model_capacity(self):
        model = ReferenceAutoDeepLab(32)
        # Published Auto-DeepLab-M has 21.62M parameters.
        self.assertAlmostEqual(sum(p.numel() for p in model.parameters()) / 1e6, 21.62, places=2)

    def test_epoch_budget(self):
        self.assertEqual(resolve_training_steps(2975, 16, 1, epochs=4000), 740000)
        self.assertEqual(resolve_training_steps(2975, 16, 1), 1500000)
        self.assertEqual(resolve_training_steps(2975, 16, 1, max_steps=20000), 20000)
        with self.assertRaises(ValueError):
            resolve_training_steps(2975, 16, 1, epochs=4000, max_steps=20000)
        with self.assertRaises(ValueError):
            resolve_training_steps(2975, 8, 2, epochs=4000)

    def test_legacy_cuda_rng_preserves_seed_and_offset(self):
        payload = torch.tensor(list(struct.pack('=Qq', 2**63 + 123, 987654320)), dtype=torch.uint8)
        legacy = torch.cat([torch.full((800,), 255, dtype=torch.uint8), payload])
        before = legacy.clone()
        converted = compatible_cuda_rng_state(legacy, 16)
        torch.testing.assert_close(converted, payload)
        torch.testing.assert_close(legacy, before)
        self.assertNotEqual(converted.data_ptr(), legacy[800:].data_ptr())
        self.assertEqual(struct.unpack('=Qq', bytes(converted.tolist())), (2**63 + 123, 987654320))
        torch.testing.assert_close(compatible_cuda_rng_state(converted, 16), payload)

    def test_unknown_cuda_rng_layouts_fail(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            compatible_cuda_rng_state(torch.zeros(32, dtype=torch.uint8), 16)
        with self.assertRaisesRegex(ValueError, 'padding'):
            compatible_cuda_rng_state(torch.zeros(816, dtype=torch.uint8), 16)
        invalid = torch.cat([torch.full((800,), 255, dtype=torch.uint8),
                             torch.tensor(list(struct.pack('=Qq', 0, 3)), dtype=torch.uint8)])
        with self.assertRaisesRegex(ValueError, 'offset'):
            compatible_cuda_rng_state(invalid, 16)

    def test_cuda_rng_restore_migration_and_native_resume(self):
        native = torch.tensor(list(struct.pack('=Qq', 0, 4096)), dtype=torch.uint8)
        legacy = torch.cat([torch.full((800,), 255, dtype=torch.uint8), native])
        with patch('torch.cuda.get_rng_state_all', return_value=[native]), \
                patch('torch.cuda.set_rng_state_all') as setter:
            events = restore_cuda_rng_states([legacy])
            self.assertEqual(events, [dict(device=0, saved_bytes=816, restored_bytes=16)])
            torch.testing.assert_close(setter.call_args.args[0][0], native)
            self.assertEqual(restore_cuda_rng_states([native]), [])
            setter.reset_mock()
            with self.assertRaisesRegex(ValueError, 'device count'):
                restore_cuda_rng_states([native, native])
            setter.assert_not_called()

    def test_bn_momentum_checkpoint_compatibility(self):
        legacy = ReferenceAutoDeepLab(2)
        self.assertNotIn('bn_momentum', legacy.model_config)
        model = ReferenceAutoDeepLab(2, bn_momentum=.01)
        restored = model_from_checkpoint(dict(model_config=model.model_config,
                                              state_dict=model.state_dict()))
        self.assertEqual(restored.model_config['bn_momentum'], .01)
        self.assertTrue(all(m.momentum == .01 for m in restored.modules()
                            if isinstance(m, torch.nn.BatchNorm2d)))

    def test_calibration_is_temporary_and_disables_dropout(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm2d(3, momentum=.0003),
                                    torch.nn.Dropout(1.0)).train()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        samples = [dict(image=torch.full((2, 3, 2, 2), v)) for v in (3., 5.)]
        for fail in (False, True):
            try:
                with calibrated_batch_norm(model, samples, torch.device('cpu'), 2):
                    self.assertFalse(model.training)
                    self.assertFalse(model[1].training)
                    torch.testing.assert_close(model[0].running_mean, torch.full((3,), 4.))
                    self.assertEqual(model[0].num_batches_tracked.item(), 2)
                    if fail:
                        raise RuntimeError('intentional evaluation failure')
            except RuntimeError:
                if not fail:
                    raise
            self.assertTrue(model.training)
            self.assertTrue(model[0].training)
            self.assertEqual(model[0].momentum, .0003)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key])

    def test_amp_retry_preserves_batchnorm_and_optimizer_update(self):
        class FakeScaler:
            def __init__(self, skip):
                self.skip, self.value = skip, 1024.
            def scale(self, loss):
                return loss
            def get_scale(self):
                return self.value
            def step(self, optimizer):
                if not self.skip:
                    optimizer.step()
            def update(self):
                if self.skip:
                    self.value /= 2
                    self.skip = False
        model = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.Conv2d(3, 19, 1))
        reference = copy.deepcopy(model)
        sample = dict(image=torch.randn(2, 3, 4, 4), label=torch.zeros(2, 4, 4, dtype=torch.long))
        for m, skip in ((model, True), (reference, False)):
            train_update(m, torch.optim.SGD(m.parameters(), lr=.01), FakeScaler(skip),
                         [sample], torch.device('cpu'), amp=False)
        self.assertEqual(model[0].num_batches_tracked.item(), 1)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[key])

    def test_nonfinite_validation_fails(self):
        class NonfiniteModel(torch.nn.Module):
            def forward(self, x):
                return torch.full((x.shape[0], 19, *x.shape[-2:]), float('nan'))
        with self.assertRaises(FloatingPointError):
            validate(NonfiniteModel(), [dict(image=torch.zeros(1, 3, 2, 2),
                                             label=torch.zeros(1, 2, 2, dtype=torch.long))],
                     torch.device('cpu'), amp=False)

    def test_optimizer_and_forward_backward(self):
        model = ReferenceAutoDeepLab(2)
        groups = parameter_groups(model, 1e-4)
        grouped = [id(p) for g in groups for p in g['params']]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(p) for p in model.parameters()})
        opt = torch.optim.SGD(groups, lr=0.05)
        for step, expected in ((0, 0.00005), (1000, 0.05), (1500000, 0.0)):
            rate = learning_rate(step, 1500000, 0.05)
            set_learning_rate(opt, rate)
            self.assertAlmostEqual(rate, expected)
            self.assertTrue(all(g['lr'] == rate for g in opt.param_groups))
        model.train()
        model.set_progress(5, 10)
        y = model(torch.randn(2, 3, 65, 65))
        self.assertEqual(tuple(y.shape), (2, 19, 65, 65))
        y.square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        model.eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.randn(1, 3, 64, 96)).shape), (1, 19, 64, 96))

    def test_cityscapes_labels_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train', 'val'):
                images = root / 'leftImg8bit' / split / 'city'
                labels = root / 'gtFine' / split / 'city'
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                Image.new('RGB', (3, 2), (0, 128, 255)).save(images / 'city_0_leftImg8bit.png')
                mask = Image.new('L', (3, 2))
                mask.putdata([7, 8, 33, 0, 255, 11])
                mask.save(labels / 'city_0_gtFine_labelIds.png')
            sample = CityscapesFine(root, 'val')[0]
            self.assertEqual(sample['label'].tolist(), [[0, 1, 18], [255, 255, 2]])
            self.assertEqual(sample['image'][0, 0, 0].item(), -1)
            self.assertEqual(sample['image'][2, 0, 0].item(), 1)
            random.seed(1)
            sample = CityscapesFine(root, 'train', crop_size=9)[0]
            self.assertEqual(tuple(sample['image'].shape), (3, 9, 9))
            self.assertTrue(set(sample['label'].flatten().tolist()) <= {0, 1, 2, 18, 255})
            self.assertIn(255, sample['label'])

    def test_metric_and_checkpoint_roundtrip(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, x):
                output = torch.zeros(x.shape[0], 19, *x.shape[-2:])
                output[:, 1] = 10
                return output
        model = ConstantModel().train()
        samples = [{'image': torch.zeros(1, 3, 2, 2),
                    'label': torch.tensor([[[1, 1], [0, 255]]])}]
        result = validate(model, samples, torch.device('cpu'), scales=(1.0, 1.5), flip=True)
        self.assertFalse(model.training)
        self.assertEqual(result['images'], 1)
        self.assertAlmostEqual(result['class_iou'][1], 200 / 3)
        self.assertEqual(result['confusion'][0][1], 1)
        self.assertAlmostEqual(result['image_cvar'], 100 / 3)
        self.assertEqual(result['image_cvar_alpha'], .2)
        model = ReferenceAutoDeepLab(2).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            atomic_save({'model_version': MODEL_VERSION, 'model_config': model.model_config,
                         'state_dict': model.state_dict(), 'python_rng': random.getstate()}, path)
            restored = model_from_checkpoint(read_checkpoint(path)).eval()
            x = torch.randn(1, 3, 33, 33)
            with torch.no_grad():
                torch.testing.assert_close(model(x), restored(x))
            atomic_save({'state_dict': model.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, 'legacy'):
                read_checkpoint(path)

    def test_image_tail_is_independent_of_evaluation_batching(self):
        class PredictOne(torch.nn.Module):
            def forward(self, x):
                logits = torch.zeros(x.shape[0], 19, *x.shape[-2:])
                logits[:, 1] = 10
                return logits
        images = torch.zeros(5, 3, 2, 2)
        labels = torch.ones(5, 2, 2, dtype=torch.long)
        labels[0] = 0
        together = validate(PredictOne(), [dict(image=images, label=labels)], torch.device('cpu'))
        apart = validate(PredictOne(), [dict(image=x[None], label=y[None])
                                       for x, y in zip(images, labels)], torch.device('cpu'))
        self.assertEqual(together['confusion'], apart['confusion'])
        self.assertEqual(together['image_cvar'], 0.)
        self.assertEqual(together['image_cvar'], apart['image_cvar'])
        self.assertEqual(together['image_miou_percent'], [0., 100., 100., 100., 100.])


if __name__ == '__main__':
    unittest.main()
