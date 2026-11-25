"""Cityscapes fine labels and reference NAS input normalization, without COCO dependencies."""
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

VALID_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
CLASS_NAMES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
               'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
               'truck', 'bus', 'train', 'motorcycle', 'bicycle')


class CityscapesFine(Dataset):
    def __init__(self, root, split, crop_size=769, training=None,
                 base_short_side=None, random_scale=True):
        self.root, self.split, self.crop_size = Path(root), split, crop_size
        self.training = split == 'train' if training is None else training
        if base_short_side is not None and base_short_side < 1:
            raise ValueError('base_short_side must be positive')
        self.base_short_side, self.random_scale = base_short_side, random_scale
        if split not in ('train', 'val'):
            raise ValueError('Only fine train/val splits are supported')
        self.images = sorted((self.root / 'leftImg8bit' / split).glob('*/*_leftImg8bit.png'))
        if not self.images:
            raise FileNotFoundError('No Cityscapes images under {}'.format(self.root / 'leftImg8bit' / split))
        self.labels = [self.root / 'gtFine' / split / p.parent.name /
                       p.name.replace('_leftImg8bit.png', '_gtFine_labelIds.png') for p in self.images]
        missing = next((p for p in self.labels if not p.is_file()), None)
        if missing is not None:
            raise FileNotFoundError('Missing fine label: {}'.format(missing))
        self.lut = [255] * 256
        for train_id, label_id in enumerate(VALID_IDS):
            self.lut[label_id] = train_id

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with Image.open(self.images[index]) as im:
            image = im.convert('RGB')
        with Image.open(self.labels[index]) as im:
            label = im.convert('L').point(self.lut)
        if self.training:
            if self.base_short_side is not None:
                ratio = self.base_short_side / min(image.size)
                size = tuple(max(1, int(d * ratio)) for d in image.size)
                image = image.resize(size, Image.Resampling.BILINEAR)
                label = label.resize(size, Image.Resampling.NEAREST)
            # Official DeepLab scale range; no additional colour jitter or blur.
            scale = random.choice([0.5 + 0.25 * i for i in range(7)]) if self.random_scale else 1.0
            size = tuple(max(1, int((d - 1) * scale + 1)) for d in image.size)
            image = image.resize(size, Image.Resampling.BILINEAR)
            label = label.resize(size, Image.Resampling.NEAREST)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            # Pad with the preprocessing mean; ignore padded labels.
            width, height = image.size
            padded = (max(width, self.crop_size), max(height, self.crop_size))
            canvas = Image.new('RGB', padded, (128, 128, 128))
            canvas.paste(image, (0, 0))
            mask = Image.new('L', padded, 255)
            mask.paste(label, (0, 0))
            left = random.randint(0, padded[0] - self.crop_size)
            top = random.randint(0, padded[1] - self.crop_size)
            box = (left, top, left + self.crop_size, top + self.crop_size)
            image, label = canvas.crop(box), mask.crop(box)
        # Use Pillow's byte buffer directly; this also works with older PyTorch
        # builds whose NumPy bridge is incompatible with an installed NumPy 2.
        width, height = image.size
        image = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        image = image.reshape(height, width, 3).permute(2, 0, 1).float()
        image = image.mul_(2.0 / 255.0).sub_(1.0)
        label = torch.frombuffer(bytearray(label.tobytes()), dtype=torch.uint8).reshape(height, width).long()
        return {'image': image, 'label': label}


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
