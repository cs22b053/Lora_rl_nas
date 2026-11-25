"""Disjoint training splits and reproducible, common image panels for NAS."""
import hashlib
import random
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from retrain_data import CityscapesFine


def split_indices(count, seed):
    order = list(range(count))
    random.Random(seed).shuffle(order)
    a, b = int(count * .6), int(count * .8)
    if a < 2 or b == a or b == count:
        raise ValueError('Not enough images for three disjoint search splits')
    return dict(weights=order[:a], controller=order[a:b], selection=order[b:])


def dataset_fingerprint(dataset):
    paths = '\n'.join(str(p.relative_to(dataset.root)) for p in dataset.images)
    return hashlib.sha256(paths.encode()).hexdigest()


class SearchPanel(Dataset):
    def __init__(self, dataset, indices, crop_size, seed, short_side=513):
        if dataset.training:
            raise ValueError('Panels require an unaugmented dataset')
        self.dataset, self.indices, self.crop_size, self.seed = dataset, indices, crop_size, seed
        self.short_side = short_side

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image_id = self.indices[index]
        sample = self.dataset[image_id]
        image, target = sample['image'], sample['label']
        h, w = target.shape
        # A deterministic 1/2-resolution view (at least crop size), with a
        # reproducible crop. All candidates in a rollout see the SAME tensors.
        short = max(self.crop_size, self.short_side)
        size = (int(h * short / min(h, w)), int(w * short / min(h, w)))
        image = F.interpolate(image[None], size=size, mode='bilinear', align_corners=True)[0]
        target = F.interpolate(target[None, None].float(), size=size, mode='nearest')[0, 0].long()
        rng = random.Random(self.seed * 1000003 + image_id)
        top = rng.randrange(size[0] - self.crop_size + 1)
        left = rng.randrange(size[1] - self.crop_size + 1)
        return {'image': image[:, top:top+self.crop_size, left:left+self.crop_size].contiguous(),
                'label': target[top:top+self.crop_size, left:left+self.crop_size].contiguous()}
