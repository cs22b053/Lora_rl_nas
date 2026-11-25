"""Validated train-only partitions for independent architecture probes."""
import json
from pathlib import Path

from torch.utils.data import Subset

from retrain_data import CityscapesFine
from search_data import dataset_fingerprint


def validate_probe_split(split, fingerprint, count):
    if split.get('format') != 'adl_probe_split_v1' or split.get('fingerprint') != fingerprint:
        raise ValueError('Probe dataset manifest/fingerprint changed')
    train, heldout = split['train_indices'], split['selection_indices']
    if not train or not heldout or any(type(i) is not int for i in train + heldout):
        raise ValueError('Invalid probe split indices')
    if (len(set(train)) != len(train) or len(set(heldout)) != len(heldout)
            or set(train) & set(heldout) or set(train) | set(heldout) != set(range(count))):
        raise ValueError('Probe training/selection must be disjoint and cover the train dataset')
    return dict(format=split['format'], fingerprint=fingerprint,
                train_indices=train, selection_indices=heldout)


def probe_datasets(training, split_file):
    raw = CityscapesFine(training.root, 'train', training=False)
    protocol = validate_probe_split(json.loads(Path(split_file).read_text()),
                                    dataset_fingerprint(raw), len(raw))
    return (Subset(training, protocol['train_indices']),
            Subset(raw, protocol['selection_indices']), protocol)
