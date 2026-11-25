"""A finite, reproducible neighborhood of the published cell, with a PPO policy."""
import random

import torch
from torch import nn

from retrain_model.reference_model import BACKBONE, CELL, SEARCH_OPS, validate_architecture
from search_policy import architecture_key
from search_supernet import final_parameter_count

LOCAL_VERSION = 'adl_reference_local_v1'
LOCAL_OPS = ('sep3', 'sep5', 'dil3', 'dil5')


def cell_distance(a, b):
    # Branch addition is commutative: exchanging the two branches is not a mutation.
    return sum(2 - len(set(map(tuple, a[i:i+2])) & set(map(tuple, b[i:i+2])))
               for i in range(0, 10, 2))


def reference_neighborhood(count=128, seed=0, filters=20, max_params=None):
    """Reference plus one/two edge replacements, all convolutions, capped in F20."""
    cap = final_parameter_count(CELL, filters) if max_params is None else max_params
    if count < 2 or filters < 1 or cap < final_parameter_count(CELL, filters):
        raise ValueError('Neighborhood must include the reference and at least one challenger')
    reference = [list(e) for e in CELL]
    candidates = {architecture_key(reference): reference}
    rng = random.Random(seed)

    def admit(cell):
        validate_architecture(cell, BACKBONE)
        if cell_distance(cell, reference) <= 2 and final_parameter_count(cell, filters) <= cap:
            candidates.setdefault(architecture_key(cell), cell)

    singles = []
    for edge in range(10):
        for source in range(edge // 2 + 2):
            if source == reference[edge ^ 1][0]:
                continue
            for op in LOCAL_OPS:
                cell = [e[:] for e in reference]
                cell[edge] = [source, op]
                if cell_distance(cell, reference) == 1:
                    singles.append(cell)
    rng.shuffle(singles)
    # Reserve approximately half the library for two-edge hypotheses.
    for cell in singles:
        if len(candidates) >= max(2, count // 2):
            break
        admit(cell)
    for _ in range(count * 1000):
        if len(candidates) >= count:
            break
        cell = [e[:] for e in reference]
        for edge in rng.sample(range(10), 2):
            sources = [s for s in range(edge // 2 + 2) if s != cell[edge ^ 1][0]]
            cell[edge] = [rng.choice(sources), rng.choice(LOCAL_OPS)]
        admit(cell)
    if len(candidates) != count:
        raise ValueError('Cannot fill the requested local library under the parameter cap')
    return list(candidates.values())


class LocalCellPolicy(nn.Module):
    """Categorical PPO over an explicit library; replay uses the same cell likelihood."""
    def __init__(self, cells, hidden=64):
        super().__init__()
        if len({architecture_key(c) for c in cells}) != len(cells):
            raise ValueError('Duplicate architectures in local policy')
        encoded = [[v for source, op in c for v in (source, SEARCH_OPS.index(op))] for c in cells]
        self.register_buffer('library', torch.tensor(encoded, dtype=torch.long))
        self.features = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, len(cells))
        self.critic = nn.Linear(hidden, 1)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def forward(self, context, actions=None, uniform=False):
        features = self.features(context)
        logits = self.actor(features)
        dist = torch.distributions.Categorical(logits=logits * 0 if uniform else logits)
        if actions is None:
            index = dist.sample()
        else:
            match = (actions[:, None, :] == self.library[None, :, :]).all(-1)
            if not (match.sum(-1) == 1).all():
                raise ValueError('PPO action is outside the fixed local library')
            index = match.long().argmax(-1)
        return self.library[index], dist.log_prob(index), dist.entropy(), self.critic(features).squeeze(-1)
