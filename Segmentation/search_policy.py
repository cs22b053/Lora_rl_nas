"""Discrete cell policy and PPO for a one-step, measured architecture reward."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from retrain_model.reference_model import CELL, BACKBONE, SEARCH_OPS, validate_architecture


def lower_tail_mean(values, alpha):
    """Empirical lower-tail CVaR, including fractional mass at the quantile."""
    if not 0 < alpha <= 1 or values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError('CVaR requires finite samples and alpha in (0,1]')
    ordered = values.flatten().sort().values
    mass = alpha * len(ordered)
    whole = int(math.floor(mass))
    total = ordered[:whole].sum()
    if whole < len(ordered):
        total = total + (mass - whole) * ordered[whole]
    return total / mass


def decode_action(action):
    action = action.tolist() if isinstance(action, torch.Tensor) else action
    cell = [[int(action[2 * edge]), SEARCH_OPS[int(action[2 * edge + 1])]] for edge in range(10)]
    validate_architecture(cell, BACKBONE)
    return cell


def architecture_key(cell):
    return tuple(tuple(sorted(tuple(edge) for edge in cell[2*n:2*n+2])) for n in range(5))


class CellPolicy(nn.Module):
    def __init__(self, max_skips=2, hidden=64):
        super().__init__()
        self.max_skips = max_skips
        self.features = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, 10 * (6 + len(SEARCH_OPS)))
        self.critic = nn.Linear(hidden, 1)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def forward(self, context, actions=None, uniform=False):
        features = self.features(context)
        logits = self.actor(features).reshape(-1, 10, 6 + len(SEARCH_OPS))
        if uniform:
            logits = logits * 0
        decisions, log_probs, entropies = [], [], []
        skips = torch.zeros(context.shape[0], dtype=torch.long, device=context.device)
        first_source = None
        for edge in range(10):
            source_logits = logits[:, edge, :6].clone()
            source_logits[:, edge // 2 + 2:] = -torch.inf
            if edge % 2:
                source_logits.scatter_(1, first_source[:, None], -torch.inf)
            source_dist = torch.distributions.Categorical(logits=source_logits)
            source = source_dist.sample() if actions is None else actions[:, 2 * edge]
            if edge % 2 == 0:
                first_source = source
            op_logits = logits[:, edge, 6:].clone()
            op_logits[skips >= self.max_skips, SEARCH_OPS.index('skip')] = -torch.inf
            op_dist = torch.distributions.Categorical(logits=op_logits)
            operation = op_dist.sample() if actions is None else actions[:, 2 * edge + 1]
            skips = skips + (operation == SEARCH_OPS.index('skip')).long()
            decisions.extend([source, operation])
            log_probs.extend([source_dist.log_prob(source), op_dist.log_prob(operation)])
            entropies.extend([source_dist.entropy(), op_dist.entropy()])
        return (torch.stack(decisions, dim=1), torch.stack(log_probs).sum(0),
                torch.stack(entropies).mean(0), self.critic(features).squeeze(-1))


def ppo_update(policy, optimizer, contexts, actions, old_log_probs, old_values,
               rewards, epochs=4, clip=0.2, entropy_coef=0.01, target_kl=0.02):
    # Each sampled architecture is a COMPLETE one-step episode. No GAE chain
    # across unrelated candidates, and no sorting of PPO advantages as "CVaR".
    advantages = rewards - old_values
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    metrics = {}
    for iteration in range(epochs):
        _, log_probs, entropy, values = policy(contexts, actions)
        log_ratio = log_probs - old_log_probs
        ratio = log_ratio.exp()
        kl = ((ratio - 1) - log_ratio).mean()
        if not torch.isfinite(kl):
            raise FloatingPointError('Non-finite PPO importance ratio')
        if iteration and kl.item() > target_kl:
            break
        policy_loss = -torch.minimum(ratio * advantages,
                                    ratio.clamp(1-clip, 1+clip) * advantages).mean()
        value_loss = F.mse_loss(values, rewards)
        loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5, error_if_nonfinite=True)
        optimizer.step()
        metrics = dict(policy_loss=policy_loss.item(), value_loss=value_loss.item(),
                       entropy=entropy.mean().item(), approx_kl=kl.item(), epochs=iteration+1,
                       reward_mean=rewards.mean().item())
    return metrics
