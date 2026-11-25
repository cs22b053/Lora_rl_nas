"""
PPO trainer for architecture search.
Key insight: Use the NAO predictor as a DENSE reward signal,
avoiding the need to train child networks for every candidate.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from ArchitecturePolicy import ArchitecturePolicy


class PPOArchTrainer:
    """
    PPO trainer that uses NAO predictor as reward oracle.
    """

    def __init__(self,
                 nao_model,
                 policy: ArchitecturePolicy,
                 embedding_dim: int = 64,
                 lr: float = 3e-4,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 max_step_magnitude: float = 3.0,
                 cvar_alpha: float = 0.2,
                 ):
        self.nao = nao_model
        self.policy = policy
        self.max_step_magnitude = max_step_magnitude
        self.cvar_alpha = cvar_alpha

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        self.clip_epsilon  = clip_epsilon
        self.value_coef    = value_coef
        self.entropy_coef  = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self._reset_buffer()

    def _reset_buffer(self):
        self.buf_arch_embs = []
        self.buf_pred_accs = []
        self.buf_actions   = []
        self.buf_log_probs = []
        self.buf_values    = []
        self.buf_rewards   = []
        self.buf_dones     = []

    @torch.no_grad()
    def _get_predictor_reward(self, new_encoder_outputs):
        new_emb = torch.mean(new_encoder_outputs, dim=1)
        new_emb = F.normalize(new_emb, 2, dim=-1)

        if len(list(self.nao.encoder.mlp.children())) > 0:
            out = self.nao.encoder.mlp(new_emb)
        else:
            out = new_emb

        predicted_acc = torch.sigmoid(self.nao.encoder.regressor(out))
        return predicted_acc.squeeze(-1)

    @torch.no_grad()
    def collect_rollout(self, arch_pool_seqs, n_steps: int = 512):
        self._reset_buffer()
        self.policy.eval()
        self.nao.eval()

        device = next(self.policy.parameters()).device
        arch_tensors = torch.LongTensor(arch_pool_seqs).to(device)

        encoder_outputs, _, arch_embs, pred_accs = self.nao.encoder(arch_tensors)

        current_embs      = arch_embs.detach()
        current_enc_outputs = encoder_outputs.detach()
        current_pred_accs = pred_accs.detach()

        n_archs   = current_embs.shape[0]
        collected = 0

        while collected < n_steps:
            idx = torch.randint(0, n_archs, (min(self.batch_size, n_steps - collected),))
            batch_embs      = current_embs[idx]
            batch_enc_outputs = current_enc_outputs[idx]
            batch_pred_accs = current_pred_accs[idx]

            actions, log_probs, _, values = self.policy.get_action_and_value(
                batch_embs, batch_pred_accs
            )

            action_norm     = actions.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            actions_clamped = actions * torch.clamp(
                self.max_step_magnitude / action_norm, max=1.0
            )

            new_enc_outputs = batch_enc_outputs + actions_clamped.unsqueeze(1)
            new_enc_outputs = F.normalize(new_enc_outputs, 2, dim=-1)

            new_pred_accs = self._get_predictor_reward(new_enc_outputs)
            old_pred_accs = batch_pred_accs.squeeze(-1)

            # Reward: focus on improvement; drop absolute term to reduce bias from a noisy predictor
            reward = new_pred_accs - old_pred_accs

            # Normalize reward per mini-batch to stabilize CVaR selection
            reward = (reward - reward.mean()) / (reward.std() + 1e-8)

            self.buf_arch_embs.append(batch_embs.cpu())
            self.buf_pred_accs.append(batch_pred_accs.cpu())
            self.buf_actions.append(actions.cpu())
            self.buf_log_probs.append(log_probs.cpu())
            self.buf_values.append(values.squeeze(-1).cpu())
            self.buf_rewards.append(reward.cpu())
            self.buf_dones.append(torch.zeros(len(idx)))

            collected += len(idx)

        logging.info('Rollout complete: %d steps, mean reward=%.4f, mean pred_acc=%.4f',
                     collected,
                     torch.cat(self.buf_rewards).mean().item(),
                     torch.cat(self.buf_pred_accs).mean().item())

    def _compute_gae(self):
        rewards = torch.cat(self.buf_rewards)
        values  = torch.cat(self.buf_values)
        dones   = torch.cat(self.buf_dones)

        n           = len(rewards)
        advantages  = torch.zeros(n)
        last_gae    = 0.0
        next_value  = values[-1]

        for t in reversed(range(n)):
            next_non_terminal = 1.0 - dones[t]
            next_val  = next_value if t == n - 1 else values[t + 1]
            delta     = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            last_gae  = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self):
        """
        PPO update with CVaR objective.
        From 'Optimizing the CVaR via Sampling' (Rockafellar & Uryasev).
        """
        self.policy.train()

        device = next(self.policy.parameters()).device
        arch_embs     = torch.cat(self.buf_arch_embs).to(device)
        pred_accs     = torch.cat(self.buf_pred_accs).to(device)
        old_actions   = torch.cat(self.buf_actions).to(device)
        old_log_probs = torch.cat(self.buf_log_probs).to(device)

        advantages, returns = self._compute_gae()
        advantages = advantages.to(device)
        returns    = returns.to(device)

        # Normalize advantages BEFORE CVaR masking (critical)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(arch_embs)
        metrics = {
            'policy_loss': [], 'value_loss': [],
            'entropy': [], 'clip_frac': [], 'cvar': []
        }

        for epoch in range(self.n_epochs):
            perm = torch.randperm(n)

            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]

                batch_embs    = arch_embs[idx]
                batch_accs    = pred_accs[idx]
                batch_actions = old_actions[idx]
                batch_old_lp  = old_log_probs[idx]
                batch_adv     = advantages[idx]
                batch_returns = returns[idx]

                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    batch_embs, batch_accs, action=batch_actions
                )

                ratio  = torch.exp(new_log_probs - batch_old_lp)
                surr1  = ratio * batch_adv
                surr2  = torch.clamp(ratio,
                                     1 - self.clip_epsilon,
                                     1 + self.clip_epsilon) * batch_adv
                clipped_surr = torch.min(surr1, surr2)

                # CVaR objective: select worst alpha fraction based on unclipped returns
                k = max(1, int(len(surr1) * self.cvar_alpha))
                worst_idx = torch.argsort(surr1)[:k]
                cvar_value  = clipped_surr[worst_idx].mean()
                policy_loss = -cvar_value

                value_loss   = F.mse_loss(new_values.squeeze(-1), batch_returns)
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + self.value_coef   * value_loss
                        + self.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                clip_frac = ((ratio - 1).abs() > self.clip_epsilon).float().mean()

                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(-entropy_loss.item())
                metrics['clip_frac'].append(clip_frac.item())
                metrics['cvar'].append(cvar_value.item())

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    @torch.no_grad()
    def generate_architectures(self, arch_pool_seqs, n_new: int, nao_decoder):
        self.policy.eval()
        self.nao.eval()

        device       = next(self.policy.parameters()).device
        arch_tensors = torch.LongTensor(arch_pool_seqs).to(device)

        encoder_outputs, _, arch_embs, pred_accs = self.nao.encoder(arch_tensors)

        new_arch_seqs = []
        attempts      = 0
        max_attempts  = n_new * 5

        while len(new_arch_seqs) < n_new and attempts < max_attempts:
            batch_size = min(32, n_new - len(new_arch_seqs))
            idx = torch.randint(0, arch_embs.shape[0], (batch_size,))

            batch_embs        = arch_embs[idx]
            batch_enc_outputs = encoder_outputs[idx]
            batch_pred_accs   = pred_accs[idx]

            action_mean, _, _ = self.policy(batch_embs, batch_pred_accs)

            action_norm    = action_mean.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            action_clamped = action_mean * torch.clamp(
                self.max_step_magnitude / action_norm, max=1.0
            )

            new_enc_outputs = batch_enc_outputs + action_clamped.unsqueeze(1)
            new_enc_outputs = F.normalize(new_enc_outputs, 2, dim=-1)

            new_arch_emb = torch.mean(new_enc_outputs, dim=1)
            new_arch_emb = F.normalize(new_arch_emb, 2, dim=-1)
            num_decoder_layers = nao_decoder.layers
            new_hidden   = (new_arch_emb.unsqueeze(0).expand(num_decoder_layers, -1, -1).contiguous(),
                            new_arch_emb.unsqueeze(0).expand(num_decoder_layers, -1, -1).contiguous())

            _, _, ret  = nao_decoder(None, new_hidden, new_enc_outputs)
            new_arch   = torch.stack(ret['sequence'], 0).permute(1, 0, 2)

            arch_data = new_arch.data.squeeze(-1)
            if arch_data.dim() == 1:
                arch_data = arch_data.unsqueeze(0)

            for seq in arch_data.tolist():
                new_arch_seqs.append([int(x) for x in seq])

            attempts += batch_size

        logging.info('PPO generated %d new architectures', len(new_arch_seqs))
        return new_arch_seqs[:n_new]
