"""
PPO-based architecture controller for NAO RNN search.
Instead of fixed gradient steps in latent space, a PPO agent
learns WHICH architecture to propose next, guided by the NAO
predictor as a reward signal (no child training needed per step).

Adapted from CNN version for RNN search:
- Default embedding_dim=96 (matching RNN controller encoder_hidden_size)
- Default num_nodes=11 (STEPS in RNN), num_ops=4 (tanh/relu/sigmoid/identity)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from torch.distributions import Categorical


class ArchitecturePolicy(nn.Module):
    """
    Actor-Critic network that operates in NAO's latent space.
    
    State:  arch_embedding (from NAO encoder) + predicted_value
    Action: step_direction_vector (continuous) that modifies the embedding
    """
    def __init__(self, embedding_dim=96, hidden_dim=256, num_nodes=11, num_ops=4):
        super(ArchitecturePolicy, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.num_ops = num_ops
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Actor: outputs mean and log_std of step direction
        # Step direction is a vector in embedding space
        self.actor_mean = nn.Linear(hidden_dim, embedding_dim)
        self.actor_log_std = nn.Parameter(torch.full((embedding_dim,), -0.5))  # std≈0.6, enables meaningful latent space exploration
        
        # Critic: estimates value of current state
        self.critic = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        # Small initialization for actor output
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
    
    def forward(self, arch_emb, pred_acc):
        """
        Args:
            arch_emb: [B, embedding_dim]
            pred_acc: [B, 1]
        Returns:
            action_mean, action_std, value
        """
        if pred_acc.dim() == 1:
            pred_acc = pred_acc.unsqueeze(-1)
        
        x = torch.cat([arch_emb, pred_acc], dim=-1)
        features = self.backbone(x)
        
        action_mean = self.actor_mean(features)
        # Keep exploration alive but bounded
        action_std = torch.exp(self.actor_log_std).clamp(0.05, 2.0)
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action_and_value(self, arch_emb, pred_acc, action=None):
        action_mean, action_std, value = self(arch_emb, pred_acc)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(-1)  # Sum over action dims
        entropy = dist.entropy().sum(-1)
        
        return action, log_prob, entropy, value
