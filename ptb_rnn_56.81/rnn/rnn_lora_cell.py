"""
LoRA-adapted NAOCellSearch for RNN architecture search.

Key idea: Each step i in the RNN cell has a weight matrix _Ws[i] of shape
[4*(i+1)*nhid, 2*nhid]. We add per-step LoRA adapters:
  effective_W = _Ws[i] + lora_A[i] @ lora_B[i]

lora_B is zero-initialized so LoRA starts as identity (no change from base).
This mirrors the CNN LoRA approach where per-input low-rank adapters
specialize the shared weights for different architecture choices.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import NAOCell
from utils import mask2d, STEPS

INITRANGE = 0.04


class NAOCellSearchLoRA(NAOCell):
    """NAOCellSearch with LoRA adapters on _W0 and _Ws parameters."""

    def __init__(self, ninp, nhid, dropouth, dropoutx, drop_path=0.0, lora_rank=16):
        super(NAOCellSearchLoRA, self).__init__(ninp, nhid, dropouth, dropoutx, arch=None)
        self.lora_rank = lora_rank
        self.drop_path = drop_path

        # Base weights (same as NAOCellSearch)
        self._W0 = nn.Parameter(
            torch.Tensor(ninp + nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE)
        )
        self._Ws = nn.ParameterList([
            nn.Parameter(
                torch.Tensor(4 * (i + 1) * nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE)
            ) for i in range(STEPS)
        ])

        # LoRA adapters for _W0: [ninp+nhid, rank] @ [rank, 2*nhid]
        self._W0_lora_A = nn.Parameter(torch.zeros(ninp + nhid, lora_rank))
        self._W0_lora_B = nn.Parameter(torch.zeros(lora_rank, 2 * nhid))

        # LoRA adapters for each _Ws[i]: [4*(i+1)*nhid, rank] @ [rank, 2*nhid]
        self._Ws_lora_A = nn.ParameterList([
            nn.Parameter(torch.zeros(4 * (i + 1) * nhid, lora_rank))
            for i in range(STEPS)
        ])
        self._Ws_lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(lora_rank, 2 * nhid))
            for i in range(STEPS)
        ])

        self._init_lora()

    def _init_lora(self):
        """Initialize LoRA: A with kaiming_uniform_, B with zeros.
        
        Matches the CNN LoRA init (operations.py). B=0 ensures LoRA starts
        as identity. With lora_lr_multiplier=2.0 (effective lr=0.5), kaiming
        init is stable for RNN.
        """
        nn.init.kaiming_uniform_(self._W0_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self._W0_lora_B)
        for i in range(STEPS):
            nn.init.kaiming_uniform_(self._Ws_lora_A[i], a=math.sqrt(5))
            nn.init.zeros_(self._Ws_lora_B[i])

    def _get_effective_W0(self):
        """Get W0 + LoRA delta (no 1/rank scaling — matches CNN)."""
        delta = self._W0_lora_A @ self._W0_lora_B
        return self._W0 + delta

    def _get_effective_Ws(self, i):
        """Get _Ws[i] + LoRA delta for step i (no 1/rank scaling — matches CNN)."""
        delta = self._Ws_lora_A[i] @ self._Ws_lora_B[i]
        return self._Ws[i] + delta

    def _compute_init_state(self, x, h_prev, x_mask, h_mask):
        if self.training:
            xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
        else:
            xh_prev = torch.cat([x, h_prev], dim=-1)
        W0_eff = self._get_effective_W0()
        c0, h0 = torch.split(xh_prev.mm(W0_eff), self.nhid, dim=-1)
        c0 = c0.sigmoid()
        h0 = h0.tanh()
        s0 = h_prev + c0 * (h0 - h_prev)
        return s0

    def cell(self, x, h_prev, x_mask, h_mask, arch=None):
        assert arch is not None
        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)

        states = [s0]
        for i in range(STEPS):
            pred, act = arch[2 * i], arch[2 * i + 1]
            s_prev = states[pred]
            w_start = (pred * 4 + act) * self.nhid
            w_end = w_start + self.nhid

            # Get effective weight with LoRA
            Ws_eff = self._get_effective_Ws(i)

            if self.training:
                self._apply_drop_path(Ws_eff)

            w = Ws_eff[w_start:w_end, :]
            if self.training:
                ch = (s_prev * h_mask).mm(w)
            else:
                ch = s_prev.mm(w)
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()
            fn = self._get_activation(act)
            h = fn(h)
            s = s_prev + c * (h - s_prev)
            states.append(s)
        output = torch.mean(torch.stack(states, -1), dim=-1)
        return output

    def forward(self, inputs, hidden, arch=None):
        T, B = inputs.size(0), inputs.size(1)

        if self.training:
            x_mask = mask2d(B, inputs.size(2), keep_prob=1. - self.dropoutx)
            h_mask = mask2d(B, hidden.size(2), keep_prob=1. - self.dropouth)
        else:
            x_mask = h_mask = None

        hidden = hidden[0]
        hiddens = []
        for t in range(T):
            hidden = self.cell(inputs[t], hidden, x_mask, h_mask, arch)
            hiddens.append(hidden)
        hiddens = torch.stack(hiddens)
        return hiddens, hiddens[-1].unsqueeze(0)

    def _apply_drop_path(self, w):
        if np.random.random() < self.drop_path:
            w.requires_grad = False
