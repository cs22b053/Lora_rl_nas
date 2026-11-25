from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import NAOCell, RNNModel
from rnn_lora_cell import NAOCellSearchLoRA
from utils import batchify, get_batch, repackage_hidden, embedded_dropout, mask2d, STEPS
import gc

INITRANGE = 0.04


class NAOCellSearch(NAOCell):

  def __init__(self, ninp, nhid, dropouth, dropoutx, drop_path=0.0):
    super(NAOCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, arch=None)
    self._W0 = nn.Parameter(torch.Tensor(ninp + nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE))
    self._Ws = nn.ParameterList([
      nn.Parameter(torch.Tensor(4 * (i+1) * nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE)) for i in range(STEPS)
    ])
    self.drop_path = drop_path

  def _compute_init_state(self, x, h_prev, x_mask, h_mask):
    if self.training:
      xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
    else:
      xh_prev = torch.cat([x, h_prev], dim=-1)
    c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1)
    c0 = c0.sigmoid()
    h0 = h0.tanh()
    s0 = h_prev + c0 * (h0-h_prev)
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
      if self.training:
        self.apply_drop_path(self._Ws[i])
      w = self._Ws[i][w_start:w_end, :]
      if self.training:
        ch = (s_prev * h_mask).mm(w)
        self._Ws[i].requires_grad = True
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
      x_mask = mask2d(B, inputs.size(2), keep_prob=1.-self.dropoutx)
      h_mask = mask2d(B, hidden.size(2), keep_prob=1.-self.dropouth)
    else:
      x_mask = h_mask = None

    hidden = hidden[0]
    hiddens = []
    for t in range(T):
      hidden = self.cell(inputs[t], hidden, x_mask, h_mask, arch)
      hiddens.append(hidden)
    hiddens = torch.stack(hiddens)
    return hiddens, hiddens[-1].unsqueeze(0)

  def apply_drop_path(self, w):
    if np.random.random() < self.drop_path:
      w.requires_grad = False

class RNNModelSearch(RNNModel):

    def __init__(self, *args):
        super(RNNModelSearch, self).__init__(*args, cell_cls=NAOCellSearch)
        self._args = args

    def clone(self):
        model_clone = RNNModelSearch(*self._args)
        for x, y in zip(model_clone.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_clone

    def _loss(self, hidden, input, target):
      log_prob, hidden_next = self(input, hidden, return_h=False)
      loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), target)
      return loss, hidden_next
    
    def arch(self):
      return self.last_arch
    
    def forward(self, input, hidden, arch=None, return_h=False):
        assert arch is not None
        self.last_arch = arch
        batch_size = input.size(1)

        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l], arch=arch)
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        logit = self.decoder(output.view(-1, self.ninp))
        log_prob = nn.functional.log_softmax(logit, dim=-1)
        model_output = log_prob
        model_output = model_output.view(-1, batch_size, self.ntoken)

        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden


class RNNModelSearchLoRA(RNNModel):
    """RNNModelSearch with LoRA-adapted weight sharing cell."""

    def __init__(self, ntoken, ninp, nhid, nhidlast,
                 dropout=0.5, dropouth=0.5, dropoutx=0.5, dropouti=0.5, dropoute=0.1,
                 drop_path=0.0, lora_rank=16):
        # We bypass RNNModel.__init__ and manually set up with LoRA cell
        nn.Module.__init__(self)
        from utils import LockedDropout
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)

        assert ninp == nhid == nhidlast
        self.rnns = torch.nn.ModuleList([
            NAOCellSearchLoRA(ninp, nhid, dropouth, dropoutx, drop_path=drop_path, lora_rank=lora_rank)
        ])

        self.decoder = nn.Linear(ninp, ntoken)
        self.decoder.weight = self.encoder.weight
        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.ntoken = ntoken
        self.cell_cls = NAOCellSearchLoRA
        self.lora_rank = lora_rank
        self._init_args = (ntoken, ninp, nhid, nhidlast, dropout, dropouth, dropoutx,
                           dropouti, dropoute, drop_path, lora_rank)

    def clone(self):
        model_clone = RNNModelSearchLoRA(*self._init_args)
        return model_clone

    def _loss(self, hidden, input, target):
        log_prob, hidden_next = self(input, hidden, return_h=False)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), target)
        return loss, hidden_next

    def arch(self):
        return self.last_arch

    def forward(self, input, hidden, arch=None, return_h=False):
        assert arch is not None
        self.last_arch = arch
        batch_size = input.size(1)

        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l], arch=arch)
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        logit = self.decoder(output.view(-1, self.ninp))
        log_prob = nn.functional.log_softmax(logit, dim=-1)
        model_output = log_prob
        model_output = model_output.view(-1, batch_size, self.ntoken)

        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden


def train(train_data, model, parallel_model, optimizer, params, epoch):
  assert params['batch_size'] % params['small_batch_size'] == 0, 'batch_size must be divisible by small_batch_size'
  
  # Turn on training mode which enables dropout.
  total_loss = 0
  start_time = time.time()
  arch_pool = params['arch_pool']
  assert arch_pool is not None
  N = len(arch_pool)
  
  hidden = [model.init_hidden(params['small_batch_size']) for _ in range(params['batch_size'] // params['small_batch_size'])]
  batch, i = 0, 0
  while i < train_data.size(0) - 1 - 1:
    bptt = params['bptt'] if np.random.random() < 0.95 else params['bptt'] / 2.
    seq_len = int(bptt)
    
    lr2 = optimizer.param_groups[0]['lr']
    optimizer.param_groups[0]['lr'] = lr2 * seq_len / params['bptt']
    model.train()
    
    data, targets = get_batch(train_data, i, params['bptt'], seq_len=seq_len)
    sample_arch_index = np.random.randint(N)
    sample_arch = arch_pool[sample_arch_index]
    optimizer.zero_grad()
    
    start, end, s_id = 0, params['small_batch_size'], 0
    while start < params['batch_size']:
      cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)
      
      # Starting each batch, we detach the hidden state from how it was previously produced.
      # If we didn't, the model would try backpropagating all the way to start of the dataset.
      # assuming small_batch_size = batch_size so we don't accumulate gradients
      optimizer.zero_grad()
      hidden[s_id] = repackage_hidden(hidden[s_id])
      
      log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = parallel_model(cur_data, hidden[s_id], sample_arch, return_h=True)
      raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)
      
      loss = raw_loss
      # Activiation Regularization
      if params['alpha'] > 0:
        loss = loss + sum(params['alpha'] * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
      # Temporal Activation Regularization (slowness)
      loss = loss + sum(params['beta'] * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
      loss *= params['small_batch_size'] / params['batch_size']
      total_loss += raw_loss.data * params['small_batch_size'] / params['batch_size']
      loss.backward()
      
      s_id += 1
      start = end
      end = start + params['small_batch_size']
      
      gc.collect()
    
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
    torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
    optimizer.step()
    
    optimizer.param_groups[0]['lr'] = lr2
    if batch % params['log_interval'] == 0 and batch > 0:
      logging.info(parallel_model.arch())
      cur_loss = total_loss.item() / params['log_interval']
      elapsed = time.time() - start_time
      ppl = math.exp(min(cur_loss, 700))  # Prevent OverflowError
      logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                   'loss {:5.2f} | ppl {:8.2f}'.format(
        epoch, batch, len(train_data) // params['bptt'], optimizer.param_groups[0]['lr'],
                      elapsed * 1000 / params['log_interval'], cur_loss, ppl))
      total_loss = 0
      start_time = time.time()
    batch += 1
    i += seq_len


def evaluate(data_source, model, parallel_model, params, batch_size=10, num_eval_batches=5):
  # Turn on evaluation mode which disables dropout.
  arch_pool = params['arch_pool']
  logging.info('Evaluating on {} archs with {} batches each'.format(len(arch_pool), num_eval_batches))
  start_time = time.time()
  valid_score_list = []
  max_batch_idx = data_source.size(0) // params['bptt']
  with torch.no_grad():
    for arch in arch_pool:
      model.eval()
      total_loss = 0.0
      total_tokens = 0
      for _ in range(num_eval_batches):
        hidden = model.init_hidden(batch_size)
        batch_idx = np.random.randint(0, max_batch_idx)
        data, targets = get_batch(data_source, batch_idx * params['bptt'], params['bptt'], evaluation=True)
        targets = targets.view(-1)
        log_prob, hidden = parallel_model(data, hidden, arch)
        loss = F.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).item()
        n_tokens = targets.size(0)
        total_loss += loss * n_tokens
        total_tokens += n_tokens
      avg_loss = total_loss / max(total_tokens, 1)
      valid_score_list.append(avg_loss)
  eval_time = time.time() - start_time
  mean_valid_score = np.mean(valid_score_list)
  logging.info('Mean loss {:5.2f} | mean ppl {:8.2f} | time {:5.2f} secs'.format(
      mean_valid_score, math.exp(min(mean_valid_score, 700)), eval_time))
  return valid_score_list