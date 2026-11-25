from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import subprocess
import math
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import copy
import json
import model_search
import data
from utils import batchify, save_checkpoint, generate_arch, parse_arch_to_seq, parse_seq_to_arch, build_arch, normalize_target, STEPS
import controller
from ArchitecturePolicy import ArchitecturePolicy
from PPOArchTrainer import PPOArchTrainer

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 Language Model')
parser.add_argument('--data_dir', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model_dir', type=str,  default='models',
                    help='path to save the final model')
parser.add_argument('--child_emb_size', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--child_nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--child_nhidlast', type=int, default=300,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--child_arch_pool', type=str, default=None,
                    help='Archs to sample from. If None for default, then randomly sampled arch is used.')
parser.add_argument('--child_lr', type=float, default=0.25,
                    help='initial learning rate')
parser.add_argument('--child_clip', type=float, default=10.0,
                    help='gradient clipping')
parser.add_argument('--child_train_epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--child_batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--child_eval_batch_size', type=int, default=256,
                    help='eval batch size')
parser.add_argument('--child_bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--child_dropout', type=float, default=0.75,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--child_dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--child_dropoutx', type=float, default=0.75,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')
parser.add_argument('--child_dropouti', type=float, default=0.2,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--child_dropoute', type=float, default=0,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--child_drop_path', type=float, default=0.0,
                    help='drop a path.')
parser.add_argument('--child_log_interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--child_alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--child_beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--child_weight_decay', type=float, default=5e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--child_eval_every_epochs', type=str, default='100',
                    help='The number of epochs to run in between evaluations.')
parser.add_argument('--child_small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--child_max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
parser.add_argument('--controller_shuffle', action='store_true', default=False)
parser.add_argument('--controller_max_new_archs', type=int, default=300)
parser.add_argument('--controller_max_step_size', type=int, default=30)
parser.add_argument('--controller_num_seed_arch', type=int, default=1000)
parser.add_argument('--controller_encoder_num_layers', type=int, default=1)
parser.add_argument('--controller_encoder_hidden_size', type=int, default=96)
parser.add_argument('--controller_encoder_emb_size', type=int, default=96)
parser.add_argument('--controller_mlp_num_layers', type=int, default=0)
parser.add_argument('--controller_mlp_hidden_size', type=int, default=200)
parser.add_argument('--controller_decoder_num_layers', type=int, default=1)
parser.add_argument('--controller_decoder_hidden_size', type=int, default=96)
parser.add_argument('--controller_source_length', type=int, default=22)
parser.add_argument('--controller_encoder_length', type=int, default=22)
parser.add_argument('--controller_decoder_length', type=int, default=22)
parser.add_argument('--controller_encoder_dropout', type=float, default=0.1)
parser.add_argument('--controller_mlp_dropout', type=float, default=0)
parser.add_argument('--controller_decoder_dropout', type=float, default=0.0)
parser.add_argument('--controller_weight_decay', type=float, default=1e-4)
parser.add_argument('--controller_encoder_vocab_size', type=int, default=16)
parser.add_argument('--controller_decoder_vocab_size', type=int, default=16)
parser.add_argument('--controller_trade_off', type=float, default=0.8)
parser.add_argument('--controller_train_epochs', type=int, default=1000)
parser.add_argument('--controller_save_frequency', type=int, default=100)
parser.add_argument('--controller_batch_size', type=int, default=100)
parser.add_argument('--controller_lr', type=float, default=0.001)
parser.add_argument('--controller_optimizer', type=str, default='adam')
parser.add_argument('--controller_max_gradient_norm', type=float, default=5.0)
parser.add_argument('--controller_predict_beam_width', type=int, default=0)
parser.add_argument('--controller_predict_lambda', type=float, default=1)
parser.add_argument('--seed', type=int, default=1267,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--single_gpu', default=True, action='store_false',
                    help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')

# === NEW: LoRA parameters ===
parser.add_argument('--lora_rank', type=int, default=16,
                    help='Rank of LoRA adapters in WS cell (0 to disable LoRA)')
parser.add_argument('--lora_lr_multiplier', type=float, default=2.0,
                    help='LR multiplier for LoRA adapter parameters (default 2.0 for RNN; CNN uses 10.0 with 10x lower base LR)')

# === NEW: CVaR RL parameters ===
parser.add_argument('--use_cvar_rl', action='store_true', default=True,
                    help='Enable CVaR RL-based PPO architecture search')
parser.add_argument('--no_cvar_rl', dest='use_cvar_rl', action='store_false',
                    help='Disable CVaR RL-based PPO architecture search')
parser.add_argument('--cvar_alpha', type=float, default=0.2,
                    help='CVaR risk level (fraction of worst-case samples)')
parser.add_argument('--ppo_lr', type=float, default=3e-4,
                    help='Learning rate for PPO policy')
parser.add_argument('--ppo_entropy_coef', type=float, default=0.1,
                    help='Entropy coefficient for PPO (exploration, higher=more diverse)')
parser.add_argument('--ppo_max_step_magnitude', type=float, default=1.0,
                    help='Maximum step magnitude in latent space for PPO')
parser.add_argument('--num_search_iterations', type=int, default=4,
                    help='Number of search iterations (PPO+CVaR needs >= 3)')

args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.model_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
  torch.cuda.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
  if not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  else:
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled=True
    torch.cuda.manual_seed_all(args.seed)


def train():
  child_params = get_child_model_params()
  controller_params = get_controller_params()
  corpus = data.Corpus(child_params['data_dir'])
  eval_batch_size = child_params['eval_batch_size']

  train_data = batchify(corpus.train, child_params['batch_size'], child_params['cuda'])
  val_data = batchify(corpus.valid, eval_batch_size, child_params['cuda'])
  ntokens = len(corpus.dictionary)
  
  if os.path.exists(os.path.join(child_params['model_dir'], 'model.pt')):
    print("Found model.pt in {}, automatically continue training.".format(os.path.join(child_params['model_dir'])))
    continue_train_child = True
  else:
    continue_train_child = False
  
  if continue_train_child:
    child_model = torch.load(os.path.join(child_params['model_dir'], 'model.pt'))
  else:
    # === Use LoRA model if lora_rank > 0 ===
    if args.lora_rank > 0:
      logging.info('Using LoRA-adapted RNN model with rank={}'.format(args.lora_rank))
      child_model = model_search.RNNModelSearchLoRA(
          ntokens, child_params['emsize'], child_params['nhid'], child_params['nhidlast'],
          child_params['dropout'], child_params['dropouth'], child_params['dropoutx'],
          child_params['dropouti'], child_params['dropoute'], child_params['drop_path'],
          lora_rank=args.lora_rank)
    else:
      logging.info('Using standard RNN model (no LoRA)')
      child_model = model_search.RNNModelSearch(
          ntokens, child_params['emsize'], child_params['nhid'], child_params['nhidlast'],
          child_params['dropout'], child_params['dropouth'], child_params['dropoutx'],
          child_params['dropouti'], child_params['dropoute'], child_params['drop_path'])

  if os.path.exists(os.path.join(controller_params['model_dir'], 'model.pt')):
    print("Found model.pt in {}, automatically continue training.".format(os.path.join(child_params['model_dir'])))
    continue_train_controller = True
  else:
    continue_train_controller = False

  if continue_train_controller:
    controller_model = torch.load(os.path.join(controller_params['model_dir'], 'model.pt'))
  else:
    controller_model = controller.Controller(controller_params)

  size = 0
  for p in child_model.parameters():
    size += p.nelement()
  logging.info('child model param size: {}'.format(size))
  
  # === Log LoRA parameters separately ===
  if args.lora_rank > 0:
    lora_size = 0
    base_size = 0
    for name, p in child_model.named_parameters():
      if 'lora_A' in name or 'lora_B' in name:
        lora_size += p.nelement()
      else:
        base_size += p.nelement()
    logging.info('  Base params: {}, LoRA params: {} ({:.2f}% of total)'.format(
        base_size, lora_size, 100.0 * lora_size / (base_size + lora_size)))

  size = 0
  for p in controller_model.parameters():
    size += p.nelement()
  logging.info('controller model param size: {}'.format(size))
  

  if args.cuda:
    if args.single_gpu:
      parallel_child_model = child_model.cuda()
      parallel_controller_model = controller_model.cuda()
    else:
      parallel_child_model = nn.DataParallel(child_model, dim=1).cuda()
      parallel_controller_model = nn.DataParallel(controller_model, dim=1).cuda()
  else:
    parallel_child_model = child_model
    parallel_controller_model = controller_model
    

  total_params = sum(x.data.nelement() for x in child_model.parameters())
  logging.info('Args: {}'.format(args))
  logging.info('Child Model total parameters: {}'.format(total_params))
  total_params = sum(x.data.nelement() for x in controller_model.parameters())
  logging.info('Args: {}'.format(args))
  logging.info('Controller Model total parameters: {}'.format(total_params))

  # Loop over epochs.

  if continue_train_child:
    optimizer_state = torch.load(os.path.join(child_params['model_dir'], 'optimizer.pt'))
    if 't0' in optimizer_state['param_groups'][0]:
      child_optimizer = torch.optim.ASGD(child_model.parameters(), lr=child_params['lr'], t0=0, lambd=0., weight_decay=child_params['wdecay'])
    else:
      child_optimizer = torch.optim.SGD(child_model.parameters(), lr=child_params['lr'], weight_decay=child_params['wdecay'])
    child_optimizer.load_state_dict(optimizer_state)
    child_epoch = torch.load(os.path.join(child_params['model_dir'], 'misc.pt'))['epoch'] - 1
  else:
    # === LoRA-aware optimizer with separate param groups ===
    if args.lora_rank > 0:
      lora_params = []
      base_params = []
      for name, param in child_model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
          lora_params.append(param)
        else:
          base_params.append(param)
      child_optimizer = torch.optim.SGD([
        {'params': base_params},
        {'params': lora_params, 'lr': child_params['lr'] * args.lora_lr_multiplier},
      ], lr=child_params['lr'], weight_decay=child_params['wdecay'])
      logging.info('Using LoRA-aware optimizer: base_lr={}, lora_lr={}'.format(
          child_params['lr'], child_params['lr'] * args.lora_lr_multiplier))
    else:
      child_optimizer = torch.optim.SGD(child_model.parameters(), lr=child_params['lr'], weight_decay=child_params['wdecay'])
    child_epoch = 0
  
  if continue_train_controller:
    optimizer_state = torch.load(os.path.join(controller_params['model_dir'], 'optimizer.pt'))
    controller_optimizer = torch.optim.Adam(controller_model.parameters(), lr=controller_params['lr'], weight_decay=controller_params['weight_decay'])
    controller_optimizer.load_state_dict(optimizer_state)
    controller_epoch = torch.load(os.path.join(controller_params['model_dir'], 'misc.pt'))['epoch'] - 1
  else:
    controller_optimizer = torch.optim.Adam(controller_model.parameters(), lr=controller_params['lr'], weight_decay=controller_params['weight_decay'])
    controller_epoch = 0

  # === Initialize PPO+CVaR if enabled ===
  ppo_policy = None
  ppo_trainer = None
  if args.use_cvar_rl:
    logging.info('Initializing CVaR RL-based PPO architecture search')
    ppo_policy = ArchitecturePolicy(
      embedding_dim=controller_params['encoder_hidden_size'],
      hidden_dim=256,
      num_nodes=STEPS,
      num_ops=4,
    )
    if args.cuda:
      ppo_policy = ppo_policy.cuda()
    
    ppo_trainer = PPOArchTrainer(
      controller_model=controller_model,
      policy=ppo_policy,
      embedding_dim=controller_params['encoder_hidden_size'],
      lr=args.ppo_lr,
      clip_epsilon=0.2,
      value_coef=0.5,
      entropy_coef=args.ppo_entropy_coef,
      n_epochs=10,
      batch_size=64,
      gamma=0.99,
      gae_lambda=0.95,
      max_step_magnitude=args.ppo_max_step_magnitude,
      cvar_alpha=args.cvar_alpha,
    )
    ppo_param_size = sum(p.nelement() for p in ppo_policy.parameters())
    logging.info('PPO Architecture Policy param size: {}'.format(ppo_param_size))

  eval_every_epochs = child_params['eval_every_epochs']
  search_iteration = 0
  
  while True:
    # Train child model
    if child_params['arch_pool'] is None:
      arch_pool = generate_arch(controller_params['num_seed_arch']) #[[arch]]
      child_params['arch_pool'] = arch_pool
    child_params['arch'] = None
    
    if isinstance(eval_every_epochs, int):
      child_params['eval_every_epochs'] = eval_every_epochs
    else:
      eval_every_epochs = list(map(int, eval_every_epochs))
      for index, e in enumerate(eval_every_epochs):
        if child_epoch < e:
          child_params['eval_every_epochs'] = e
          break
    
    for e in range(child_params['eval_every_epochs']):
      child_epoch += 1
      model_search.train(train_data, child_model, parallel_child_model, child_optimizer, child_params, child_epoch)
      if child_epoch % child_params['eval_every_epochs'] == 0:
        save_checkpoint(child_model, child_optimizer, child_epoch, child_params['model_dir'])
        logging.info('Saving Model!')
      if child_epoch >= child_params['train_epochs']:
        break
    
    # Evaluate seed archs
    valid_accuracy_list = model_search.evaluate(val_data, child_model, parallel_child_model, child_params, eval_batch_size)

    # Output archs and evaluated error rate
    old_archs = child_params['arch_pool']
    old_archs_perf = valid_accuracy_list
    
    old_archs_sorted_indices = np.argsort(old_archs_perf)
    old_archs = np.array(old_archs)[old_archs_sorted_indices].tolist()
    old_archs_perf = np.array(old_archs_perf)[old_archs_sorted_indices].tolist()
    with open(os.path.join(child_params['model_dir'], 'arch_pool.{}'.format(child_epoch)), 'w') as fa:
      with open(os.path.join(child_params['model_dir'], 'arch_pool.perf.{}'.format(child_epoch)), 'w') as fp:
        with open(os.path.join(child_params['model_dir'], 'arch_pool'), 'w') as fa_latest:
          with open(os.path.join(child_params['model_dir'], 'arch_pool.perf'),'w') as fp_latest:
            for arch, perf in zip(old_archs, old_archs_perf):
              arch = ' '.join(map(str, arch))
              fa.write('{}\n'.format(arch))
              fa_latest.write('{}\n'.format(arch))
              fp.write('{}\n'.format(perf))
              fp_latest.write('{}\n'.format(perf))
      
    if child_epoch >= child_params['train_epochs']:
      logging.info('Training finished!')
      break

    # Train Encoder-Predictor-Decoder
    # [[arch]]
    encoder_input = list(map(lambda x : parse_arch_to_seq(x), old_archs))
    encoder_target = normalize_target(old_archs_perf)
    decoder_target = copy.copy(encoder_input)
    controller_params['batches_per_epoch'] = math.ceil(len(encoder_input) / controller_params['batch_size'])
    controller_epoch = controller.train(encoder_input, encoder_target, decoder_target, controller_model, parallel_controller_model, controller_optimizer, controller_params, controller_epoch)
      
    
    # === Generate new archs with hybrid NAO + PPO+CVaR ===
    new_archs = []
    controller_params['predict_lambda'] = 0
    top100_archs = list(map(lambda x : parse_arch_to_seq(x), old_archs[:100]))
    max_step_size = controller_params['max_step_size']

    if search_iteration == 0 or not args.use_cvar_rl:
      # Iteration 0 or CVaR disabled: NAO gradient step only
      # Use adaptive (Fibonacci-like) step sizes for better latent space coverage
      logging.info('=== NAO Gradient-Only Architecture Search (Iteration %d) ===', search_iteration)
      step_sizes = [1, 2, 3, 5, 8, 13, 21, 30]
      for predict_step_size in step_sizes:
        controller_params['predict_lambda'] = predict_step_size
        new_arch = controller.infer(top100_archs, controller_model, parallel_controller_model, controller_params, direction='-')
        for arch in new_arch:
          if arch not in encoder_input and arch not in new_archs:
            new_archs.append(arch)
          if len(new_archs) >= controller_params['max_new_archs']:
            break
        logging.info('{} new archs generated now (step_size={})'.format(len(new_archs), predict_step_size))
        if len(new_archs) >= controller_params['max_new_archs']:
          break
    else:
      # Iteration 1+: Hybrid approach — 50% NAO gradient + 50% PPO+CVaR
      logging.info('=== Hybrid NAO+PPO Architecture Search (Iteration %d) ===', search_iteration)

      # Part A: NAO gradient-based generation (50% of new archs)
      nao_target = controller_params['max_new_archs'] // 2
      step_sizes = [1, 2, 3, 5, 8, 13, 21, 30]
      for predict_step_size in step_sizes:
        controller_params['predict_lambda'] = predict_step_size
        new_arch = controller.infer(top100_archs, controller_model, parallel_controller_model, controller_params, direction='-')
        for arch in new_arch:
          if arch not in encoder_input and arch not in new_archs:
            new_archs.append(arch)
          if len(new_archs) >= nao_target:
            break
        if len(new_archs) >= nao_target:
          break
      logging.info('NAO gradient generated %d archs', len(new_archs))

      # Part B: PPO+CVaR generation (remaining 50%)
      logging.info('Starting PPO+CVaR rollout and update')
      ppo_trainer.collect_rollout(top100_archs, n_steps=2048)

      for ppo_update in range(5):
        metrics = ppo_trainer.update()
        logging.info(
          'PPO update %d: policy_loss=%.4f value_loss=%.4f '
          'entropy=%.4f clip_frac=%.4f cvar=%.4f',
          ppo_update,
          metrics['policy_loss'], metrics['value_loss'],
          metrics['entropy'], metrics['clip_frac'],
          metrics.get('cvar', 0.0)
        )
        if metrics['clip_frac'] > 0.2:
          logging.info('Early stopping PPO (clip_frac=%.3f)', metrics['clip_frac'])
          break

      ppo_archs = ppo_trainer.generate_architectures(
        top100_archs,
        n_new=controller_params['max_new_archs'] - len(new_archs),
      )
      for arch in ppo_archs:
        if arch not in new_archs:
          new_archs.append(arch)
      logging.info('After PPO: total %d new archs', len(new_archs))

      # Fallback if PPO generated too few
      if len(new_archs) < controller_params['max_new_archs']:
        logging.warning('PPO generated %d/%d, filling with NAO gradient step',
                        len(new_archs), controller_params['max_new_archs'])
        predict_step_size = 0
        while len(new_archs) < controller_params['max_new_archs']:
          predict_step_size += 1
          controller_params['predict_lambda'] = predict_step_size
          extra = controller.infer(top100_archs, controller_model, parallel_controller_model, controller_params, direction='-')
          for arch in extra:
            if arch not in new_archs:
              new_archs.append(arch)
          if predict_step_size > max_step_size:
            break

    # Add noise-perturbed variants of top architectures for diversity
    for arch_seq in top100_archs[:20]:
      for _ in range(3):
        noisy = list(arch_seq)
        # Randomly perturb 2 positions
        for _ in range(2):
          pos = np.random.randint(0, len(noisy))
          step_idx = pos // 2  # which step this belongs to (0-10)
          if pos % 2 == 0:  # node index position
            # Step i can reference nodes 0..i, encoded as 1..i+1
            max_token = step_idx + 1
            noisy[pos] = np.random.randint(1, max_token + 1)
          else:  # operation position
            noisy[pos] = np.random.randint(12, 16)  # ops encoded as 12-15
        if noisy not in new_archs:
          new_archs.append(noisy)

    # Validate generated architectures
    logging.info('Validating %d new architectures...', len(new_archs))
    valid_new_archs = []
    expected_seq_length = 2 * STEPS  # 22 tokens for 11 steps

    for idx, arch_seq in enumerate(new_archs):
      if not isinstance(arch_seq, list):
        continue
      if len(arch_seq) != expected_seq_length:
        logging.warning('Arch %d wrong length %d != %d, skipping', idx, len(arch_seq), expected_seq_length)
        continue

      # Validate and clamp node indices and ops
      fixed_seq = list(arch_seq)
      for i in range(STEPS):
        # Node index at position 2*i: should be in [1, i+1] (encoded as 1-based)
        node_pos = 2 * i
        max_token = i + 1  # can reference 0..i, encoded as 1..i+1
        val = fixed_seq[node_pos]
        if val < 1 or val > max_token:
          fixed_seq[node_pos] = max(1, min(val, max_token))

        # Op at position 2*i+1: should be in [12, 15]
        op_pos = 2 * i + 1
        val = fixed_seq[op_pos]
        if val < 12 or val > 15:
          fixed_seq[op_pos] = max(12, min(val, 15))

      valid_new_archs.append(fixed_seq)

    if len(valid_new_archs) == 0:
      logging.error('No valid archs generated! Falling back to random generation.')
      valid_new_archs = [parse_arch_to_seq(a) for a in generate_arch(50)]

    logging.info('%d valid new archs after validation', len(valid_new_archs))

    #[[arch]]
    new_archs = list(map(lambda x: parse_seq_to_arch(x), valid_new_archs)) #[[arch]]
    num_new_archs = len(new_archs)
    logging.info("Generate {} new archs".format(num_new_archs))
    random_new_archs = generate_arch(50)
    new_arch_pool = old_archs[:len(old_archs)-num_new_archs-50] + new_archs + random_new_archs
    logging.info("Totally {} archs now to train".format(len(new_arch_pool)))
    child_params['arch_pool'] = new_arch_pool
    with open(os.path.join(child_params['model_dir'], 'arch_pool'), 'w') as f:
      for arch in new_arch_pool:
        arch = ' '.join(map(str, arch))
        f.write('{}\n'.format(arch))
    
    search_iteration += 1
    if search_iteration >= args.num_search_iterations:
      logging.info('Reached max search iterations (%d), finishing.', args.num_search_iterations)

  logging.info('LoRA + CVaR RL combined search complete')

          
def get_child_model_params():
  params = {
    'data_dir': args.data_dir,
    'model_dir': os.path.join(args.model_dir, 'child'),
    'arch_pool': args.child_arch_pool,
    'emsize': args.child_emb_size,
    'nhid': args.child_nhid,
    'nhidlast': args.child_emb_size if args.child_nhidlast < 0 else args.child_nhidlast,
    'lr': args.child_lr,
    'clip': args.child_clip,
    'train_epochs': args.child_train_epochs,
    'eval_every_epochs': eval(args.child_eval_every_epochs),
    'batch_size': args.child_batch_size,
    'eval_batch_size': args.child_eval_batch_size,
    'bptt': args.child_bptt,
    'dropout': args.child_dropout,
    'dropouth': args.child_dropouth,
    'dropoutx': args.child_dropoutx,
    'dropouti': args.child_dropouti,
    'dropoute': args.child_dropoute,
    'drop_path': args.child_drop_path,
    'seed': args.seed,
    'nonmono': args.nonmono,
    'log_interval': args.child_log_interval,
    'alpha': args.child_alpha,
    'beta': args.child_beta,
    'wdecay': args.child_weight_decay,
    'small_batch_size': args.child_batch_size if args.child_small_batch_size < 0 else args.child_small_batch_size,
    'max_seq_len_delta': args.child_max_seq_len_delta,
    'cuda': args.cuda,
    'single_gpu': args.single_gpu,
    'gpu': args.gpu,
  }
  if args.child_arch_pool is not None:
    with open(args.child_arch_pool) as f:
      archs = f.read().splitlines()
      archs = list(map(build_arch, archs))
      params['arch_pool'] = archs
  if os.path.exists(os.path.join(params['model_dir'], 'arch_pool')):
    logging.info('Found arch_pool in child model dir, loading')
    with open(os.path.join(params['model_dir'], 'arch_pool')) as f:
      archs = f.read().splitlines()
      archs = list(map(build_arch, archs))
      params['arch_pool'] = archs
  return params

def get_controller_params():
  params = {
    'model_dir': os.path.join(args.model_dir, 'controller'),
    'shuffle': args.controller_shuffle,
    'num_seed_arch': args.controller_num_seed_arch,
    'encoder_num_layers': args.controller_encoder_num_layers,
    'encoder_hidden_size': args.controller_encoder_hidden_size,
    'encoder_emb_size': args.controller_encoder_emb_size,
    'mlp_num_layers': args.controller_mlp_num_layers,
    'mlp_hidden_size': args.controller_mlp_hidden_size,
    'decoder_num_layers': args.controller_decoder_num_layers,
    'decoder_hidden_size': args.controller_decoder_hidden_size,
    'source_length': args.controller_source_length,
    'encoder_length': args.controller_encoder_length,
    'decoder_length': args.controller_decoder_length,
    'encoder_dropout': args.controller_encoder_dropout,
    'mlp_dropout': args.controller_mlp_dropout,
    'decoder_dropout': args.controller_decoder_dropout,
    'weight_decay': args.controller_weight_decay,
    'encoder_vocab_size': args.controller_encoder_vocab_size,
    'decoder_vocab_size': args.controller_decoder_vocab_size,
    'trade_off': args.controller_trade_off,
    'train_epochs': args.controller_train_epochs,
    'save_frequency': args.controller_save_frequency,
    'batch_size': args.controller_batch_size,
    'lr': args.controller_lr,
    'optimizer': args.controller_optimizer,
    'max_gradient_norm': args.controller_max_gradient_norm,
    'predict_beam_width': args.controller_predict_beam_width,
    'predict_lambda': args.controller_predict_lambda,
    'max_step_size': args.controller_max_step_size,
    'max_new_archs': args.controller_max_new_archs,
  }
  return params

def main():
  all_params = vars(args)
  with open(os.path.join(args.model_dir, 'hparams.json'), 'w') as f:
    json.dump(all_params, f)
  logging.info('LoRA rank: %d, LoRA LR multiplier: %.1f', args.lora_rank, args.lora_lr_multiplier)
  logging.info('CVaR RL: %s, alpha=%.2f, PPO lr=%.4f', args.use_cvar_rl, args.cvar_alpha, args.ppo_lr)
  train()

if __name__ == '__main__':
  main()
