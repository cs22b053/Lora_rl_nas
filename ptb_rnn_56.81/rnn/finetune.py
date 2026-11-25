import os
import gc
import sys
import glob
import time
import math
import numpy as np
import torch
import torch.nn as nn
import logging
import argparse
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import data
import model

from torch.autograd import Variable
from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint, parse_arch

parser = argparse.ArgumentParser(description='Finetune a trained PTB RNN model')
parser.add_argument('--data', type=str, default='../data/penn/',
                    help='location of the data corpus')
parser.add_argument('--lr', type=float, default=25,
                    help='initial learning rate for finetuning')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=4000,
                    help='upper epoch limit for finetuning')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=2345,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='non-monotone interval for triggering ASGD')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='models/finetune',
                    help='path to save the finetuned model')
parser.add_argument('--model_path', type=str, required=True,
                    help='path to the pretrained model directory (e.g. models/train_best_lora_cvar_1)')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization')
parser.add_argument('--wdecay', type=float, default=8e-7,
                    help='weight decay')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='small batch size for gradient accumulation')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length delta')
parser.add_argument('--single_gpu', default=True, action='store_false',
                    help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
args = parser.parse_args()

if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

os.makedirs(args.save, exist_ok=True)
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
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
        cudnn.enabled = True
        torch.cuda.manual_seed_all(args.seed)

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args.cuda)
val_data = batchify(corpus.valid, eval_batch_size, args.cuda)
test_data = batchify(corpus.test, test_batch_size, args.cuda)

ntokens = len(corpus.dictionary)

# Load the pretrained model
logging.info('Loading pretrained model from {}'.format(args.model_path))
model = torch.load(os.path.join(args.model_path, 'model.pt'))
logging.info('Model total parameters: {}'.format(sum(x.data.nelement() for x in model.parameters())))

if args.cuda:
    if args.single_gpu:
        parallel_model = model.cuda()
    else:
        parallel_model = nn.DataParallel(model, dim=1).cuda()
else:
    parallel_model = model


def evaluate(data_source, batch_size=10):
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args.bptt, evaluation=True)
            targets = targets.view(-1)

            log_prob, hidden = parallel_model(data, hidden)
            loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

            total_loss += loss * len(data)

            hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train():
    assert args.batch_size % args.small_batch_size == 0

    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args.bptt, seq_len=seq_len)

        optimizer.zero_grad()

        start, end, s_id = 0, args.small_batch_size, 0
        while start < args.batch_size:
            cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)

            hidden[s_id] = repackage_hidden(hidden[s_id])

            log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = parallel_model(cur_data, hidden[s_id], return_h=True)
            raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

            loss = raw_loss
            if args.alpha > 0:
                loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss *= args.small_batch_size / args.batch_size
            total_loss += raw_loss.data * args.small_batch_size / args.batch_size
            loss.backward()

            s_id += 1
            start = end
            end = start + args.small_batch_size

            gc.collect()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        optimizer.param_groups[0]['lr'] = lr2

        if np.isnan(total_loss.item()):
            raise OverflowError('NAN loss')

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        batch += 1
        i += seq_len


# Evaluate the loaded model first
val_loss_init = evaluate(val_data)
test_loss_init = evaluate(test_data, test_batch_size)
logging.info('Pretrained model | valid ppl {:8.2f} | test ppl {:8.2f}'.format(
    math.exp(val_loss_init), math.exp(test_loss_init)))

# Start finetuning with SGD (fresh optimizer, new random seed changes data order)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

best_val_loss = []
stored_loss = val_loss_init  # Start from the pretrained model's validation loss

# Save the initial pretrained model as our starting best so rollback works
torch.save(model, os.path.join(args.save, 'model.pt'))
torch.save(optimizer.state_dict(), os.path.join(args.save, 'optimizer.pt'))
logging.info('Saved initial pretrained model to {} for rollback safety'.format(args.save))

try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        try:
            train()
        except Exception as e:
            logging.info('Exception during training: {}'.format(e))
            logging.info('rolling back to the previous best model ...')
            try:
                model = torch.load(os.path.join(args.save, 'model.pt'))
                parallel_model = model.cuda()
                
                optimizer_state = torch.load(os.path.join(args.save, 'optimizer.pt'))
                if 't0' in optimizer_state['param_groups'][0]:
                    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
                optimizer.load_state_dict(optimizer_state)
            except Exception as e2:
                logging.info('Rollback failed: {}, reloading from pretrained'.format(e2))
                model = torch.load(os.path.join(args.model_path, 'model.pt'))
                parallel_model = model.cuda()
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
            continue

        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data)
            logging.info('-' * 89)
            logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss2, math.exp(val_loss2)))
            logging.info('-' * 89)

            if val_loss2 < stored_loss:
                stored_loss = val_loss2
                save_checkpoint(model, optimizer, epoch, args.save, finetune=True,
                               stored_loss=stored_loss, best_val_loss=best_val_loss)
                # Also save as model.pt for compatibility
                torch.save(model, os.path.join(args.save, 'model.pt'))
                logging.info('Saving Averaged!')

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(val_data, eval_batch_size)
            logging.info('-' * 89)
            logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            logging.info('-' * 89)

            if val_loss < stored_loss:
                stored_loss = val_loss
                save_checkpoint(model, optimizer, epoch, args.save, stored_loss=stored_loss, best_val_loss=best_val_loss)
                logging.info('Saving Normal!')

            if 't0' not in optimizer.param_groups[0] and (len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                logging.info('Switching to ASGD!')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    logging.info('-' * 89)
    logging.info('Exiting from training early')

# Load the best saved model
if os.path.exists(os.path.join(args.save, 'model.pt')):
    model = torch.load(os.path.join(args.save, 'model.pt'))
else:
    model = torch.load(os.path.join(args.model_path, 'model.pt'))
parallel_model = model.cuda()

# Run on test data
test_loss = evaluate(test_data, test_batch_size)
logging.info('=' * 89)
logging.info('| End of finetuning | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logging.info('=' * 89)

