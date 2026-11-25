import os
import sys
import glob
import time
import copy
import random
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from model import NASNetworkCIFAR, NASNetworkImageNet
from model_search import NASWSNetworkCIFAR, NASWSNetworkImageNet
from operations import OPERATIONS_search_small, OPERATIONS_search_middle
from controller import NAO
from ArchitecturePolicy import ArchitecturePolicy
from PPOArchTrainer import PPOArchTrainer

parser = argparse.ArgumentParser(description='NAO Search')

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--zip_file', action='store_true', default=False)
parser.add_argument('--lazy_load', action='store_true', default=False)
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--search_space', type=str, default='middle', choices=['small', 'middle'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--child_batch_size', type=int, default=64)
parser.add_argument('--child_eval_batch_size', type=int, default=500)
parser.add_argument('--child_epochs', type=int, default=50)
parser.add_argument('--child_layers', type=int, default=3)
parser.add_argument('--child_nodes', type=int, default=5)
parser.add_argument('--child_channels', type=int, default=20)
parser.add_argument('--child_cutout_size', type=int, default=None)
parser.add_argument('--child_grad_bound', type=float, default=5.0)
parser.add_argument('--child_lr_max', type=float, default=0.025)
parser.add_argument('--child_lr_min', type=float, default=0.001)
parser.add_argument('--child_keep_prob', type=float, default=1.0)
parser.add_argument('--child_drop_path_keep_prob', type=float, default=0.9)
parser.add_argument('--child_l2_reg', type=float, default=3e-4)
parser.add_argument('--child_use_aux_head', action='store_true', default=False)
parser.add_argument('--child_arch_pool', type=str, default=None)
parser.add_argument('--child_lr', type=float, default=0.1)
parser.add_argument('--child_label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--child_gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--child_decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--controller_seed_arch', type=int, default=1000)
parser.add_argument('--controller_expand', type=int, default=10)
parser.add_argument('--controller_new_arch', type=int, default=300)
parser.add_argument('--controller_encoder_layers', type=int, default=1)
parser.add_argument('--controller_encoder_hidden_size', type=int, default=64)
parser.add_argument('--controller_encoder_emb_size', type=int, default=32)
parser.add_argument('--controller_mlp_layers', type=int, default=0)
parser.add_argument('--controller_mlp_hidden_size', type=int, default=200)
parser.add_argument('--controller_decoder_layers', type=int, default=1)
parser.add_argument('--controller_decoder_hidden_size', type=int, default=64)
parser.add_argument('--controller_source_length', type=int, default=40)
parser.add_argument('--controller_encoder_length', type=int, default=20)
parser.add_argument('--controller_decoder_length', type=int, default=40)
parser.add_argument('--controller_encoder_dropout', type=float, default=0)
parser.add_argument('--controller_mlp_dropout', type=float, default=0.1)
parser.add_argument('--controller_decoder_dropout', type=float, default=0)
parser.add_argument('--controller_l2_reg', type=float, default=0)
parser.add_argument('--controller_encoder_vocab_size', type=int, default=12)
parser.add_argument('--controller_decoder_vocab_size', type=int, default=12)
parser.add_argument('--controller_trade_off', type=float, default=0.8)
parser.add_argument('--controller_epochs', type=int, default=1000)
parser.add_argument('--controller_batch_size', type=int, default=100)
parser.add_argument('--controller_lr', type=float, default=0.001)
parser.add_argument('--controller_optimizer', type=str, default='adam')
parser.add_argument('--controller_grad_bound', type=float, default=5.0)
parser.add_argument('--lora_rank', type=int, default=16, help='Rank of LoRA adapters in WS operations')
parser.add_argument('--lora_lr_multiplier', type=float, default=10.0, help='LR multiplier for LoRA adapter parameters')
args = parser.parse_args()

utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def get_builder(dataset):
    if dataset == 'cifar10':
        return build_cifar10
    elif dataset == 'cifar100':
        return build_cifar100
    else:
        return build_imagenet
    

def build_cifar10(model_state_dict=None, optimizer_state_dict=None, **kwargs):
    epoch = kwargs.pop('epoch')
    ratio = kwargs.pop('ratio')
    train_transform, valid_transform = utils._data_transforms_cifar10(args.child_cutout_size)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)

    num_train = len(train_data)
    assert num_train == len(valid_data)
    indices = list(range(num_train))    
    split = int(np.floor(ratio * num_train))
    np.random.shuffle(indices)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.child_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.child_eval_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=16)
    
    model = NASWSNetworkCIFAR(args, 10, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob, args.child_drop_path_keep_prob,
                       args.child_use_aux_head, args.steps, lora_rank=args.lora_rank)
    model = model.cuda()
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # Separate param groups: base params vs LoRA params (higher LR)
    lora_params = []
    base_params = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
        else:
            base_params.append(param)
    optimizer = torch.optim.SGD(
        [
            {'params': base_params},
            {'params': lora_params, 'lr': args.child_lr_max * args.lora_lr_multiplier},
        ],
        args.child_lr_max,
        momentum=0.9,
        weight_decay=args.child_l2_reg,
    )
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.child_epochs, args.child_lr_min, epoch)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def build_cifar100(model_state_dict=None, optimizer_state_dict=None, **kwargs):
    epoch = kwargs.pop('epoch')
    ratio = kwargs.pop('ratio')
    train_transform, valid_transform = utils._data_transforms_cifar10(args.cutout_size)
    train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=valid_transform)

    num_train = len(train_data)
    assert num_train == len(valid_data)
    indices = list(range(num_train))    
    split = int(np.floor(ratio * num_train))
    np.random.shuffle(indices)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.child_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.child_eval_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=16)
    
    model = NASWSNetworkCIFAR(args, 100, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob, args.child_drop_path_keep_prob,
                       args.child_use_aux_head, args.steps, lora_rank=args.lora_rank)
    model = model.cuda()
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # Separate param groups: base params vs LoRA params (higher LR)
    lora_params = []
    base_params = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
        else:
            base_params.append(param)
    optimizer = torch.optim.SGD(
        [
            {'params': base_params},
            {'params': lora_params, 'lr': args.child_lr_max * args.lora_lr_multiplier},
        ],
        args.child_lr_max,
        momentum=0.9,
        weight_decay=args.child_l2_reg,
    )
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.child_epochs, args.child_lr_min, epoch)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def build_imagenet(model_state_dict, optimizer_state_dict, **kwargs):
    ratio = kwargs.pop('ratio')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if args.zip_file:
        logging.info('Loading data from zip file')
        traindir = os.path.join(args.data, 'train.zip')
        if args.lazy_load:
            train_data = utils.ZipDataset(traindir, train_transform)
        else:
            logging.info('Loading data into memory')
            train_data = utils.InMemoryZipDataset(traindir, train_transform, num_workers=32)
    else:
        logging.info('Loading data from directory')
        traindir = os.path.join(args.data, 'train')
        if args.lazy_load:
            train_data = dset.ImageFolder(traindir, train_transform)
        else:
            logging.info('Loading data into memory')
            train_data = utils.InMemoryDataset(traindir, train_transform, num_workers=32)
       
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(ratio * num_train))
    train_indices = sorted(indices[:split])
    valid_indices = sorted(indices[split:])

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.child_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
        pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.child_eval_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_indices),
        pin_memory=True, num_workers=16)
    
    model = NASWSNetworkImageNet(args, 1000, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob,
                       args.child_drop_path_keep_prob, args.child_use_aux_head, args.steps, lora_rank=args.lora_rank)
    model = model.cuda()
    train_criterion = CrossEntropyLabelSmooth(1000, args.child_label_smooth).cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # Separate param groups: base params vs LoRA params (higher LR)
    lora_params = []
    base_params = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
        else:
            base_params.append(param)
    optimizer = torch.optim.SGD(
        [
            {'params': base_params},
            {'params': lora_params, 'lr': args.child_lr * args.lora_lr_multiplier},
        ],
        args.child_lr,
        momentum=0.9,
        weight_decay=args.child_l2_reg,
    )
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.child_decay_period, gamma=args.child_gamma)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def get_scheduler(optimizer, dataset):
    if 'cifar' in dataset:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.child_epochs, args.child_lr_min)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.child_decay_period, gamma=args.child_gamma)
    return scheduler


def child_train(train_queue, model, optimizer, global_step, arch_pool, arch_pool_prob, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = input.cuda().requires_grad_()
        target = target.cuda()

        optimizer.zero_grad()
        # sample an arch to train
        arch = utils.sample_arch(arch_pool, arch_pool_prob)
        logits, aux_logits = model(input, arch, global_step)
        global_step += 1
        loss = criterion(logits, target)
        if aux_logits is not None:
            aux_loss = criterion(aux_logits, target)
            loss += 0.4 * aux_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.child_grad_bound)
        optimizer.step()
        
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)
        
        if (step+1) % 100 == 0:
            logging.info('Train %03d loss %e top1 %f top5 %f', step+1, objs.avg, top1.avg, top5.avg)
            logging.info('Arch: %s', ' '.join(map(str, arch[0] + arch[1])))

    return top1.avg, objs.avg, global_step


def child_valid(valid_queue, model, arch_pool, criterion, num_eval_batches=10):
    valid_acc_list = []
    with torch.no_grad():
        model.eval()
        for i, arch in enumerate(arch_pool):
            # Multi-batch evaluation for robust ranking
            top1_meter = utils.AvgrageMeter()
            valid_iter = iter(valid_queue)
            for _ in range(num_eval_batches):
                try:
                    inputs, targets = next(valid_iter)
                except StopIteration:
                    valid_iter = iter(valid_queue)
                    inputs, targets = next(valid_iter)
                inputs = inputs.cuda()
                targets = targets.cuda()
                logits, _ = model(inputs, arch, bn_train=True)
                prec1, _ = utils.accuracy(logits, targets, topk=(1, 5))
                top1_meter.update(prec1.data, inputs.size(0))
            valid_acc_list.append(top1_meter.avg.cpu().item()/100)
            
            if (i+1) % 100 == 0:
                logging.info('Valid arch %s top1 %f', ' '.join(map(str, arch[0] + arch[1])), top1_meter.avg)
        
    return valid_acc_list


def train_and_evaluate_top_on_cifar10(archs, train_queue, valid_queue):
    res = []
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    for i, arch in enumerate(archs):
        objs.reset()
        top1.reset()
        top5.reset()
        logging.info('Train and evaluate the {} arch'.format(i+1))
        model = NASNetworkCIFAR(args, 10, args.child_layers, args.child_nodes, args.child_channels, 0.6, 0.8,
                        True, args.steps, arch)
        model = model.cuda()
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.child_lr_max,
            momentum=0.9,
            weight_decay=args.child_l2_reg,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, args.child_lr_min)
        global_step = 0
        for e in range(50):
            scheduler.step()
            for step, (input, target) in enumerate(train_queue):
                input = input.cuda().requires_grad_()
                target = target.cuda()

                optimizer.zero_grad()
                # sample an arch to train
                logits, aux_logits = model(input, global_step)
                global_step += 1
                loss = train_criterion(logits, target)
                if aux_logits is not None:
                    aux_loss = train_criterion(aux_logits, target)
                    loss += 0.4 * aux_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.child_grad_bound)
                optimizer.step()
            
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)
            
                if (step+1) % 100 == 0:
                    logging.info('Train epoch %03d %03d loss %e top1 %f top5 %f', e+1, step+1, objs.avg, top1.avg, top5.avg)
        objs.reset()
        top1.reset()
        top5.reset()
        with torch.no_grad():
            model.eval()
            for step, (input, target) in enumerate(valid_queue):
                input = input.cuda()
                target = target.cuda()
            
                logits, _ = model(input)
                loss = eval_criterion(logits, target)
            
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)
            
                if (step+1) % 100 == 0:
                    logging.info('valid %03d %e %f %f', step+1, objs.avg, top1.avg, top5.avg)
        res.append(top1.avg.cpu().item())
    return res


def train_and_evaluate_top_on_cifar100(archs, train_queue, valid_queue):
    res = []
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    for i, arch in enumerate(archs):
        objs.reset()
        top1.reset()
        top5.reset()
        logging.info('Train and evaluate the {} arch'.format(i+1))
        model = NASNetworkCIFAR(args, 100, args.child_layers, args.child_nodes, args.child_channels, 0.6, 0.8,
                        True, args.steps, arch)
        model = model.cuda()
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.child_lr_max,
            momentum=0.9,
            weight_decay=args.child_l2_reg,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, args.child_lr_min)
        global_step = 0
        for e in range(50):
            scheduler.step()
            for step, (input, target) in enumerate(train_queue):
                input = input.cuda().requires_grad_()
                target = target.cuda()

                optimizer.zero_grad()
                # sample an arch to train
                logits, aux_logits = model(input, global_step)
                global_step += 1
                loss = train_criterion(logits, target)
                if aux_logits is not None:
                    aux_loss = train_criterion(aux_logits, target)
                    loss += 0.4 * aux_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.child_grad_bound)
                optimizer.step()
            
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)
            
                if (step+1) % 100 == 0:
                    logging.info('Train %3d %03d loss %e top1 %f top5 %f', e+1, step+1, objs.avg, top1.avg, top5.avg)
        objs.reset()
        top1.reset()
        top5.reset()
        with torch.no_grad():
            model.eval()
            for step, (input, target) in enumerate(valid_queue):
                input = input.cuda()
                target = target.cuda()
            
                logits, _ = model(input)
                loss = eval_criterion(logits, target)
            
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)
            
                if (step+1) % 100 == 0:
                    logging.info('valid %03d %e %f %f', step+1, objs.avg, top1.avg, top5.avg)
        res.append(top1.avg.cpu().item())
    return res


def train_and_evaluate_top_on_imagenet(archs, train_queue, valid_queue):
    res = []
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    for i, arch in enumerate(archs):
        objs.reset()
        top1.reset()
        top5.reset()
        logging.info('Train and evaluate the {} arch'.format(i+1))
        model = NASNetworkImageNet(args, 1000, args.child_layers, args.child_nodes, args.child_channels, 1.0, 1.0,
                        True, args.steps, arch)
        model = model.cuda()
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.child_lr,
            momentum=0.9,
            weight_decay=args.child_l2_reg,
        )
        for step, (input, target) in enumerate(train_queue):
            input = input.cuda().requires_grad_()
            target = target.cuda()

            optimizer.zero_grad()
            # sample an arch to train
            logits, aux_logits = model(input, step)
            loss = train_criterion(logits, target)
            if aux_logits is not None:
                aux_loss = train_criterion(aux_logits, target)
                loss += 0.4 * aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.child_grad_bound)
            optimizer.step()
            
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)
            
            if (step+1) % 100 == 0:
                logging.info('Train %03d loss %e top1 %f top5 %f', step+1, objs.avg, top1.avg, top5.avg)
            if step+1 == 500:
                break

        objs.reset()
        top1.reset()
        top5.reset()
        with torch.no_grad():
            model.eval()
            for step, (input, target) in enumerate(valid_queue):
                input = input.cuda()
                target = target.cuda()
            
                logits, _ = model(input)
                loss = eval_criterion(logits, target)
            
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)
            
                if (step+1) % 100 == 0:
                    logging.info('valid %03d %e %f %f', step+1, objs.avg, top1.avg, top5.avg)
        res.append(top1.avg.cpu().item())
    return res


def nao_train(train_queue, model, optimizer):
    objs = utils.AvgrageMeter()
    mse = utils.AvgrageMeter()
    nll = utils.AvgrageMeter()
    model.train()
    for step, sample in enumerate(train_queue):
        encoder_input = sample['encoder_input']
        encoder_target = sample['encoder_target']
        decoder_input = sample['decoder_input']
        decoder_target = sample['decoder_target']
        
        encoder_input = encoder_input.cuda()
        encoder_target = encoder_target.cuda().requires_grad_()
        decoder_input = decoder_input.cuda()
        decoder_target = decoder_target.cuda()
        
        optimizer.zero_grad()
        predict_value, log_prob, arch = model(encoder_input, decoder_input)
        loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze())
        loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1))
        loss = args.controller_trade_off * loss_1 + (1 - args.controller_trade_off) * loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.controller_grad_bound)
        optimizer.step()
        
        n = encoder_input.size(0)
        objs.update(loss.data, n)
        mse.update(loss_1.data, n)
        nll.update(loss_2.data, n)
        
    return objs.avg, mse.avg, nll.avg


def nao_valid(queue, model):
    inputs = []
    targets = []
    predictions = []
    archs = []
    with torch.no_grad():
        model.eval()
        for step, sample in enumerate(queue):
            encoder_input = sample['encoder_input']
            encoder_target = sample['encoder_target']
            decoder_target = sample['decoder_target']
            
            encoder_input = encoder_input.cuda()
            encoder_target = encoder_target.cuda()
            decoder_target = decoder_target.cuda()
            
            predict_value, logits, arch = model(encoder_input)
            n = encoder_input.size(0)
            
            # FIX: Use .view(-1) instead of .squeeze() to prevent scalar collapse
            # when batch_size=1, squeeze() reduces [1,40] -> scalar, view(-1) keeps it as [40]
            inputs      += encoder_input.data.view(n, -1).tolist()
            targets     += encoder_target.data.view(-1).tolist()
            predictions += predict_value.data.view(-1).tolist()
            archs       += arch.data.squeeze(-1).view(n, -1).tolist()

    pa = utils.pairwise_accuracy(targets, predictions)
    hd = utils.hamming_distance(inputs, archs)
    return pa, hd


def nao_infer(queue, model, step, direction='+'):
    new_arch_list = []
    model.eval()
    for i, sample in enumerate(queue):
        encoder_input = sample['encoder_input']
        encoder_input = encoder_input.cuda()
        model.zero_grad()
        new_arch = model.generate_new_arch(encoder_input, step, direction=direction)
        new_arch_list.extend(new_arch.data.squeeze().tolist())
    return new_arch_list


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    if args.dataset == 'cifar10':
        args.num_class = 10
    elif args.dataset == 'cifar100':
        args.num_class = 100
    else:
        args.num_class = 10
    
    if args.search_space == 'small':
        OPERATIONS = OPERATIONS_search_small
    elif args.search_space == 'middle':
        OPERATIONS = OPERATIONS_search_middle
    args.child_num_ops = len(OPERATIONS)
    args.controller_op_offset = utils.get_op_token_offset(args.child_nodes)
    args.controller_encoder_vocab_size = args.controller_op_offset + args.child_num_ops
    logging.info(
        f'Encoder vocab size: {args.controller_encoder_vocab_size} '
        f'(nodes: {args.child_nodes}, ops: {args.child_num_ops}, op_offset: {args.controller_op_offset})'
    )
    args.controller_decoder_vocab_size = args.controller_encoder_vocab_size
    args.steps = int(np.ceil(45000 / args.child_batch_size)) * args.child_epochs
    args.controller_encoder_length = 4 * args.child_nodes  # One cell: 4 elements per node
    args.controller_decoder_length = 4 * args.child_nodes * 2  # Two cells (conv + reduc)
    args.controller_source_length = args.controller_decoder_length  # Must match for encoder input

    logging.info(f'Controller lengths: encoder={args.controller_encoder_length}, decoder={args.controller_decoder_length}, source={args.controller_source_length}')

    # Verify
    expected_seq_len = 4 * args.child_nodes * 2
    logging.info(f'Expected sequence length for {args.child_nodes} nodes: {expected_seq_len}')

    logging.info("args = %s", args)
    
    if args.child_arch_pool is not None:
        logging.info('Architecture pool is provided, loading')
        with open(args.child_arch_pool) as f:
            archs = f.read().splitlines()
            archs = list(map(utils.build_dag, archs))
            child_arch_pool = archs
    elif os.path.exists(os.path.join(args.output_dir, 'arch_pool')):
        logging.info('Architecture pool is founded, loading')
        with open(os.path.join(args.output_dir, 'arch_pool')) as f:
            archs = f.read().splitlines()
            archs = list(map(utils.build_dag, archs))
            child_arch_pool = archs
    else:
        child_arch_pool = None

    build_fn = get_builder(args.dataset)
    train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = build_fn(ratio=0.9, epoch=-1)

    nao = NAO(
        args.controller_encoder_layers,
        args.controller_encoder_vocab_size,
        args.controller_encoder_hidden_size,
        args.controller_encoder_dropout,
        args.controller_encoder_length,
        args.controller_source_length,
        args.controller_encoder_emb_size,
        args.controller_mlp_layers,
        args.controller_mlp_hidden_size,
        args.controller_mlp_dropout,
        args.controller_decoder_layers,
        args.controller_decoder_vocab_size,
        args.controller_decoder_hidden_size,
        args.controller_decoder_dropout,
        args.controller_decoder_length,
        num_nodes=args.child_nodes,
    )
    nao = nao.cuda()
    logging.info("Encoder-Predictor-Decoder param size = %fMB", utils.count_parameters_in_MB(nao))


    # ===========================================
    # NEW: Initialize CVaR-based step size policy
    # ===========================================
    ppo_policy = ArchitecturePolicy(
        embedding_dim=args.controller_encoder_hidden_size,  # 64
        hidden_dim=256,
        num_nodes=args.child_nodes,
        num_ops=args.child_num_ops,
    ).cuda()

    ppo_trainer = PPOArchTrainer(
        nao_model=nao,
        policy=ppo_policy,
        embedding_dim=args.controller_encoder_hidden_size,
        lr=3e-4,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,       # Encourages exploration of arch space
        n_epochs=10,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        max_step_magnitude=3.0,  # Replaces predict_lambda
        cvar_alpha=0.2,          # CVaR risk level
    )
    logging.info("PPO Architecture Policy param size = %fMB", utils.count_parameters_in_MB(ppo_policy))
    
    if child_arch_pool is None:
        logging.info('Architecture pool is not provided, randomly generating now')
        child_arch_pool = utils.generate_arch(args.controller_seed_arch, args.child_nodes, args.child_num_ops)
        
        # ========================================
        # ADD VALIDATION HERE
        # ========================================
        logging.info(f'Validating {len(child_arch_pool)} generated architectures...')
        for idx, arch in enumerate(child_arch_pool):
            conv_arch, reduc_arch = arch[0], arch[1]
            
            # Check length
            expected_len = 4 * args.child_nodes
            if len(conv_arch) != expected_len:
                raise ValueError(f"Conv arch {idx}: expected length {expected_len}, got {len(conv_arch)}")
            if len(reduc_arch) != expected_len:
                raise ValueError(f"Reduc arch {idx}: expected length {expected_len}, got {len(reduc_arch)}")
            
            # Check indices for each node
            for i in range(args.child_nodes):
                x_id_conv = conv_arch[4*i]
                y_id_conv = conv_arch[4*i+2]
                x_id_reduc = reduc_arch[4*i]
                y_id_reduc = reduc_arch[4*i+2]
                
                max_id = i + 1
                
                if not (0 <= x_id_conv <= max_id):
                    raise ValueError(f"Arch {idx}, conv node {i}: x_id={x_id_conv} > max={max_id}")
                if not (0 <= y_id_conv <= max_id):
                    raise ValueError(f"Arch {idx}, conv node {i}: y_id={y_id_conv} > max={max_id}")
                if not (0 <= x_id_reduc <= max_id):
                    raise ValueError(f"Arch {idx}, reduc node {i}: x_id={x_id_reduc} > max={max_id}")
                if not (0 <= y_id_reduc <= max_id):
                    raise ValueError(f"Arch {idx}, reduc node {i}: y_id={y_id_reduc} > max={max_id}")
        
        logging.info('\u2713 All generated architectures valid!')
        # ========================================
    arch_pool = []
    arch_pool_valid_acc = []
    num_search_iterations = 4  # Was 2 — PPO+CVaR needs at least 3 iterations to actually run
    for i in range(num_search_iterations):
        logging.info('Iteration %d', i)

        child_arch_pool_prob = []
        for arch in child_arch_pool:
            if args.dataset == 'cifar10':
                tmp_model = NASNetworkCIFAR(args, args.num_class, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob, args.child_drop_path_keep_prob,
                    args.child_use_aux_head, args.steps, arch)
            elif args.dataset == 'cifar100':
                tmp_model = NASNetworkCIFAR(args, args.num_class, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob, args.child_drop_path_keep_prob,
                    args.child_use_aux_head, args.steps, arch)
            else:
                tmp_model = NASNetworkImageNet(args, args.num_class, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob,
                    args.child_drop_path_keep_prob, args.child_use_aux_head, args.steps, arch)
            child_arch_pool_prob.append(utils.count_parameters_in_MB(tmp_model))
            del tmp_model
        
        step = 0
        scheduler = get_scheduler(optimizer, args.dataset)
        for epoch in range(1, args.child_epochs + 1):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)
            # sample an arch to train
            train_acc, train_obj, step = child_train(train_queue, model, optimizer, step, child_arch_pool, child_arch_pool_prob, train_criterion)
            logging.info('train_acc %f', train_acc)

        logging.info("Evaluate seed archs")
        arch_pool += child_arch_pool
        arch_pool_valid_acc = child_valid(valid_queue, model, arch_pool, eval_criterion)

        arch_pool_valid_acc_sorted_indices = np.argsort(arch_pool_valid_acc)[::-1]
        arch_pool = [arch_pool[i] for i in arch_pool_valid_acc_sorted_indices]
        arch_pool_valid_acc = [arch_pool_valid_acc[i] for i in arch_pool_valid_acc_sorted_indices]
        with open(os.path.join(args.output_dir, 'arch_pool.{}'.format(i)), 'w') as fa:
            with open(os.path.join(args.output_dir, 'arch_pool.perf.{}'.format(i)), 'w') as fp:
                for arch, perf in zip(arch_pool, arch_pool_valid_acc):
                    arch = ' '.join(map(str, arch[0] + arch[1]))
                    fa.write('{}\n'.format(arch))
                    fp.write('{}\n'.format(perf))
        if i == num_search_iterations - 1:
            break
                            
        # Train Encoder-Predictor-Decoder
        logging.info('Train Encoder-Predictor-Decoder')
        encoder_input = list(map(lambda x: utils.parse_arch_to_seq(x[0], nodes=args.child_nodes) + utils.parse_arch_to_seq(x[1], nodes=args.child_nodes), arch_pool))
        # [[conv, reduc]]
        min_val = min(arch_pool_valid_acc)
        max_val = max(arch_pool_valid_acc)
        encoder_target = [(i - min_val) / (max_val - min_val) for i in arch_pool_valid_acc]

        if args.controller_expand:
            dataset = list(zip(encoder_input, encoder_target))
            n = len(dataset)
            ratio = 0.9
            split = int(n*ratio)
            np.random.shuffle(dataset)
            encoder_input, encoder_target = list(zip(*dataset))
            train_encoder_input = list(encoder_input[:split])
            train_encoder_target = list(encoder_target[:split])
            valid_encoder_input = list(encoder_input[split:])
            valid_encoder_target = list(encoder_target[split:])
            for _ in range(args.controller_expand-1):
                for src, tgt in zip(encoder_input[:split], encoder_target[:split]):
                    a = np.random.randint(0, args.child_nodes)
                    b = np.random.randint(0, args.child_nodes)
                    cell_len = 4 * args.child_nodes
                    src = src[:4 * a] + src[4 * a + 2:4 * a + 4] + \
                            src[4 * a:4 * a + 2] + src[4 * (a + 1):cell_len + 4 * b] + \
                            src[cell_len + 4 * b + 2:cell_len + 4 * b + 4] + src[cell_len + 4 * b:cell_len + 4 * b + 2] + \
                            src[cell_len + 4 * (b + 1):]
                    train_encoder_input.append(src)
                    train_encoder_target.append(tgt)
        else:
            train_encoder_input = encoder_input
            train_encoder_target = encoder_target
            valid_encoder_input = encoder_input
            valid_encoder_target = encoder_target
        logging.info('Train data: {}\tValid data: {}'.format(len(train_encoder_input), len(valid_encoder_input)))

        # Validate encoder input sequences
        logging.info('Validating encoder input sequences...')
        max_token_id = args.controller_encoder_vocab_size - 1
        valid_count = 0
        invalid_count = 0

        for idx, seq in enumerate(train_encoder_input):
            if any(token > max_token_id or token < 0 for token in seq):
                invalid_tokens = [t for t in seq if t > max_token_id or t < 0]
                logging.warning(f'Sequence {idx} has invalid tokens: {invalid_tokens} (max allowed: {max_token_id})')
                train_encoder_input[idx] = [min(max(0, t), max_token_id) for t in seq]
                invalid_count += 1
            else:
                valid_count += 1

        for idx, seq in enumerate(valid_encoder_input):
            if any(token > max_token_id or token < 0 for token in seq):
                invalid_tokens = [t for t in seq if t > max_token_id or t < 0]
                logging.warning(f'Valid sequence {idx} has invalid tokens: {invalid_tokens}')
                valid_encoder_input[idx] = [min(max(0, t), max_token_id) for t in seq]

        logging.info(f'Validation complete: {valid_count} valid, {invalid_count} invalid (fixed)')

        nao_train_dataset = utils.NAODataset(train_encoder_input, train_encoder_target, True, swap=True if args.controller_expand is None else False, nodes=args.child_nodes)
        nao_valid_dataset = utils.NAODataset(valid_encoder_input, valid_encoder_target, False, nodes=args.child_nodes)
        nao_train_queue = torch.utils.data.DataLoader(
            nao_train_dataset, batch_size=args.controller_batch_size, shuffle=True, pin_memory=True)
        nao_valid_queue = torch.utils.data.DataLoader(
            nao_valid_dataset, batch_size=args.controller_batch_size, shuffle=False, pin_memory=True, drop_last=True)
        nao_optimizer = torch.optim.Adam(nao.parameters(), lr=args.controller_lr, weight_decay=args.controller_l2_reg)
        for nao_epoch in range(1, args.controller_epochs+1):
            nao_loss, nao_mse, nao_ce = nao_train(nao_train_queue, nao, nao_optimizer)
            logging.info("epoch %04d train loss %.6f mse %.6f ce %.6f", nao_epoch, nao_loss, nao_mse, nao_ce)
            if nao_epoch % 100 == 0:
                pa, hs = nao_valid(nao_valid_queue, nao)
                logging.info("Evaluation on valid data")
                logging.info('epoch %04d pairwise accuracy %.6f hamming distance %.6f', nao_epoch, pa, hs)

        # === PPO+CVaR Architecture Generation ===

        # Step 1: PPO rollout + CVaR update (only after iteration 0)
        new_archs = []
        max_step_size = 50
        top100_archs = list(map(
            lambda x: utils.parse_arch_to_seq(x[0], nodes=args.child_nodes) +
                      utils.parse_arch_to_seq(x[1], nodes=args.child_nodes),
            arch_pool[:100]
        ))
        nao_infer_dataset = utils.NAODataset(top100_archs, None, False, nodes=args.child_nodes)
        nao_infer_queue = torch.utils.data.DataLoader(
            nao_infer_dataset,
            batch_size=len(nao_infer_dataset),
            shuffle=False,
            pin_memory=True
        )

        if i == 0:
            # Iteration 0: NAO gradient step only — PPO predictor not trained yet
            # Use adaptive (Fibonacci-like) step sizes for better latent space coverage
            step_sizes = [1, 2, 3, 5, 8, 13, 21, 34, 50]
            for predict_step_size in step_sizes:
                new_arch = nao_infer(nao_infer_queue, nao, predict_step_size, direction='+')
                for arch in new_arch:
                    if arch not in encoder_input and arch not in new_archs:
                        new_archs.append(arch)
                if len(new_archs) >= args.controller_new_arch:
                    break
        else:
            # Iteration 1+: Hybrid approach — 50% NAO gradient + 50% PPO+CVaR
            logging.info('=== Hybrid NAO+PPO Architecture Search (Iteration %d) ===', i)

            # Part A: NAO gradient-based generation (50% of new archs)
            nao_target = args.controller_new_arch // 2
            step_sizes = [1, 2, 3, 5, 8, 13, 21, 34, 50]
            for predict_step_size in step_sizes:
                new_arch = nao_infer(nao_infer_queue, nao, predict_step_size, direction='+')
                for arch in new_arch:
                    if arch not in encoder_input and arch not in new_archs:
                        new_archs.append(arch)
                if len(new_archs) >= nao_target:
                    break
            logging.info('NAO gradient generated %d archs', len(new_archs))

            # Part B: PPO+CVaR generation (remaining 50%)
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
                n_new=args.controller_new_arch - len(new_archs),
                nao_decoder=nao.decoder
            )
            for arch in ppo_archs:
                if arch not in new_archs:
                    new_archs.append(arch)
            logging.info('After PPO: total %d new archs', len(new_archs))

            # Fallback if PPO generated too few
            if len(new_archs) < args.controller_new_arch:
                logging.warning('PPO generated %d/%d, filling with NAO gradient step',
                                len(new_archs), args.controller_new_arch)
                predict_step_size = 0
                while len(new_archs) < args.controller_new_arch:
                    predict_step_size += 1
                    extra = nao_infer(nao_infer_queue, nao, predict_step_size, direction='+')
                    for arch in extra:
                        if arch not in new_archs:
                            new_archs.append(arch)
                    if predict_step_size > max_step_size:
                        break


        # Add noise-perturbed variants of top architectures
        noise_op_start = args.controller_op_offset
        for arch_seq in top100_archs[:20]:
            for _ in range(5):
                noisy = list(arch_seq)
                cell_len = 4 * args.child_nodes
                # Randomly perturb 2 positions
                for _ in range(2):
                    pos = np.random.randint(0, len(noisy))
                    if pos % 2 == 0:  # node index
                        # Determine which node this position belongs to within its cell
                        pos_in_cell = pos % cell_len
                        node_in_cell = pos_in_cell // 4
                        max_token = node_in_cell + 2 + 1  # can reference 0..node_in_cell+1, encoded as 1..node_in_cell+2
                        noisy[pos] = np.random.randint(1, max_token)
                    else:  # operation
                        noisy[pos] = np.random.randint(noise_op_start, noise_op_start + args.child_num_ops)
                if noisy not in new_archs:
                    new_archs.append(noisy)


        # Step 2: Validate generated sequences (length + node index bounds)
        logging.info('Validating %d new architectures...', len(new_archs))
        valid_new_archs = []
        expected_seq_length = 4 * args.child_nodes * 2
        op_start = args.controller_op_offset

        for idx, arch_seq in enumerate(new_archs):
            if not isinstance(arch_seq, list):
                continue
            if len(arch_seq) != expected_seq_length:
                logging.warning('Arch %d wrong length %d != %d, skipping',
                                idx, len(arch_seq), expected_seq_length)
                continue

            # Validate and clamp node indices for both cells (conv + reduc)
            fixed_seq = list(arch_seq)
            was_fixed = False
            cell_len = 4 * args.child_nodes
            for cell_offset in [0, cell_len]:  # conv cell, then reduc cell
                for node_i in range(args.child_nodes):
                    base = cell_offset + 4 * node_i
                    # Node index positions are at base+0 and base+2 (encoded as raw_id+1)
                    max_token = node_i + 2  # node_i can reference 0..node_i+1, encoded as 1..node_i+2
                    for pos in [base, base + 2]:
                        val = fixed_seq[pos]
                        if val < 1 or val > max_token:
                            clamped = max(1, min(val, max_token))
                            logging.warning('Arch %d pos %d: node idx %d out of [1,%d], clamped to %d',
                                            idx, pos, val, max_token, clamped)
                            fixed_seq[pos] = clamped
                            was_fixed = True
                    # Op positions are at base+1 and base+3 (encoded as 7+op_id)
                    for pos in [base + 1, base + 3]:
                        val = fixed_seq[pos]
                        if val < op_start or val >= op_start + args.child_num_ops:
                            clamped = max(op_start, min(val, op_start + args.child_num_ops - 1))
                            fixed_seq[pos] = clamped
                            was_fixed = True

            valid_new_archs.append(fixed_seq)

        if len(valid_new_archs) == 0:
            logging.error('No valid archs! Falling back to random generation.')
            valid_new_archs = [
                utils.parse_arch_to_seq(a[0], nodes=args.child_nodes) +
                utils.parse_arch_to_seq(a[1], nodes=args.child_nodes)
                for a in utils.generate_arch(10, args.child_nodes, args.child_num_ops)
            ]

        # Step 3: Parse to arch pool for next iteration
        child_arch_pool = list(map(
            lambda x: utils.parse_seq_to_arch(x, nodes=args.child_nodes),
            valid_new_archs
        ))
        logging.info("Generated %d new archs for next iteration", len(child_arch_pool))

    logging.info('Finish Searching')
    logging.info('Reranking top 5 architectures')
    # reranking top 5
    top_archs = arch_pool[:5]
    if args.dataset == 'cifar10':
        top_archs_perf = train_and_evaluate_top_on_cifar10(top_archs, train_queue, valid_queue)
    elif args.dataset == 'cifar100':
        top_archs_perf = train_and_evaluate_top_on_cifar100(top_archs, train_queue, valid_queue)
    else:
        top_archs_perf = train_and_evaluate_top_on_imagenet(top_archs, train_queue, valid_queue)
    top_archs_sorted_indices = np.argsort(top_archs_perf)[::-1]
    top_archs = [top_archs[i] for i in top_archs_sorted_indices]
    top_archs_perf = [top_archs_perf[i] for i in top_archs_sorted_indices]
    with open(os.path.join(args.output_dir, 'arch_pool.final'), 'w') as fa:
        with open(os.path.join(args.output_dir, 'arch_pool.perf.final'), 'w') as fp:
            for arch, perf in zip(top_archs, top_archs_perf):
                arch = ' '.join(map(str, arch[0] + arch[1]))
                fa.write('{}\n'.format(arch))
                fp.write('{}\n'.format(perf))

    import shutil
    import tempfile
    try:
        torch.cuda.empty_cache()
        logging.info('Cleaned up GPU cache')
    except Exception as e:
        logging.warning(f'Failed to clean GPU cache: {e}')

    print("LoRA + CVaR stepsize combined search complete")
  

if __name__ == '__main__':
    main()
    print("Lora Rank = 16 and 10 operation search, saved in exp_fixed_10op")
