"""Single-path cell supernet with shared bases and input-specific LoRA adapters."""
import copy
import torch
from torch import nn
from torch.nn import functional as F

from retrain_model.reference_model import (BACKBONE, CELL, SEARCH_OPS, CellOperation,
                                           ReferenceAutoDeepLab, resize, scaled_size,
                                           validate_architecture)

SEARCH_VERSION = 'adl_cell_lora_cvar_v2'
# The tensor layout is unchanged, so completed v1 supernets remain evaluable.
SUPPORTED_SEARCH_VERSIONS = ('adl_cell_lora_cvar_v1', SEARCH_VERSION)


class InputLoRAConv(nn.Module):
    def __init__(self, conv, inputs, rank):
        super().__init__()
        self.conv = conv
        self.source = 0
        self.active_sources = set()
        fan_in = conv.weight[0].numel()
        self.rank = min(rank, conv.out_channels, fan_in)
        # Fixed alpha=effective rank -> scale=1. No architecture-probability gate.
        self.lora_A = nn.Parameter(torch.empty(inputs, self.rank, fan_in))
        self.lora_B = nn.Parameter(torch.zeros(inputs, conv.out_channels, self.rank))
        for weight in self.lora_A:
            nn.init.kaiming_uniform_(weight, a=5 ** 0.5)

    def forward(self, x):
        if self.training:
            self.active_sources.add(self.source)
        delta = (self.lora_B[self.source] @ self.lora_A[self.source]).reshape_as(self.conv.weight)
        return F.conv2d(x, self.conv.weight + delta, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)


class InputBN(nn.Module):
    def __init__(self, template, inputs):
        super().__init__()
        self.norms = nn.ModuleList(copy.deepcopy(template) for _ in range(inputs))
        self.source = 0

    def forward(self, x):
        return self.norms[self.source](x)


def specialize(module, inputs, rank):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and child.groups == 1 and rank > 0:
            setattr(module, name, InputLoRAConv(child, inputs, rank))
        elif isinstance(child, nn.BatchNorm2d):
            setattr(module, name, InputBN(child, inputs))
        else:
            specialize(child, inputs, rank)


class SearchCell(nn.Module):
    def __init__(self, reference_cell, channels, stride, rank):
        super().__init__()
        self.current = reference_cell.current
        self.previous = reference_cell.previous
        self.upsample = reference_cell.upsample
        self.stride = stride
        self.cell = CELL
        self.bank = nn.ModuleDict()
        for node in range(5):
            for operation in SEARCH_OPS:
                for s in sorted({1, stride}):
                    op = CellOperation(channels, operation, s)
                    specialize(op, node + 2, rank)
                    self.bank['{}_{}_{}'.format(node, operation, s)] = op

    def forward(self, previous, current, drop_probability=0):
        if self.upsample:
            current = resize(current, scaled_size(current, 2))
        if previous.shape[-2:] != current.shape[-2:]:
            previous = resize(previous, current.shape[-2:])
        states = [self.current(current), self.previous(previous)]
        for node in range(5):
            branches = []
            for source, operation in self.cell[2*node:2*node+2]:
                stride = self.stride if source < 2 else 1
                op = self.bank['{}_{}_{}'.format(node, operation, stride)]
                for module in op.modules():
                    if isinstance(module, (InputLoRAConv, InputBN)):
                        module.source = source
                branches.append(op(states[source]))
            states.append(branches[0] + branches[1])
        return torch.cat(states[2:], dim=1)


class SearchSupernet(ReferenceAutoDeepLab):
    def __init__(self, filter_multiplier=8, lora_rank=4):
        super().__init__(filter_multiplier=filter_multiplier, drop_path_prob=0)
        self.search_config = dict(filter_multiplier=filter_multiplier, lora_rank=lora_rank)
        previous_level = 0
        for index, level in enumerate(BACKBONE):
            self.cells[index] = SearchCell(self.cells[index], filter_multiplier * 2 ** level,
                                            2 if level > previous_level else 1, lora_rank)
            previous_level = level
        # Frequently switched paths need responsive search BN. Candidate ranking
        # recalibrates BN from training-A images, then restores these buffers.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0.1

    def set_architecture(self, cell):
        validate_architecture(cell, BACKBONE)
        for layer in self.cells:
            layer.cell = tuple(tuple(edge) for edge in cell)


def optimizer_groups(model, weight_decay, lora_multiplier):
    grouped = {(False, False): [], (False, True): [], (True, False): []}
    for name, p in model.named_parameters():
        adapter = 'lora_A' in name or 'lora_B' in name
        decay = p.ndim > 1 and not adapter
        grouped[(adapter, decay)].append(p)
    return [dict(params=params, weight_decay=weight_decay if decay else 0.,
                 lr_multiplier=lora_multiplier if adapter else 1.,
                 name='lora' if adapter else ('weights' if decay else 'bn_bias'))
            for (adapter, decay), params in grouped.items() if params]


def final_parameter_count(cell, filters=32):
    """Exact parameter count for the exported ordinary-convolution architecture."""
    validate_architecture(cell, BACKBONE)
    # Stem plus fixed F-dependent ASPP, decoder and classifier.
    # Head input is 10F; low projection input is 5F.
    c = 10 * filters
    total = 3*64*9 + 2*64 + 64*64*9 + 2*64 + 64*128*9 + 2*128
    total += 2*(c*256 + 512) + 3*(c*256*9 + 512)
    total += 1280*256 + 512 + 5*filters*48 + 96
    total += 304*256*9 + 512 + 256*256*9 + 512 + 256*19 + 19
    previous_channels, current_channels, previous_level = 64, 128, 0
    for level in BACKBONE:
        c = filters * 2 ** level
        total += current_channels*c + 2*c
        if previous_channels != c:
            total += previous_channels*c + 2*c
        for source, operation in cell:
            if operation.startswith('sep'):
                total += 2*c*int(operation[-1])**2 + 2*c*c + 4*c
            elif operation.startswith('dil'):
                total += c*c*int(operation[-1])**2 + 2*c
            elif operation == 'skip' and level > previous_level and source < 2:
                total += c*c + 2*c
        previous_channels, current_channels = current_channels, 5*c
        previous_level = level
    return total
