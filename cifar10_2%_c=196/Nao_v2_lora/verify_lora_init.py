"""Verify that LoRA-B matrices remain zero-initialized after model construction."""
import sys
import argparse
import torch
import torch.nn as nn

args = argparse.Namespace(
    search_space='middle',
    child_layers=3,
    child_nodes=5,
    child_channels=20,
    child_keep_prob=1.0,
    child_drop_path_keep_prob=0.9,
    child_use_aux_head=False,
    child_num_ops=10,
    steps=1000,
)

from model_search import NASWSNetworkCIFAR
model = NASWSNetworkCIFAR(args, 10, 3, 5, 20, 1.0, 0.9, False, 1000, lora_rank=4)

lora_b_count = 0
lora_b_nonzero = 0
lora_a_count = 0
for name, p in model.named_parameters():
    if 'lora_B' in name or 'lora_b' in name:
        lora_b_count += 1
        if p.abs().max() > 0:
            lora_b_nonzero += 1
            print('  FAIL: %s has max=%.6f' % (name, p.abs().max().item()))
    if 'lora_A' in name or 'lora_a' in name:
        lora_a_count += 1

print('Total lora_B params:', lora_b_count)
print('Total lora_A params:', lora_a_count)
if lora_b_nonzero == 0:
    print('SUCCESS: All lora_B matrices are zero-initialized!')
else:
    print('FAIL: %d / %d lora_B matrices are non-zero!' % (lora_b_nonzero, lora_b_count))

total = sum(p.numel() for p in model.parameters())
lora_total = sum(p.numel() for n, p in model.named_parameters() if 'lora' in n)
print('Total params: %.2fM, LoRA params: %.2fM (%.1f%%)' % (total/1e6, lora_total/1e6, 100*lora_total/total))
