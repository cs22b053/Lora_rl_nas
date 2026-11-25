"""Diagnostic: Inspect checkpoint to verify what's saved and the actual LR at save time"""
import sys
sys.stdout = open("debug_checkpoint_output.txt", "w")
import torch
import glob
import os

data_dir = 'data_3000'

# Find all checkpoints
files = glob.glob(f'{data_dir}/deeplab_autodeeplab_cityscapes_v3_cvar_lora_retrain_epoch*.pth')
files.sort(key=lambda x: int(x.split('epoch')[1].split('.pth')[0]))

print(f"Found {len(files)} checkpoints:")
print()

for f in files:
    checkpoint = torch.load(f, map_location='cpu')
    epoch = checkpoint['epoch']
    
    # Check optimizer state
    opt_state = checkpoint['optimizer']
    param_groups = opt_state['param_groups']
    lrs = [pg['lr'] for pg in param_groups]
    
    # Check if momentum buffers exist
    has_momentum = any('momentum_buffer' in v for v in opt_state['state'].values()) if opt_state['state'] else False
    n_state_entries = len(opt_state['state'])
    
    print(f"Checkpoint: epoch={epoch}")
    print(f"  Optimizer LRs: {lrs}")
    print(f"  # param groups: {len(param_groups)}")
    print(f"  # state entries (momentum buffers): {n_state_entries}")
    print(f"  Has momentum buffers: {has_momentum}")
    print(f"  Keys in checkpoint: {list(checkpoint.keys())}")
    print(f"  File size: {os.path.getsize(f) / 1024 / 1024:.1f} MB")
    print()

    # Only inspect first state entry details
    if opt_state['state']:
        first_key = list(opt_state['state'].keys())[0]
        first_state = opt_state['state'][first_key]
        print(f"  First param state keys: {list(first_state.keys())}")
    print()
