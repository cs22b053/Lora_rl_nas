"""Diagnostic: Compare first-run checkpoints (data_temp) vs resumed run (data_3000)"""
import torch
import math

# First run checkpoints
first_run_files = [
    'data_temp/deeplab_autodeeplab_cityscapes_v3_cvar_lora_retrain_epoch300.pth',
#    'data_temp/deeplab_autodeeplab_cityscapes_v3_cvar_lora_retrain_epoch500.pth',
#    'data_temp/deeplab_autodeeplab_cityscapes_v3_cvar_lora_retrain_epoch550.pth',
    'data_temp/deeplab_autodeeplab_cityscapes_v3_cvar_lora_retrain_epoch600.pth',
]

# Resumed run checkpoints
resumed_files = [
    'data_3000/deeplab_autodeeplab_cityscapes_v3_cvar_lora_retrain_epoch1800.pth',
    'data_3000/deeplab_autodeeplab_cityscapes_v3_cvar_lora_retrain_epoch2000.pth',
]

base_lr = 0.05
warmup_iters = 1000
warmup_start_lr = 5e-6
iters_per_epoch = 185

def expected_lr(epoch, total_epochs):
    """What the LR SHOULD be at this epoch for a given total_epochs setting"""
    max_iter = iters_per_epoch * total_epochs
    it = epoch * iters_per_epoch
    if it < warmup_iters:
        warmup_factor = (base_lr / warmup_start_lr) ** (1.0 / warmup_iters)
        return warmup_start_lr * (warmup_factor ** it)
    else:
        return base_lr * pow((1 - (it - warmup_iters) / (max_iter - warmup_iters)), 0.9)

print("=" * 90)
print("FIRST RUN CHECKPOINTS (data_temp/)")
print("=" * 90)
for f in first_run_files:
    try:
        checkpoint = torch.load(f, map_location='cpu')
        epoch = checkpoint['epoch']
        actual_lr = checkpoint['optimizer']['param_groups'][0]['lr']
        
        # Check which --epochs setting matches
        lr_600 = expected_lr(epoch, 600)
        lr_1000 = expected_lr(epoch, 1000)
        lr_3000 = expected_lr(epoch, 3000)
        lr_4000 = expected_lr(epoch, 4000)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Actual LR in checkpoint:    {actual_lr:.10f}")
        print(f"  Expected LR (--epochs=600): {lr_600:.10f}  {'<<< MATCH' if abs(actual_lr - lr_600) < 1e-6 else ''}")
        print(f"  Expected LR (--epochs=1000):{lr_1000:.10f}  {'<<< MATCH' if abs(actual_lr - lr_1000) < 1e-6 else ''}")
        print(f"  Expected LR (--epochs=3000):{lr_3000:.10f}  {'<<< MATCH' if abs(actual_lr - lr_3000) < 1e-6 else ''}")
        print(f"  Expected LR (--epochs=4000):{lr_4000:.10f}  {'<<< MATCH' if abs(actual_lr - lr_4000) < 1e-6 else ''}")
    except Exception as e:
        print(f"  Error loading {f}: {e}")

print()
print("=" * 90)
print("DIAGNOSIS")
print("=" * 90)

# Load epoch 600 from first run
try:
    ckpt_600 = torch.load(first_run_files[-1], map_location='cpu')
    lr_600_actual = ckpt_600['optimizer']['param_groups'][0]['lr']
    epoch_600 = ckpt_600['epoch']
    
    # Load epoch 1800 from resumed run
    ckpt_1800 = torch.load(resumed_files[0], map_location='cpu')
    lr_1800_actual = ckpt_1800['optimizer']['param_groups'][0]['lr']
    
    print(f"\nFirst run ended at epoch {epoch_600} with LR = {lr_600_actual:.10f}")
    print(f"Resumed run at epoch 1800 has LR = {lr_1800_actual:.10f}")
    
    # The key question: at epoch 600, what was the LR?
    # If --epochs=600 was used: LR should be ~0 (training was "done")
    # If --epochs=3000 was used: LR should be ~0.0410
    lr_if_600 = expected_lr(600, 600)
    lr_if_3000 = expected_lr(600, 3000)
    
    print(f"\nAt epoch 600:")
    print(f"  If first run used --epochs=600:  LR would be {lr_if_600:.10f} (schedule done)")
    print(f"  If first run used --epochs=3000: LR would be {lr_if_3000:.10f} (20% through)")
    print(f"  Actual LR at epoch 600:          {lr_600_actual:.10f}")
    
    if abs(lr_600_actual - lr_if_3000) < abs(lr_600_actual - lr_if_600):
        print(f"\n  >>> First run used --epochs=3000 (or similar long schedule)")
        print(f"  >>> LR schedule was CONSISTENT across resume! The problem is elsewhere.")
    else:
        print(f"\n  >>> First run used --epochs=600!")
        print(f"  >>> LR went to ~0 at epoch 600, then JUMPED back to 0.041 on resume!")
        print(f"  >>> THIS IS THE ROOT CAUSE of the performance drop!")
except Exception as e:
    print(f"Error: {e}")
