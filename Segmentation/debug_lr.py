"""Diagnostic: Simulate the poly LR schedule to check resume behavior"""
import sys
sys.stdout = open("debug_lr_output.txt", "w")
import math

base_lr = 0.05
warmup_iters = 1000
warmup_start_lr = 5e-6

# Cityscapes train set ~2975 images, batch_size=16 => iters/epoch
iters_per_epoch = 2975 // 16  # = 185
epochs = 3000
max_iteration = iters_per_epoch * epochs

warmup_factor = (base_lr / warmup_start_lr) ** (1.0 / warmup_iters)

def get_lr(iteration, max_iter):
    if iteration < warmup_iters:
        return warmup_start_lr * (warmup_factor ** iteration)
    else:
        return base_lr * pow((1 - (iteration - warmup_iters) / (max_iter - warmup_iters)), 0.9)

# Show LR at key epochs
key_epochs = [0, 5, 50, 100, 200, 300, 400, 500, 600, 900, 1200, 1500, 1800, 2000, 2400, 2500, 2650, 2750, 2900, 2950, 3000]
print(f'iters_per_epoch = {iters_per_epoch}')
print(f'max_iteration (3000 ep) = {max_iteration}')
print(f'warmup ends at iter {warmup_iters} (epoch ~{warmup_iters/iters_per_epoch:.1f})')
print()
print(f'{"Epoch":>6} | {"Iteration":>10} | {"LR (3000ep)":>14} | {"LR (600ep)":>14} | {"LR (4000ep)":>14}')
print('-' * 75)

max_iter_600 = iters_per_epoch * 600
max_iter_4000 = iters_per_epoch * 4000

for e in key_epochs:
    it = e * iters_per_epoch
    lr_3000 = get_lr(min(it, max_iteration - 1), max_iteration)
    lr_600  = get_lr(min(it, max_iter_600 - 1), max_iter_600) if it < max_iter_600 else 0
    lr_4000 = get_lr(min(it, max_iter_4000 - 1), max_iter_4000)
    print(f'{e:6d} | {it:10d} | {lr_3000:14.8f} | {lr_600:14.8f} | {lr_4000:14.8f}')

print()
print("=" * 75)
print("KEY QUESTION: What --epochs did you use in the FIRST run (0-600)?")
print("If you used --epochs=600 first, then --epochs=3000 on resume,")
print("the LR schedule RESTARTS because max_iteration changes!")
print()
print("Example: At epoch 300 with --epochs=600:")
print(f"  LR = {get_lr(300*iters_per_epoch, max_iter_600):.8f}  (50% through schedule)")
print(f"Example: At epoch 300 with --epochs=3000:")
print(f"  LR = {get_lr(300*iters_per_epoch, max_iteration):.8f}  (10% through schedule)")
print()
print("The poly schedule shape COMPLETELY changes with different total epochs!")
print("At epoch 600 with --epochs=600: LR ≈ 0 (schedule is done)")
print(f"At epoch 600 with --epochs=3000: LR = {get_lr(600*iters_per_epoch, max_iteration):.8f} (still high)")
