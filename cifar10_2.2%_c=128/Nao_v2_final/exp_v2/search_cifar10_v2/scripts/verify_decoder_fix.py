#!/usr/bin/env python3
"""Verify decoder fix logic."""
num_nodes_per_cell = 5
length = 40  # decoder_length = 4 * child_nodes * 2

print("=== With num_nodes = length // 4 =", length//4, "(CURRENT FIX) ===")
nn = length // 4  # = 10
for step in [0,2,4,6,8,10,12,14,16,18, 20,22,24,26,28,30,32,34,36,38]:
    if step % 2 == 0:
        node_in_cell = step // 2 % (2 * nn) // 2
        max_token = 2 + node_in_cell + 1
        cell = 'conv' if step < length//2 else 'reduc'
        actual_node = (step % (length//2)) // 4
        print(f'  step={step:2d} ({cell} node {actual_node}): computed_node={node_in_cell}, range [1:{max_token}] = {max_token-1} choices')

print()
print("=== With num_nodes = length // 8 =", length//8, "(CORRECTED) ===")
nn = length // 8  # = 5
for step in [0,2,4,6,8,10,12,14,16,18, 20,22,24,26,28,30,32,34,36,38]:
    if step % 2 == 0:
        node_in_cell = step // 2 % (2 * nn) // 2
        max_token = 2 + node_in_cell + 1
        cell = 'conv' if step < length//2 else 'reduc'
        actual_node = (step % (length//2)) // 4
        print(f'  step={step:2d} ({cell} node {actual_node}): computed_node={node_in_cell}, range [1:{max_token}] = {max_token-1} choices')
