import sys
sys.path.insert(0, '/home/ajin/temp_2/NAO_pytorch/NAO_V2')
from operations import LoRAWSSepConv, LoRAWSDilSepConv, LoRAWSReLUConvBN
from operations import OPERATIONS_search_small, OPERATIONS_search_middle
import torch

# Test LoRAWSSepConv forward
m = LoRAWSSepConv(num_possible_inputs=4, C_in=16, C_out=16, kernel_size=3, padding=1, lora_rank=4)
x = torch.randn(2, 16, 8, 8)
out = m(x, x_id=0, stride=1)
assert out.shape == (2, 16, 8, 8), f'Expected (2,16,8,8), got {out.shape}'
print(f'LoRAWSSepConv output shape: {out.shape}')

# Test LoRAWSDilSepConv forward
m2 = LoRAWSDilSepConv(num_possible_inputs=4, C_in=16, C_out=16, kernel_size=3, padding=2, dilation=2, lora_rank=4)
out2 = m2(x, x_id=1, stride=1)
assert out2.shape == (2, 16, 8, 8), f'Expected (2,16,8,8), got {out2.shape}'
print(f'LoRAWSDilSepConv output shape: {out2.shape}')

# Test LoRAWSReLUConvBN forward
m3 = LoRAWSReLUConvBN(num_possible_inputs=4, C_out=16, C_in=16, kernel_size=1, lora_rank=4)
out3 = m3(x, x_id=[0, 1], bn_train=False)
assert out3.shape[1] == 16
print(f'LoRAWSReLUConvBN output shape: {out3.shape}')

# Verify LoRA B params are zero-initialized
for name, p in m.named_parameters():
    if 'lora_B' in name:
        assert (p == 0).all(), f'{name} should be zero-initialized'
print('LoRA B zero-initialization: OK')

# Verify operation dicts use LoRA variants
for k, v in OPERATIONS_search_small.items():
    op = v(4, 16, 16, 1, True, lora_rank=8)
    print(f'  search_small[{k}] -> {type(op).__name__}')

for k, v in OPERATIONS_search_middle.items():
    op = v(4, 16, 16, 1, True, lora_rank=8)
    print(f'  search_middle[{k}] -> {type(op).__name__}')

# Count LoRA params vs base params
lora_count = sum(p.numel() for n, p in m.named_parameters() if 'lora' in n)
base_count = sum(p.numel() for n, p in m.named_parameters() if 'lora' not in n)
print(f'LoRAWSSepConv params: base={base_count}, lora={lora_count}, ratio={lora_count/base_count:.2%}')

print('All LoRA operation tests passed!')
