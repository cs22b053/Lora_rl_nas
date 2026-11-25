## License
The codes and models in this repo are released under the GNU GPLv3 license.

## Search by NAO-V2 + LoRA

To run the search phase for CNN architectures on CIFAR-10 using our modified NAO-V2 with LoRA, please use:
```bash
sbatch run_search_v2.sh
```
> **Note:** The complete execution log for our successful search run is preserved in [`nao_search_44276.out`](./nao_search_44276.out).

Once the search is completed, the generated architecture pools are saved in the `exp_v2/search_cifar10_v2/` directory. The absolute best discovered architectures from the search are specifically stored in:
- **`exp_v2/search_cifar10_v2/arch_pool.final`**

## Training the Best Architecture

You can train the best discovered architecture from the pool by passing it via the `--arch` argument in the training script. 

To train our best discovered 10-op architecture from scratch on CIFAR-10, use:
```bash
sbatch run_train_v2.sh
```

### Our Best Result (CIFAR-10)
| Dataset | Script | GPU | Time | Top1 Error Rate | Training Log |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | 
|CIFAR-10| run_train_v2.sh | 1 DGX-1 | ~5 day | **2.00%** | [`nao_train_44453.out`](./nao_train_44453.out) |

> **Note:** The complete training log demonstrating the progression to the **2.00% error rate** (98.00% validation accuracy) is preserved in [`nao_train_44453.out Line 2186`](./nao_train_44453.out Line 2186).

