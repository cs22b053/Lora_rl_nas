# LoRA + CVaR-PPO Neural Architecture Search

This project adds two components to existing weight-sharing NAS methods and
applies them to five benchmarks:

1. **LoRA adapters in the weight-sharing supernet.** A shared operation keeps
   one base weight. Each possible input to that operation adds its own low-rank
   residual, `W = W_base + A·B`, with `B = 0` at initialisation. The aim is to
   reduce interference between architectures that share weights. Adapters exist
   only during search; final networks are ordinary models trained from scratch.
2. **A PPO controller with a CVaR (worst-case, α = 0.2) objective.** It proposes
   architectures instead of relying only on the base method's fixed rule.

The CNN and RNN experiments extend **NAO-V2 / NAO-WS** (Luo et al.). The
segmentation experiment extends **Auto-DeepLab** (Liu et al.).

---

## Results at a glance

| Task | Folder | Metric | Our result | Params | Main log |
| --- | --- | --- | ---: | ---: | --- |
| CIFAR-10 | [`cifar10_2%_c=196/`](cifar10_2%25_c=196/Nao_v2_lora/) | test error | **2.00%** (best epoch) · 2.22% final · 2.20% SWA | 86.9M | `nao_train_44453.out` line 2186 |
| CIFAR-10 (smaller) | [`cifar10_2.2%_c=128/`](cifar10_2.2%25_c=128/Nao_v2_final/) | test error | **2.14%** (SWA) · 2.20% best non-SWA | 39.2M | `nao_train_44736.out` (last line) |
| CIFAR-100 | [`cifar100_14.5/`](cifar100_14.5/) | test error | **14.48%** (best epoch) · 14.66% final | 87.2M | `nao_test_c100_44989.out` |
| ImageNet | [`Imagenet_24.3/`](Imagenet_24.3/Nao_test/) | top-1 / top-5 error | **24.29% / 7.85%** (final epoch 300) | 10.4M | `nao_test_imagenet_46117.out` |
| Penn Treebank | [`ptb_rnn_56.81/`](ptb_rnn_56.81/rnn/) | test perplexity | **56.81** (static) · 52.11 dynamic eval* | 27.3M | `ptb_train47041.out` |
| Cityscapes segmentation | [`Segmentation/`](Segmentation/) | val mIoU | **74.29** single-scale · **76.32** multi-scale+flip | 10.1M | `posteval_49242_1.out` |

Every number in this table can be found in the log named in its row. The
sections below give the exact lines.

\* The dynamic-evaluation hyperparameters (lr, λ) were chosen by grid search
on **test** perplexity. 52.11 is therefore test-tuned. **56.81** is the clean number.

---

## Read these caveats before comparing numbers

- **CIFAR "best epoch" numbers are the maximum test accuracy over 1000 epochs.**
  In this NAO code, `valid_acc` is measured on the official CIFAR test set
  (`train=False`). The final-epoch and SWA numbers do not depend on that
  choice, so they are listed as well.
- **Search happened once, on CIFAR-10.** The CIFAR-10, CIFAR-100 and ImageNet
  networks all use the **same cell**, searched on CIFAR-10 (`nao_search_44276.out`)
  and transferred. They differ only in width and depth.
- **Each number comes from one training run with one seed.** No error bars.
- **Model sizes differ from the papers we compare against.** For example,
  86.9M parameters at CIFAR-10 width 192. Compare at equal parameter count before
  claiming a method improvement.
- **The folder name `cifar10_2%_c=196` does not match its run.** The run used
  `--channels=192`, not 196. Folder names are kept unchanged because the logs
  refer to them.
- Each subfolder's `exp_v2/<run>/scripts/` holds the **exact copy of the code**
  that was running when that run started. NAO makes this copy automatically.

---

## 1. CNN experiments (NAO-V2 + LoRA + CVaR-PPO)

Applies to `cifar10_2%_c=196/Nao_v2_lora/`, `cifar10_2.2%_c=128/Nao_v2_final/`,
`cifar100_14.5/` and `Imagenet_24.3/Nao_test/`. The four folders share almost
identical code.

### Algorithm
1. **Supernet (`model_search.py`, `operations.py`).** NAO-V2's
   weight-sharing CNN uses the "middle" search space with 10 operations:
   zero, identity, separable conv 3/5/7, dilated separable conv 3/5/7,
   avg-pool and max-pool. Convolutions are `LoRAWSSepConv` /
   `LoRAWSDilSepConv` / `LoRAWSReLUConvBN`: a shared base weight plus a
   **rank-16 LoRA adapter for each possible input node**. Adapter LR is 10×
   the base LR.
2. **NAO controller (`controller.py`, `encoder.py`, `decoder.py`).** An
   encoder maps an architecture string to a continuous embedding. A predictor
   estimates its accuracy, and a decoder turns embeddings back into architectures.
3. **PPO policy (`ArchitecturePolicy.py`, `PPOArchTrainer.py`).** An
   actor-critic reads `(embedding, predicted accuracy)` and outputs a
   continuous **step direction** in embedding space. The reward is the
   predictor's estimate, so no child training is needed per step.
4. **CVaR objective (`PPOArchTrainer.py` around line 204).** The policy loss is
   the mean clipped PPO surrogate over the **worst 20%** of samples, ranked by
   the unclipped surrogate. This is a risk-averse version of PPO.
5. **Search loop (`train_search.py`).** Iteration 0 uses NAO's own gradient
   steps. Later iterations generate half the new architectures with NAO steps
   and half with PPO+CVaR. There are 4 iterations, 2,000 seed architectures and
   80 supernet epochs. The top 5 are re-ranked at the end.
6. **Final training (`test_cifar.py` / `test_imagenet.py` with `model.py`).**
   The discovered cell is trained from scratch, without LoRA.

Discovered cell, used by all four CNN folders. The string lists the normal
cell, then the reduction cell, as `in1 op1 in2 op2` for each of 5 nodes:
```
0 1 0 4 2 3 1 8 1 8 3 3 0 8 2 8 2 1 1 8 | 0 4 0 3 0 7 0 0 2 8 1 3 1 8 0 1 2 0 0 2
```

### Where to look
| What | File |
| --- | --- |
| Search log (CIFAR-10, LoRA rank 16, 10 ops, 4 iterations) | `cifar10_2%_c=196/Nao_v2_lora/nao_search_44276.out` (identical copy in `cifar100_14.5/`) |
| Search outputs (architecture pools, top-5 in `arch_pool.final`) | `*/exp_v2/search_cifar10_v2/` |
| CIFAR-10 w=192 training | `nao_train_44289 → 44402 → 44453.out` (one run, resumed). Best 98.00% at line 2186. `nao_train_44531.out` holds the SWA evaluation (97.80%). |
| CIFAR-10 w=128 training | `cifar10_2.2%_c=128/Nao_v2_final/nao_train_44667 → 44736.out`. The final line reads `SWA valid_acc 97.86`. |
| CIFAR-100 w=192 (headline) | `cifar100_14.5/nao_test_c100_44895 → 44946 → 44989.out`. Best 85.52%, final 85.34%. |
| CIFAR-100 w=128 | `cifar100_14.5/nao_test_c100_44796 → 44871.out`. Best 85.25%, final 85.05%. |
| ImageNet 14 layers, 48 channels, 300 epochs, cosine LR | `Imagenet_24.3/Nao_test/nao_test_imagenet_45772 … 46117.out` (resumed chain). Final top-1 75.71%, top-5 92.15%. `45333.out` is a 2-epoch smoke test. |
| Launchers | CIFAR-10: `run_search_v2.sh`, `run_train_v2.sh`. CIFAR-100: `run_test_cifar100.sh` (w=128), `run_test_cifar100_2.sh` (w=192). ImageNet: `run_train_imagenet_3.sh`. |

Training recipe (CIFAR): 1000 epochs, batch 96, AutoAugment, cutout 16,
label smoothing 0.1, auxiliary head, drop-path keep 0.8, cosine LR with
maximum 0.03, SWA from epoch 750. The full `Args = Namespace(...)` is on the
first lines of every log.

---

## 2. Language model (NAO-WS RNN + LoRA + CVaR-PPO): `ptb_rnn_56.81/rnn/`

### Algorithm
The approach is the same as for the CNNs, applied to NAO-WS's recurrent cell:
- **Supernet:** `model_search.py` with `rnn_lora_cell.py`. Rank-16 LoRA,
  1.39M adapter parameters (2.7% of the supernet). Adapter LR is 2×.
- **NAO controller:** `controller.py`, `encoder/`, `decoder/`.
- **PPO+CVaR policy:** `ArchitecturePolicy.py`, `PPOArchTrainer.py`.
- **Search loop:** `train_search.py`. Use `--no_cvar_rl` to disable PPO+CVaR.
- **Final training:** `train.py` trains the chosen cell from scratch with
  embedding and hidden size 850, for 8000 epochs.

### Results and where to look
| Run | Architecture source | Test PPL | Log |
| --- | --- | ---: | --- |
| **LoRA + CVaR-PPO (headline)** | `[0,3,1,2,1,2,3,2,3,2,3,2,1,1,1,2,3,2,5,1,5,3]` | **56.81** | `ptb_train46616 → 46656 → 46710 → 47041.out`. The last line of `47041` reads `test ppl 56.81` (valid 59.20). |
| Dynamic evaluation of the same model | n/a | 52.11* | `ptb_dyneval_48676.out` |
| Plain NAO-WS baseline (no LoRA/CVaR) | baseline search `logs/train.search_ptb.log` | 58.43 | `logs/train.log` |
| LoRA + CVaR-PPO, search v2 | `logs/train.search_ptb_lora_cvar_v2.log` | 57.95 | `logs/train.best_lora_cvar_v2.log` |

Where the headline architecture came from:
- It appears in the LoRA+CVaR search log, `ptb_lora46597.out` line 3065,
  during supernet epoch 617. The same run was also logged to
  `logs/train.search_ptb_lora_cvar_1.log`, where it is line 3027.
- `run_train.sh` records it as the best architecture of that search.
- The search job later stopped with an `OverflowError` in a logging
  `math.exp` call, so no final architecture pool was written for it.

`logs/train.best_lora_cvar.log` combines several runs in one file. Lines before
23951 belong to an earlier, abandoned run that used a different architecture.
The 56.81 run starts at line 23951 (`save=models/train_best_lora_cvar_1`).
`logs/train.v3a.log` is an unfinished rerun.

---

## 3. Semantic segmentation (Auto-DeepLab + LoRA + CVaR-PPO): `Segmentation/`

This part uses a separate code base and its own version of the method:
- The search is a **cell-level, single-path** search over Auto-DeepLab's
  12-layer macro path.
- LoRA adapters are input-specific, on a shared base, with rank 4.
- The PPO controller is rewarded with `0.8·mIoU + 0.2·CVaR₀.₂`, where the CVaR
  term is computed over **per-image IoU** on held-out *training* images.
- The finalists are re-checked by independent short trainings before one
  full retraining.

| Model (F20, 500k updates, seed 0) | Single-scale mIoU | Multi-scale + flip mIoU |
| --- | ---: | ---: |
| Published Auto-DeepLab cell (our reproduction) | 73.94 | 75.99 |
| **Searched candidate 4** | **74.29** | **76.32** |

A separate global-search cell at F32 and 740k updates reaches 75.08 single-scale.

[`Segmentation/RESULTS.md`](Segmentation/RESULTS.md) has the full algorithm,
a table mapping every log file to its number, and the limits of the result.
That file is the place to start for this part.

---

## Repository layout

```
.
├── README.md                     ← this file
├── cifar10_2%_c=196/Nao_v2_lora/ ← CIFAR-10, width 192 (search log + training logs)
├── cifar10_2.2%_c=128/Nao_v2_final/ ← CIFAR-10, width 128
├── cifar100_14.5/                ← CIFAR-100 (cell transferred from CIFAR-10)
├── Imagenet_24.3/Nao_test/       ← ImageNet (cell transferred from CIFAR-10)
├── ptb_rnn_56.81/rnn/            ← Penn Treebank RNN search + training
└── Segmentation/                 ← Cityscapes Auto-DeepLab variant (see RESULTS.md)
```

Log naming: `<job>_<slurm-job-id>.out` is the SLURM stdout and `.err` is
stderr. A long run hit the 48-hour job limit and was resumed several times, so
it is spread over several logs. Read them in increasing job-id order. Every
NAO log starts with the full argument list (`Args = Namespace(...)`).

## Not included in git
- **Checkpoints** (`*.pt`, `*.pth`): download them from this repository's
  **GitHub Releases** page.
- **Datasets**:
  - CIFAR-10 and CIFAR-100 download automatically through torchvision into `data/`.
  - ImageNet (ILSVRC-2012) must be obtained separately.
  - Penn Treebank goes in `ptb_rnn_56.81/rnn/data/penn/` (standard Mikolov splits).
  - Cityscapes `leftImg8bit` and `gtFine` come from cityscapes-dataset.com.

## Hardware and software
All runs used a SLURM cluster with NVIDIA A100-40GB GPUs (DGX-1 node). The
Segmentation runs used PyTorch 2.3.1 with CUDA 11.8. The NAO runs used a conda
environment set up in each `run_*.sh`.

## Upstream work
- NAO / NAO-V2: Luo et al., *Neural Architecture Optimization* (NeurIPS 2018)
  and *Understanding and Improving One-shot NAO* (arXiv:1808.07233). The CNN
  and RNN code derives from the official NAO repositories (GPLv3).
- Auto-DeepLab: Liu et al., CVPR 2019. The segmentation code derives from
  NoamRosenberg/autodeeplab.
- LoRA: Hu et al., 2021. PPO: Schulman et al., 2017. CVaR policy gradients:
  Tamar et al., 2015.
