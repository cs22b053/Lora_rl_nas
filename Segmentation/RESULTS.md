# Auto-DeepLab + LoRA supernet + CVaR-reward PPO: results and how to read the logs

This repository is a PyTorch Auto-DeepLab variant for Cityscapes semantic
segmentation. The architecture search uses a weight-sharing supernet with
**input-specific LoRA adapters**. A **PPO controller** chooses cells and is
rewarded with a **CVaR (worst-image) risk-adjusted IoU**. The searched cell is
then retrained from scratch as an ordinary network, with no LoRA and no RL.

All numbers below come from the log files in this folder. Every run used
Cityscapes `train` (2,975 images) for training. Every score is measured on all
500 official `val` images, on one NVIDIA A100-40GB, with PyTorch 2.3.1 + CUDA 11.8.

---

## 1. Headline results

### 1a. Matched comparison: published cell vs. searched candidate 4 (F20, 500k updates)

Both runs use the same recipe. Filter multiplier F20, 500,000 updates,
batch 16, crop 769, SGD with LR 0.05, poly power 0.9, 1k warmup steps,
BN momentum 0.01, no drop-path, seed 0. The only difference is the cell.

| Protocol | Published Auto-DeepLab cell (reproduced) | **Candidate 4 (ours)** | Δ |
| --- | ---: | ---: | ---: |
| Single-scale, best checkpoint (AMP) | 73.942 (@475k) | **74.285** (@490k) | +0.34 |
| Single-scale, final 500k checkpoint | 73.117 | **74.132** | +1.02 |
| Single-scale best, FP32, stored BN | 73.943 | **74.288** | +0.35 |
| Single-scale best, FP32, train-recalibrated BN | 73.909 | **74.181** | +0.27 |
| **Multi-scale (0.5–1.5) + flip TTA, best checkpoint** | 75.989 | **76.323** | +0.33 |
| Worst-20% image IoU (CVaR), TTA | 56.411 | 56.037 | −0.37 |
| Parameters | 10,147,427 | 10,121,187 | −26k |

For reference, the Auto-DeepLab paper reports 75.20 for Auto-DeepLab-S
(F20, 500k, no SDP), measured single-scale. Compare that number with our
single-scale rows, not with the TTA row.

### 1b. Global search winner, F32, 4,000 epochs (740k updates)

| Run | Protocol | mIoU |
| --- | --- | ---: |
| Searched cell `architecture_seed0_halfres_reranked.json`, F32, 19.48M params | Single-scale, AMP | **75.079** |

This run used F32, 740,000 updates (4,000 epochs × 185 batches), BN momentum 0.01
and drop-path 0.1. No reference cell was trained under the same F32/740k recipe,
so this number has no matched control. The paper's F32 figure is 80.04,
obtained with a 1.5M-update schedule, about twice as long as this run.

---

## 2. Which log shows which number

Logs are SLURM outputs named `<job>_<jobid>[_<arrayindex>].out`. A job that hit
the 48-hour limit was resubmitted and resumed from `latest.pth`, so one long
training run can be split across several log files. Read those files in jobid order.

### Final results (start here)

| Log file | What it is | Key line |
| --- | --- | --- |
| `posteval_49242_1.out` | Candidate 4 best checkpoint, multi-scale + flip TTA | `Mean IoU 76.323%` |
| `posteval_49242_0.out` | Published-cell reference best checkpoint, same TTA | `Mean IoU 75.989%` |
| `candidate4_500k_48648.out` → `48846` → `49154` | Candidate 4 full 500k training (3 allocations) | `Best single-scale validation mIoU: 74.285%` |
| `reference_retrain_47702.out` → `47798` → `47894` | Published-cell 500k training (3 allocations) | `Best single-scale validation mIoU: 73.942%` |
| `selected_4000_47160` → `47162` → `47166` → `47332` → `47460` | F32 global-search cell, 740k training | `Best single-scale validation mIoU: 75.079%` |
| `selected_eval_47533.out` | Independent re-evaluation of that F32 checkpoint | `Mean IoU 75.079%` + per-class IoU |

Machine-readable versions:

- `posteval_candidate4_500k/tta/*/…/best_step_*.json`: TTA per-class IoU, confusion matrix, per-image IoU.
- `posteval_candidate4_500k/bn/*.json`: FP32 stored-BN vs. train-recalibrated-BN, for both best and final checkpoints of both models.
- `retrain_candidate4_f20_500k_seed0_nosdp/validation.jsonl` and `config.json`: every 5k-step validation and the exact training recipe.

### Search and selection logs

| Log file | Stage |
| --- | --- |
| `search_select_47077.out` | Re-ranks 758 visited cells from the global LoRA/CVaR search on 595 held-out *train* images, then exports `architecture_seed0_halfres_reranked.json` |
| `local_search_47973.out` | Reference-guided local search: 100 epochs, 29,700 weight updates, 50 PPO updates, 4.0 h |
| `local_search_47972.out` | Smoke test of the same search. It stopped on a non-finite AMP logit, and this led to the FP32 fallback fix. It is not a result. |
| `local_select_47987.out` | Re-ranks all 128 library cells and writes the reference + 4 challenger shortlist to `local_candidates_v1_seed0/` |
| `local_probe_47989_{0..4}.out` | Independent 50k-update probe training of shortlist IDs 0–4. ID 0 is the published cell. |
| `local_probe_48149_{0,1,4}.out` | The 3 survivors continued to 150k updates |

### Diagnostics, not results

| Log file | What it showed |
| --- | --- |
| `pilot_bn_47156_0.out` | 20k-update pilot of the **reference** cell with BN momentum 0.0003: 22.91% with stored BN, 59.76% after BN recalibration on train images |
| `pilot_bn_47156_1.out` | 20k-update pilot of the **selected** cell: 45.71% → 62.28% after recalibration |

These pilots showed that the running BN statistics were badly stale under
momentum 0.0003. Every later run therefore uses BN momentum **0.01**. The
pilot numbers measure 20k-step models; they are not final accuracies.

---

## 3. Algorithm

### 3.1 Search space

- **Macro path (fixed):** the published Auto-DeepLab 12-layer path
  `[0,0,0,1,2,1,2,2,3,3,2,1]`, where level *l* means output stride 4·2^l.
  It uses an ASPP head (rates 12/24/36) and a decoder on the third cell.
  This makes the method **cell-level NAS**, not a joint network/cell search.
- **Cell:** 5 nodes with 2 edges each. An edge is `[input_state, op]`. State 0
  is the previous cell's output and state 1 is the output of the cell before it.
  States 2–6 are the outputs of nodes 0–4. The cell output concatenates the five nodes.
- **Operations:** `sep3, sep5` (separable), `dil3, dil5` (full atrous), `skip`,
  `avg3`, `max3`. The global search allows at most 2 skips and a 24M parameter cap.

### 3.2 LoRA-parameterised supernet (search only)

For every operation shared by different input states, the convolution weight is

```
W_s = W_base + B_s · A_s          (rank 4, A random, B = 0 at init)
```

`W_base` is shared and trained from scratch. Each input state `s` gets its
own low-rank residual and its own BN statistics. The aim is to reduce
interference between cell choices that share one convolution. Adapters sit on
pointwise and atrous kernels only, not on depthwise convolutions, the stem,
ASPP or the decoder. Adapter LR is 2× the base LR, with no weight decay.
Adapter slices that are not selected keep their weights and momentum frozen
(snapshot and restore), because plain SGD momentum would otherwise move them.

This is a LoRA-style parameterisation with a trainable base. It is not
frozen-base fine-tuning. The adapters are discarded after search.

### 3.3 CVaR reward

Every sampled cell is scored on the same panel of held-out training images,
using fixed supernet weights. BN is recalibrated on training images before each
score and restored afterwards. The reward combines three quantities:

```
M(a) = 19-class mIoU from the panel's aggregate confusion matrix
I_i  = per-image IoU (mean over classes present in image i)
T(a) = CVaR_0.2 = mean of the worst 20% of I_i (fractional boundary mass handled)
R(a) = 0.8 · M(a) + 0.2 · T(a)
```

### 3.4 PPO controller

Each architecture is treated as a one-step episode. PPO uses the clipped
likelihood ratio on the joint action log-probability, advantage `R − baseline`,
4 PPO epochs per rollout of 16 cells, and KL early stopping. The schedule has
three phases:

- Warmup epochs sample cells uniformly.
- After warmup, sampling is a 50/50 mix of the uniform distribution and the policy.
- Every 10th minibatch trains the published cell, so that it remains a fair
  reference inside the shared weights.

### 3.5 Data hygiene

Cityscapes train is split into fixed, disjoint subsets: 1,785 images for
weights, 595 for controller reward and 595 for selection. The splits use a
saved fingerprint. **Official val is never seen by search, re-ranking or probes.**
It is used only to evaluate the final retrained models.

### 3.6 The two search variants that produced the results

**A. Global search → F32 cell (75.079%).**
A supernet with F8 and crop 321 was searched with LoRA + CVaR-PPO. Every
visited cell (758 in total) was then re-ranked on the 595 selection images. The
winner scored 34.46% proxy mIoU against 33.83% for the reference. Its cell is:

```
node0: (0,dil3)+(1,skip)   node1: (0,max3)+(1,max3)   node2: (0,skip)+(2,sep3)
node3: (2,sep5)+(3,dil5)   node4: (2,max3)+(1,dil5)
```

**B. Reference-guided local search → candidate 4 (74.285% single-scale / 76.323% TTA).**

1. Build a library of 128 convolution-only cells. Each is the published cell
   or a one- or two-edge change of it, with parameters capped at the reference size.
2. Search with the same LoRA supernet and CVaR reward. A categorical PPO policy
   picks cells from the library.
3. Re-rank all 128 cells on 595 held-out images and shortlist the reference plus
   4 challengers.
4. Train each shortlisted cell independently at F20/crop 769 on 2,380 training
   images, and score it on the 595 held-out training images. Each probe score
   averages the last three evaluations.

| Stage | Ref (ID 0) | ID 1 | ID 2 | ID 3 | **ID 4** |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50k probe (avg 30/40/50k) | 60.60 | 61.48 | 59.85 | 58.23 | 60.75 |
| 150k probe (avg 130/140/150k) | 66.00 | 64.96 | – | – | **66.61** |

5. Candidate 4 beat the reference by +0.61 points at 150k and was retrained
   on all 2,975 images for 500k updates.

Candidate 4 differs from the published cell in exactly **one edge**. In node 3,
the second input changes from `(node0, sep5)` to `(prev-prev cell, sep3)`:

```
published:   (1,dil5)(0,sep3) | (1,sep3)(0,dil3) | (3,sep3)(1,sep3) | (4,sep5)(2,sep5) | (3,sep5)(5,dil5)
candidate 4: (1,dil5)(0,sep3) | (1,sep3)(0,dil3) | (3,sep3)(1,sep3) | (4,sep5)(1,sep3) | (3,sep5)(5,dil5)
```

### 3.7 Final retraining

`train.py` calls `train_baseline.py`, which builds `ReferenceAutoDeepLab` in
`retrain_model/reference_model.py`. Training is from scratch: SGD (momentum 0.9,
weight decay 1e-4), poly LR schedule, AMP, scale/crop/flip augmentation, and
fine labels with void label 255. Validation runs every 5k updates at full
resolution and uses one dataset-wide confusion matrix. `best.pth` is selected by
single-scale val mIoU.

---

## 4. Reproduce

The commands assume a SLURM cluster. Adjust `mypath.py` and the dataset root as needed.

```bash
bash setup_retrain_env.sh                       # private torch 2.3.1+cu118 venv

# B. local search -> shortlist -> probes -> full retrain
SEARCH_SEED=0 sbatch run_search_local.sh
SEARCH_SEED=0 sbatch run_select_local.sh
sbatch run_probe_local.sh                        # 50k probes, IDs 0-4
STOP_AT_STEP=150000 PROBE_SEED=0 sbatch --array=0,1,4%1 run_probe_local.sh
sbatch run_retrain_candidate4_500k.sh            # resubmit until 500k
# matched reference (published cell, same recipe):
RUN_DIR=retrain_reference_f20_500k_seed0_nosdp MAX_STEPS=500000 sbatch run_retrain_baseline.sh \
  --filter-multiplier 20 --seed 0 --bn-momentum 0.01 --drop-path-prob 0 --power 0.9
sbatch run_posteval_candidate4.sh                # FP32 + BN recalibration check
POSTEVAL_MODE=tta sbatch --array=0-1%1 run_posteval_candidate4.sh   # TTA (76.323 / 75.989)

# A. global search -> F32 4000-epoch retrain
SEARCH_SEED=0 sbatch run_search_ppo.sh
sbatch run_select_search.sh --search-dirs <search_dir> --all-visited --rank-images 595 \
  --calibration-batches 32 --output architecture_seed0_halfres_reranked.json
sbatch run_retrain_selected_4000.sh              # resubmit until 740k

# CPU unit tests
OMP_NUM_THREADS=1 python -m unittest -v test_reference_baseline test_search_corrected test_local_workflow
```

The candidate-4 checkpoint `best.pth` (82 MB) is not stored in git. Download it
from this repository's GitHub Releases page. Its `config.json` and
`search_architecture.json` are in `retrain_candidate4_f20_500k_seed0_nosdp/`.
`evaluate.py --checkpoint <best.pth>` restores the architecture from the
checkpoint itself.

---

## 5. Limits of these results

- Each result comes from **one training seed**. A gain of +0.3 points is about
  the size of seed-to-seed variation for Cityscapes models, so it is not
  statistically established.
- Candidate 4 was promoted on a single probe seed. The planned second-seed
  confirmation was skipped for lack of time.
- The CVaR reward did **not** improve the worst-20% image IoU of the final model
  (56.04 vs. 56.41 with TTA).
- No ablation has yet separated the effect of LoRA, of CVaR and of PPO.
  `run_search_ablation.sh` prepares those five arms (random, PPO, LoRA+PPO,
  CVaR+PPO, LoRA+CVaR+PPO).
- TTA scores use a different protocol from single-scale scores. Compare only like with like.
- All scores are on Cityscapes **val**, which was also used to choose the best
  checkpoint. No predictions were submitted to the hidden test server.
