# From-scratch LoRA/CVaR cell search

After the F20 comparison and final BN checks, the next experiment is documented
in [LOCAL_SEARCH.md](LOCAL_SEARCH.md): reference-guided proposals, independent
F20 probes on held-out training images, and a staged promotion rule. The global
workflow below remains available for controls; it is not the recommended next
full retraining selection.

This is an executable research design, **not a demonstrated 80% mIoU method**.
The latest [75% result audit and next experiments](AUDIT_75MIOU.md) document
the LoRA/CVaR comparison with NAO, version-2 fixes and ablation commands.
New searches use `adl_cell_lora_cvar_v2`, fresh `search_lora_cvar_v2_seed*`
directories, and the installed private CUDA 11.8 environment. The launcher now
defaults to 100 epochs / 50 warmup (29,700 weight updates). Do not resume a V1
search under V2; completed V1 supernets remain usable for separate re-ranking.
For the completed seed-0 run, official-budget comparison, and subsequent input
resizing/ranking corrections, see [SEARCH_AUDIT_46703.md](SEARCH_AUDIT_46703.md).
The new `train_autodeeplab.py` entry point calls `train_search_corrected.py`.
It replaces the old dense DARTS/PPO loop with a single-path, discrete-cell search.
Old search checkpoints, alpha/beta decoders and their scripts are incompatible.
The legacy `auto_deeplab.py`, `cell_level_search.py` and `arch_policy.py` are not
used by this workflow; they have not become corrected merely by remaining here.

## Design choice for the available budget

Keep the published 12-layer multiscale path and search the five-node cell's
connections and operations. This reduces search scope and preserves the
resolution/decoder structure of the corrected reference model. It is **cell-only
NAS**, not a reproduction of Auto-DeepLab's joint network/cell search. Searching
the macro path is a separate experiment, not an unimplemented command-line mode.

Each node chooses two distinct inputs and an operation per input. Operations are
separable 3x3/5x5, full atrous 3x3/5x5, skip, and average/max pooling. At most two
skip edges are allowed by default. The final F32 parameter cap is 24M; the
published 21.622M architecture is always included in final comparison. These
constraints apply to every ablation of this search method.

New searches use F8/crop321 after resizing the image short side to 512, with
random crop/flip. Reward views use the same base size. Full retraining uses the same discrete operations and
cell ordering, with F32/crop769 and the corrected decoder. The exported child
has ordinary convolution weights and is trained from scratch on all 2,975
training images. Search weights and adapters are not mistaken for final weights.

## What the audit found

- Old search ran 60 epochs, but PPO began at `alpha_epoch + 50 = 70`: no PPO
  updates could execute. The new warmup is checked against the total duration.
- Old rewards compared losses from different minibatches/weight states and
  associated improvements with the wrong perturbation. The new rollout holds
  weights fixed and evaluates every sampled action on identical cached images.
- Selecting the worst PPO surrogates is not an estimate of the lower tail of
  segmentation returns. That pattern also exists in `Nao_test/PPOArchTrainer.py`;
  it is not copied into the new objective.
- Old search-to-retraining operations, state ordering and architecture defaults
  disagreed. The new supernet uses the corrected retraining operation class and
  exports an explicit validated cell schema.
- The old scheduler overwrote the LoRA LR multiplier. The new scheduler applies
  each group's stored multiplier on every weight update.
- Old training subsets overlapped. New weight/controller/selection subsets are
  disjoint and persisted with an ordered image manifest and fingerprint.

## Where LoRA belongs

The implemented hypothesis is that **input-specific adapters reduce interference
between cell choices sharing a convolution**. Within each layer/node/operation/
stride, the convolution base is shared across possible input states. Each input
has its own low-rank residual and its own BN statistics/affine parameters. This
is related to NAO's per-input adapters, but the inspected NAO code also allocates
separate base tensors per input; the new implementation explicitly shares bases.

- Adapt pointwise convolutions in separable operations and full atrous kernels.
- Leave depthwise kernels, stem, projections, ASPP and decoder without adapters.
  This avoids spending adaptation parameters on every layer indiscriminately.
- Start at rank 4, capped by the layer's input/output dimensions. Initialize one
  factor randomly and the other to zero, so the initial residual is exactly zero.
- Use adapter LR 2x base LR, no adapter weight decay, and no extra multiplication
  by architecture probabilities. Test rank 0/2/4/8 and LR multipliers 1/2 only if
  the budget supports meaningful matched experiments; 4 and 2 are hypotheses,
  not measured optimums.
- Train the base from scratch throughout search. Freezing a randomly initialized
  base would undermine learning. This is LoRA-style parameterization with a
  trainable base, **not a claim of classic frozen-base parameter-efficient tuning**.

I would not initially put LoRA on architecture logits or freeze the entire
backbone after a very short warmup. I would also keep adapters out of the final
from-scratch comparison until the search benefit has been established; otherwise
search changes and final-model changes become difficult to distinguish.

## Where CVaR belongs

For each architecture `a`, use a common panel of labeled images:

```
M(a) = 19-class mIoU from the panel's aggregate confusion matrix
T(a) = mean of the lowest alpha fraction of per-image IoU scores
R(a) = (1 - risk_weight) * M(a) + risk_weight * T(a)
```

Defaults are `alpha=0.2`, `risk_weight=0.2`. Per-image IoU averages classes with
ground-truth pixels in that image, avoiding absent-class zeros dominating the
image score. The panel-wide metric retains all 19 classes. Fractional mass at
the tail boundary is handled explicitly, and at least eight labeled images in
the tail are required. These cropped-image metrics are search proxies, not
official full-resolution Cityscapes validation scores.

PPO optimizes the expected **conditional image-risk score** `R(a)` over sampled
architectures. This is not a claim to implement CVaR over the distribution of
policy trajectories. Each architecture evaluation is a terminal, one-step
episode; there is no GAE chain linking unrelated candidates. Clipping uses the
stored joint action likelihood, ordinary advantages, and KL-based early stopping.
CVaR is calculated from measured image scores before PPO, never from sorted
clipped advantages. Discrete sampled actions are the actions actually evaluated.

The moderate risk weight protects the mean-IoU objective from being dominated
by a noisy tail estimate. Final selection additionally requires mean proxy IoU
at least as high as the reference candidate. This is an empirical guard, not
statistical evidence that final mIoU will improve. Image-tail performance also
does not automatically imply better rare-class IoU; inspect class-level final
validation scores separately.

## Data and candidate ranking

Cityscapes **train** is partitioned deterministically into 60% weight training,
20% controller rewards and 20% final selection. All seeds share `--split-seed 0`.
Official validation images are not loaded by search.

Weight training uses uniformly sampled architectures during warmup, then a
50/50 mixture of uniform and policy sampling. Every tenth weight minibatch trains
the published cell to keep a meaningful reference candidate in the shared model.
This gives that candidate extra coverage and should be disclosed in comparisons.
All ablations use the same coverage rule.

After each post-warmup epoch, 16 sampled candidates are measured on a fresh,
common 64-image controller panel. No weight updates occur within a rollout.
Before scoring each candidate, BN is recalibrated using 48 images from the weight
training split; the original buffers/modes are restored afterward. The same
calibration tensors are used for all candidates, without validation labels or
gradient updates. Finalists are re-ranked on 128 fixed images from the third
split at the same final supernet weights.

Weight-sharing rankings can still be inaccurate. A few short independent child
retraining trials would provide stronger evidence if affordable. With only one
full retraining available, multiple search seeds plus common-panel re-ranking
reduce reliance on one supernet, but cannot remove this uncertainty.

## Run on the GPU

First submit a small, separate end-to-end hardware check. It exercises weight
updates, PPO, BN recalibration, scoring and JSON export:

```bash
cd /scratch/ajin/Autodeeplab/modified
sbatch run_search_ppo.sh --smoke-test
```

Smoke mode uses two tiny epochs in a `_smoke` directory. Its exported architecture
is marked as a smoke artifact and the final retrainer refuses it. Do not interpret
its scores as search quality.

After the AMP fix, a successful smoke test must print `Successful weight updates:
4; PPO updates: 2`. Skipped AMP updates retry the same minibatch with a reduced
scale and cannot advance training progress. The scaler now starts at 1024.
The earlier smoke log `search_corrected_46701.out` completed with zero weight
updates and is not a successful training check. Use a fresh directory for the
replacement smoke run, for example `SEARCH_SEED=0 sbatch run_search_ppo.sh
--smoke-test`. Completed zero-update checkpoints are rejected on resume.

Then start independent real searches (each submission requests an A100):

```bash
SEARCH_SEED=1 sbatch run_search_ppo.sh
SEARCH_SEED=2 sbatch run_search_ppo.sh
SEARCH_SEED=3 sbatch run_search_ppo.sh
```

Each uses `search_lora_cvar_v2_seedN`. Multiple login sessions are fine for submitting
jobs, but training must run in allocated GPU jobs, not on login-node CPUs.
Resubmit the same seed/arguments when the 48-hour allocation ends. Optimizers,
AMP, policy, progress, splits and RNG are saved in `latest.pth`. A partially
completed rollout is discarded and recomputed on restart; do not mix policy
versions within a rollout. Exact bitwise data replay is not promised.
Do not submit two simultaneous jobs to the same directory.

The normal launcher uses 100 epochs with 50 warmup epochs, so it performs 50 PPO
updates. Search time is **additional** to the final retraining budget. There is
no automatic GPU job submission or job chain in the Python code.

Each completed search writes `architecture.json`, `ranking.json`, and
`controller.jsonl`. The export can legitimately select the published cell if
none of the searched candidates passes the mean-IoU guard with a better reward.
That outcome must not be presented as a new discovered architecture.

For stronger selection, `run_select_search.sh` also accepts `--all-visited`,
`--rank-images 595`, and `--calibration-batches 32`. This expands evaluation to
all valid archived candidates and the whole selection subset; it adds GPU
evaluation cost and still produces cropped-image proxy scores.

With multiple completed seeds, compare the union of finalists on the same panel
under every final supernet, rather than comparing their original raw proxy scores:

```bash
sbatch run_select_search.sh --search-dirs \
  search_lora_cvar_v2_seed1 search_lora_cvar_v2_seed2 search_lora_cvar_v2_seed3 \
  --output architecture_consensus.json
```

This performs additional GPU evaluation, not retraining. It selects by average
risk-adjusted proxy score across seeds, with the same reference mean-IoU guard.
It only accepts completed runs of the same recipe with distinct search seeds.

After selection, start the corrected final retraining from scratch:

```bash
ARCHITECTURE_JSON=architecture_consensus.json RUN_DIR=data_searched_f32 \
  sbatch run_retrain_searched.sh --stop-after-steps 200 --val-every 200
# After the short check finishes, continue with the same architecture/directory:
ARCHITECTURE_JSON=architecture_consensus.json RUN_DIR=data_searched_f32 \
  sbatch run_retrain_searched.sh
RUN_DIR=data_searched_f32 sbatch run_test_baseline.sh
```

For one search seed, use its `search_lora_cvar_v2_seed1/architecture.json` instead.
The retraining checkpoint stores the cell, so evaluation restores it automatically.
The generic launcher's final horizon remains 1.5M updates, approximately twice
the completed seven-day schedule (measure throughput for each architecture).
Choose any shorter `MAX_STEPS` budget **before** starting final training
and keep it unchanged on resume. See [BASELINE.md](BASELINE.md).

## Ablations and publication claims

Use separate directories and matched search/retraining/inference budgets:

| Experiment | Extra search flags |
| --- | --- |
| Random cell search, no LoRA | `--random-search --lora-rank 0 --risk-weight 0` |
| Mean-IoU PPO, no LoRA | `--lora-rank 0 --risk-weight 0` |
| Mean-IoU PPO + LoRA | `--lora-rank 4 --risk-weight 0` |
| CVaR-reward PPO, no LoRA | `--lora-rank 0 --risk-weight 0.2` |
| CVaR-reward PPO + LoRA | default |

For example, set `SEARCH_DIR=search_mean_ppo_seed1` when changing flags. Do not
combine different ablations with the cross-seed selector. Proxy ablations alone
are not enough to demonstrate a final segmentation improvement for a paper.
One successful full training can establish feasibility; it cannot establish
seed variance, component attribution, or a guaranteed conference acceptance.

Small CPU regression checks cover disjoint splits, fractional CVaR, action
likelihoods, PPO updates, LoRA initialization/input isolation, optimizer coverage,
sampled model gradients, export compatibility, BN restoration and baseline
compatibility. No GPU search or accuracy experiment was run during implementation.

```bash
OMP_NUM_THREADS=1 python -m unittest -v test_search_corrected test_reference_baseline
```

References informing the design:

- [LoRA](https://arxiv.org/abs/2106.09685): low-rank weight updates; frozen-base
  fine-tuning is different from this trainable-base search parameterization.
- [Optimizing CVaR via Sampling](https://arxiv.org/abs/1404.3862): CVaR policy
  gradients require a defined return distribution; sorting PPO surrogates does
  not implement that estimator.
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347): clipped policy
  likelihood-ratio objective used here for terminal architecture actions.
- Official Auto-DeepLab architecture/operation references are in [BASELINE.md](BASELINE.md).

Only NAO source code was inspected; no CIFAR or ImageNet result search was performed.
