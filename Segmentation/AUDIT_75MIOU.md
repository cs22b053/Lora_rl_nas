# Search and retraining audit after 75.079% validation mIoU

Audited 2026-10-15. No GPU training or evaluation was launched during this audit.
`Nao_test` source was inspected; its CIFAR/ImageNet results were not searched.

## What the completed experiment establishes

`selected_4000_47460.out` finishes at update 740000 / epoch 4000, with best and
final single-scale validation mIoU 75.079%. `selected_eval_47533.out` independently
reproduces 75.079% on all 500 Cityscapes validation images, with AMP, scale 1,
and no flip. The saved evaluation JSON gives 75.07912287736531%.

This is a real improvement over the earlier pipeline, but it does not isolate a
LoRA or CVaR benefit: the architecture, training implementation, preprocessing
and BN treatment changed. Validation also selects the checkpoint, so this is a
validation result, not an independent hidden-test result.

The selected child has 19,481,587 parameters; the published reference cell in
this implementation has 21,622,451. The Auto-DeepLab paper reports 80.04%
single-scale validation for F32, and specifies 1.5 million retraining iterations.
The current run used 740,000 successful optimizer updates, about half that
schedule. Longer training is a plausible experiment, not a guarantee of 80%.
See [the paper, section 5.2 and table 2](https://chenxi116.github.io/papers/auto_deeplab_cvpr19.pdf).

## Training and evaluation checks

The active path is `train.py -> train_baseline.py -> ReferenceAutoDeepLab`,
with `evaluate_baseline.py` restoring architecture and BN configuration from
the checkpoint. The old dense-search modules are not used in this path.

The implementation has the published 12-layer path, explicit cell inputs and
operations, five-node concatenation, third-cell decoder feature, and OS8 ASPP
rates 12/24/36. The reference parameter count agrees with the paper. These
structural checks were compared with the official
[NAS network](https://github.com/tensorflow/models/blob/master/research/deeplab/core/nas_network.py)
and [NAS cell](https://github.com/tensorflow/models/blob/master/research/deeplab/core/nas_cell.py).
This is not a TensorFlow/PyTorch numerical equivalence test: initialization,
resize kernels and BN behavior can still differ.

Optimizer groups cover all trainable parameters; polynomial LR reaches every
group. Batch 16, crop 769, SGD LR .05, momentum .9, weight decay .0001,
1000-update warmup, power .9 and scheduled drop path .1 are coherent. Fine-label
training uses scale/crop/flip, void 255 and [-1,1] normalization. Validation uses
full-resolution images and one dataset-wide confusion matrix, not an average
of minibatch mIoUs. AMP retries do not count as completed training updates.

The successful run uses PyTorch BN momentum .01 following the pilot diagnosis.
The model default .0003 approximates the official decay .9997; these are
different recipes. The matched reference launcher explicitly uses .01.
An older checkpoint crossed PyTorch/CUDA versions during training, so a fresh
control in the current environment is recipe-matched, not a bitwise replay.

No new training defect found in this audit explains the entire remaining
five-point gap. The tests establish specific invariants, not guaranteed model
quality. Avoid changing LR, crop, loss and architecture together in the next run.

## LoRA: coherent, but different from canonical LoRA and NAO

For a shared operation and selected input s, the effective weight is
`W_s = W_base + B_s @ A_s`, with scale 1 (alpha equals effective rank).
The base is trainable from scratch. A is random, B initially zero, so each
adapter begins with zero residual. Rank defaults to 4 and the adapter LR is 2x
the base LR without adapter weight decay.

Adapters specialize dense atrous and pointwise convolution weights by input.
Depthwise weights, stem, projections, ASPP and decoder have no adapters.
Input-specific BN remains present in rank-zero controls as well. The intended
hypothesis is reduced interference between choices sharing an operation.

This is **LoRA-style supernet parameterization with trainable shared bases**,
not frozen-pretrained-base parameter-efficient fine-tuning as in
[canonical LoRA](https://arxiv.org/abs/2106.09685). It is used only during search.
The exported child is an ordinary network trained from scratch, without LoRA
or PPO. Therefore any final benefit must come through architecture selection.

`Nao_test/operations.py` also gives each input its own base tensors and adapts
depthwise as well as pointwise convolutions. Despite its comments about shared
bases, those `ParameterList` entries are distinct parameters. The current
Auto-DeepLab implementation shares bases across inputs and is not an exact copy.

### Fixed: inactive adapter momentum

The adapters are stacked tensors. Selecting one input leaves other slices with
zero dense gradients, but SGD momentum can still change those inactive slices.
A tiny CPU reproduction confirmed this movement. Search now snapshots/restores
inactive slices and their momentum, while retaining updates to every input used
by the current architecture. This matches the intended input-specific behavior.
The helper explicitly supports SGD, the optimizer used here.

## CVaR and PPO: the objective must be named accurately

For an architecture a on a fixed panel:

```
M(a) = 19-class mIoU from the aggregate confusion matrix
I_i(a) = image-i mean IoU over ground-truth-present classes, excluding void
T(a) = mean of the lowest 20% of I_i(a), including fractional boundary mass
R(a) = 0.8 * M(a) + 0.2 * T(a)
```

This is empirical **lower-tail CVaR over image scores conditional on an
architecture**. PPO maximizes expected R over sampled architectures. It is not
CVaR over policy trajectories, and it does not specifically optimize rare
classes. False positives for classes absent from an image are excluded from
that image's tail score; the aggregate metric still penalizes them.

Every evaluated architecture is a complete one-step episode. The code uses
measured segmentation scores, joint action likelihoods with the same masks at
sampling/update time, advantages `reward - value`, clipping and KL stopping.
This is a coherent application of [PPO](https://arxiv.org/abs/1707.06347).
There is no GAE chain between independent architecture samples.

In contrast, `Nao_test/PPOArchTrainer.py` uses predictor-improvement rewards for
continuous embedding actions, joins independently sampled candidates in GAE
with nonterminal flags, and selects the tail of PPO surrogate values. That
surrogate-tail operation is not an estimate of the image-return CVaR above.
The current implementation should not copy those semantics merely to resemble
NAO. State the exact objective in any paper; general background is
[Optimizing the CVaR via Sampling](https://arxiv.org/abs/1404.3862).

## Search correctness and remaining uncertainty

The 2975 training images are split into 1785 weight / 595 controller / 595
selection images with persisted split/fingerprint. Official validation is not
used by search. Candidates in each rollout see identical held-out images and
identical train-only BN calibration tensors, at fixed supernet weights. Scoring
restores the original buffers and modes afterward. Final selection includes
the published cell and requires mean proxy IoU at least as high as that cell.

The search holds the macro path fixed and searches cells. It is not a
reproduction of the paper's joint differentiable macro/cell search. The current
launcher uses 100 epochs, 50 warmup, batch 6: 29,700 weight updates and 50 PPO
updates. Similar weight-update counts do not make different search algorithms
or FLOP budgets equivalent. Uniform warmup, a later 50/50 uniform/policy mixture,
and a reference-cell anchor every tenth weight minibatch are shared by all arms.

The selected cell beat the reference in the held-out shared-weight proxy
(34.455% vs 33.831%), not yet in a matched full retraining comparison. Search
uses F8/crop321; final training uses F32/crop769. Shared-weight rank bias,
capacity transfer and the noisy crop-level tail remain experimental risks.
Search weight preprocessing uses PIL resize while fixed panels use tensor
interpolation; this is another proxy difference, not proof of a broken metric.

### Other fixes and compatibility

- AMP overflow retries now roll back BN buffers before retrying the same search
  minibatch, so skipped steps do not update statistics multiple times.
- Nonfinite candidate logits fail explicitly before argmax/scoring.
- Search resume uses the validated CUDA RNG compatibility helper.
- Search, re-ranking, reference training and BN-diagnostic launchers use the
  already working private CUDA 11.8 environment, without GPU-node downloads.
- New searches are versioned `adl_cell_lora_cvar_v2` and use fresh output paths.
  V1 searches cannot resume under the changed training semantics. Completed V1
  supernets can still be re-ranked because tensor layouts are unchanged, but
  V1/V2 runs cannot be combined. The completed 75% child remains valid evidence
  for the V1 experiment; these fixes do not retroactively change its score.
- Evaluation now reports image CVaR and individual image scores, in addition to
  the unchanged aggregate metric. New result paths include `_cvar_0.2`, keeping
  the original result file separate. Re-run evaluation to obtain these metrics;
  they cannot be recovered from the aggregate confusion matrix alone.

## Recommended next jobs

Assuming approximately seven days per full run, first compare the published
reference cell with the selected child under the successful training recipe.
This is more informative than another blind search-and-retrain cycle.

```bash
cd /scratch/ajin/Autodeeplab/modified
sbatch run_diagnose_selected_final.sh
sbatch run_retrain_reference_4000.sh
```

The diagnostic measures stored vs train-only recalibrated BN in FP32, with
frozen weights, and writes `retrain_selected_seed0_4000_bn001/bn_final_diagnostic.json`.
It does not save a replacement checkpoint. Compare its two FP32 measurements
with each other before attributing any difference to BN. It can run independently
of reference training if another GPU is available.

The reference run writes `retrain_reference_seed0_4000_bn001`, uses seed 0,
BN .01 and 740000 updates. It is larger than the selected model, so elapsed time
may differ. Resubmit the **same command** after each 48-hour allocation to resume.
After completion, evaluate it using:

```bash
RUN_DIR=retrain_reference_seed0_4000_bn001 sbatch run_test_baseline.sh
```

Obtain the selected model's new tail metric, and optionally a separate TTA score:

```bash
sbatch run_test_selected_4000.sh
sbatch run_test_selected_4000.sh --eval_scales 0.5 0.75 1.0 1.25 1.5 --eval-flip
```

Do not compare TTA scores to a single-scale paper result as if protocols matched.
If the reference matches/beats the selected cell, prioritize recipe reproduction
and proxy ranking. If the selected cell wins, that supports its architecture
under this budget, but a single seed still does not isolate LoRA/CVaR causality.

If roughly 14 GPU-days **per model** are available instead, start a fresh 1.5M
schedule for each of the selected and reference architectures. For example:

```bash
RUN_DIR=retrain_reference_seed0_1500k_bn001 MAX_STEPS=1500000 \
  sbatch run_retrain_reference_4000.sh
RUN_DIR=retrain_selected_seed0_1500k_bn001 MAX_STEPS=1500000 \
  sbatch run_retrain_baseline.sh --seed 0 --bn-momentum 0.01 \
  --architecture-json architecture_seed0_halfres_reranked.json
```

These are alternatives to the seven-day plan, not required additional jobs.
Do not extend the completed 740k polynomial schedule by changing its horizon
on resume. A fresh 1.5M schedule tests a different LR history from its first step.

## Search ablations for the research claim

`run_search_ablation.sh` prepares five version-2 arms with common splits,
candidate constraints, weight-update count and evaluation panels:

| Array index | Arm | Rank | Tail reward weight |
| --- | --- | --- | --- |
| 0 | Random search | 0 | 0 |
| 1 | PPO | 0 | 0 |
| 2 | LoRA + PPO | 4 | 0 |
| 3 | CVaR + PPO | 0 | .2 |
| 4 | LoRA + CVaR + PPO | 4 | .2 |

Start with seed 0. One GPU at a time:

```bash
SEARCH_SEED=0 sbatch --array=0-4%1 run_search_ablation.sh
```

To run only the corrected full-method search initially:

```bash
SEARCH_SEED=0 sbatch --array=4 run_search_ablation.sh
```

Resubmit unfinished indices with the same seed/settings. Repeat seeds 1 and 2
when budget allows. This launcher submits searches only; it does not train child
networks. Equal update counts are not equal GPU-hours: record actual search
time and resource use for each arm. LoRA increases supernet parameters/compute.

Re-rank each completed arm separately using the same protocol. Example for
seed-0 LoRA+CVaR+PPO:

```bash
sbatch run_select_search.sh \
  --search-dirs search_ablation_v2_lora_cvar_ppo_seed0 \
  --all-visited --rank-images 595 --calibration-batches 32 \
  --output architecture_v2_lora_cvar_ppo_seed0.json
```

Do not pool different arms in a consensus selection. For variability estimates,
retain per-seed selections and results; multi-seed consensus is a separate,
higher-cost selection method. Compare independently retrained children with
the same final training budgets/seeds, and report mean/class IoU, image CVaR,
parameters, search GPU-hours and variability. Rank-zero controls retain
input-specific BN, isolating adapter weights rather than changing both at once.

## Verification performed

All 22 lightweight CPU regression tests passed (`test_reference_baseline` and
`test_search_corrected`), including LoRA momentum isolation, zero-residual
initialization, PPO updates/masks, fractional CVaR, disjoint splits, nonfinite
score rejection, AMP retry/BN restoration, model/checkpoint behavior and
batch-independent image-tail metrics. Modified/new shell launchers passed
`bash -n`. No full search, training, CUDA benchmark or validation was run.
