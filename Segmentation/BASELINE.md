# Corrected Auto-DeepLab baseline

The selected cell has now reached independently reproduced 75.079% validation
mIoU. See [AUDIT_75MIOU.md](AUDIT_75MIOU.md) for the current audit and matched
reference control, and [RETRAIN_4000.md](RETRAIN_4000.md) for the pilot history.
`run_retrain_reference_4000.sh` uses the successful BN .01 / 740k-update recipe;
the generic launcher below retains its BN .0003 / 1.5M-update defaults.

The active `train.py` now trains the **published Auto-DeepLab architecture from
scratch**, using `retrain_model/reference_model.py`. It does not load the old PPO
genotype or the 68% checkpoint. This is a baseline reproduction attempt, not a
measured 80% result and not evidence of a LoRA/CVaR improvement.

## What changed

- The official 12-cell path starts at output stride 4 and finishes at stride 8.
  Cell branches explicitly pair each operation with its input. The official
  current/previous-state ordering is respected.
- Atrous cell operations use full convolutions. Reductions use the reference's
  stride-dependent separable/atrous behavior instead of shrinking all inputs
  before every operation. Stem and cell activations are explicit.
- The decoder uses the third cell at stride 4, not the stem. ASPP uses rates
  12/24/36 for stride 8. F32 has **21,622,451 parameters**, consistent with the
  paper's 21.62M model. Matching parameter count is a sanity check, not proof of
  numerical equivalence or reproduced accuracy.
- BatchNorm has affine parameters and explicit epsilon/momentum. Backbone
  epsilon is 0.001, head epsilon is 0.00001, and PyTorch momentum is 0.0003.
  Scheduled drop path increases with both cell depth and training progress.
- SGD now covers every trainable parameter exactly once. Both optimizer groups
  receive the polynomial LR schedule; convolution weights get weight decay,
  while normalization parameters and biases do not. Previously the scheduled
  group was empty and the group containing the weights stayed at LR 0.05.
- Fine-label Cityscapes training uses scale/crop/flip, void label 255, and input
  normalization to [-1, 1]. Validation uses the complete original resolution.
  The baseline does not add the previous strong colour jitter/blur or OHEM.
- Full validation selects `best.pth`; `latest.pth` includes optimizer, AMP,
  schedule progress and RNG state. Checkpoints record model and training config
  and reject legacy weights. Periodic snapshots use update numbers.
- Evaluation restores the checkpoint's architecture, uses evaluation mode,
  aggregates dataset-wide confusion counts, and records scales/flip/precision.
  Multi-scale inference averages class probabilities. Separate result files
  identify the protocol.

The original model modules remain available for evaluating old checkpoints.
The new baseline intentionally has a separate model format. Search, LoRA and
PPO/CVaR modules from the old implementation remain legacy code. The corrected
cell-search workflow is now documented separately in [SEARCH.md](SEARCH.md).
Its accuracy benefit remains experimentally unverified.

## Submit

From `/scratch/ajin/Autodeeplab/modified`, first check the GPU, data loading,
memory, loss, checkpoint and full validation path with a short run:

```bash
sbatch run_retrain_baseline.sh --stop-after-steps 200 --val-every 200
```

This uses the full training LR horizon, saves into `data_reference_f32`, and can
be continued. Early validation is only a functionality check: 200 updates are
far too early to assess final quality, especially with slow BN running averages.
After it finishes, continue with:

```bash
sbatch run_retrain_baseline.sh
```

The SLURM allocation is 48 hours. A pre-timeout signal requests a checkpoint;
`latest.pth` is also saved every 1,000 updates and before validation. **Resubmit
the same command after each allocation finishes** to resume. No automatic job
chain is submitted by this code. Do not run two jobs against the same directory.
An abrupt kill can lose updates since the last periodic save. Resume restores
training state, but bitwise-identical replay is not promised with AMP/cuDNN or
different data-worker settings.

The default schedule is **1,500,000 optimizer updates**, crop 769, physical batch
16, F32, initial LR 0.05, 1,000 warmup updates, polynomial power 0.9, weight decay
0.0001 and maximum drop-path probability 0.1. This is a reference-informed
PyTorch recipe, **not a numerically verified TensorFlow port**. Pillow resizing,
PyTorch BN/convolution details, AMP and warmup can differ from the reference.

The corrected model has different compute/memory costs from the old one. Do
not assume it finishes in the previous 20 days. Logs report seconds/update and
estimated days remaining; estimates become more useful after periodic validation.
If you must keep a shorter update budget, set it **before the first submission**:

```bash
MAX_STEPS=740000 RUN_DIR=data_reference_budget sbatch run_retrain_baseline.sh --stop-after-steps 200 --val-every 200
MAX_STEPS=740000 RUN_DIR=data_reference_budget sbatch run_retrain_baseline.sh
```

740,000 updates roughly matches the previous 4,000-epoch update count at batch
16, **not its wall time**, and is shorter than the paper's schedule. The 80% target
is unverified for either recipe here. Do not change the horizon halfway through:
it changes both LR and drop-path schedules; resume deliberately rejects this.

If physical batch 16 is out of memory, start in a fresh directory with
`--batch-size 8 --accumulation-steps 2`. Keep these arguments on subsequent
submissions. This preserves effective SGD batch 16, but BN sees only 8 samples;
it is a recipe change, not an exact substitute. Physical batch 1 is rejected
because the global ASPP branch contains BN. Use `--dataset-root /path/to/cityscapes`
if needed; the current repository dataset path remains the default.

## Evaluate

```bash
sbatch run_test_baseline.sh
sbatch run_test_baseline.sh --checkpoint data_reference_f32/latest.pth
# Optional, report separately from the single-scale result:
sbatch run_test_baseline.sh --eval_scales 0.5 0.75 1.0 1.25 1.5 --eval-flip
# Optional full-precision verification:
sbatch run_test_baseline.sh --eval-fp32
```

Training selects `best.pth` by single-scale validation using AMP. Evaluation
defaults to the same precision and protocol. No validation labels participate
in gradient updates. A missing best checkpoint means validation has not yet
completed; `latest.pth` can still be evaluated.

`run_retrain_ppo.sh` now forwards to the corrected baseline and prints this fact.
`run_test_ppo.sh` retains old-checkpoint evaluation by default; supplying
`--checkpoint corrected/path.pth` forwards to the corrected evaluator. Legacy
recovery is still a separate workflow for the legacy model.

## Checks and research interpretation

Lightweight CPU checks are in `test_reference_baseline.py`: model capacity,
odd/even output shapes, finite gradients, optimizer coverage/LR, label mapping,
augmentation/ignore labels, global IoU, and checkpoint compatibility. No GPU
training, full-resolution GPU memory check or accuracy reproduction was run.

```bash
OMP_NUM_THREADS=1 python -m unittest -v test_reference_baseline
```

Establish this baseline before judging the method. To claim a LoRA/CVaR benefit,
subsequent experiments need a corrected search implementation and matched
baseline/method training and evaluation budgets. Beating an older baseline by
changing its architecture, training recipe or inference protocol alone would
not establish the proposed method's contribution.

Reference sources:

- [Auto-DeepLab paper](https://chenxi116.github.io/papers/auto_deeplab_cvpr19.pdf)
- [Official architecture](https://github.com/tensorflow/models/blob/master/research/deeplab/core/nas_network.py)
- [Official cell operations](https://github.com/tensorflow/models/blob/master/research/deeplab/core/nas_cell.py)
- [Decoder endpoints and normalization](https://github.com/tensorflow/models/blob/master/research/deeplab/core/feature_extractor.py)
