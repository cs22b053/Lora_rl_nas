# Pilot audit and 4,000-epoch retraining

For the audit after the independently reproduced 75.079% result, new matched
reference launcher, final BN diagnostic and controlled search experiments, see
[AUDIT_75MIOU.md](AUDIT_75MIOU.md).

## Completed training and evaluation

Job 47460 completed update 740000 (4000 epochs). The best and final validation
score is 75.079% at update 740000, using single-scale AMP inference on all 500
Cityscapes validation images. The 80% target has not been reached by this run.

```bash
cd /scratch/ajin/Autodeeplab/modified
sbatch run_test_selected_4000.sh
```

This evaluates `retrain_selected_seed0_4000_bn001/best.pth` with its saved
architecture and BN settings, using the working private CUDA 11.8 environment.
No optimizer or RNG state restore is needed for evaluation. Logs are
`selected_eval_<jobid>.out/.err`; per-class IoU, aggregate mIoU and confusion
counts were written under `log_selected_4000_eval/scales_1_flip_0_amp_1/`.
New evaluations also include worst-20%-image IoU and individual image scores,
and write under `log_selected_4000_eval/scales_1_flip_0_amp_1_cvar_0.2/`.
Small numerical differences from the training-time validation are possible.

Optional multi-scale and horizontal-flip evaluation:

```bash
sbatch run_test_selected_4000.sh --eval_scales 0.5 0.75 1.0 1.25 1.5 --eval-flip
```

That is a separate, slower inference protocol and writes a separate results
subdirectory. Report its score separately from single-scale validation. Neither
command submits predictions to the Cityscapes hidden-test evaluation server.

## October 9: legacy random-state resume compatibility

Job 47330 passed the CUDA 11.8 GPU check on dgx1 (A100, driver 470.161.03).
It then failed with `RNG state is wrong size` while restoring the saved CUDA
random-number generator, after loading model/optimizer/scaler state.
The checkpoint remains at update 252064; this failed job did not overwrite it.

The saved RNG tensor has 816 bytes: 800 bytes of 0xff legacy padding followed
by the 8-byte Philox seed and 8-byte offset. PyTorch 2.3.1 expects just the last
16 bytes. The formats are documented by the implementation in
[PyTorch 2.0.1](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp)
and [PyTorch 2.3.1](https://github.com/pytorch/pytorch/blob/v2.3.1/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp).
The exact previous PyTorch version was not recorded; the saved layout matches
that legacy implementation.

Resume now validates that known layout and converts it in memory, preserving
the seed and offset. Unknown sizes, invalid padding/offsets or changed CUDA
device counts fail explicitly. There is no RNG reseeding or optimizer reset.
New checkpoints contain native-format RNG state, runtime versions and a record
of the conversion. Unchanged-format resumes continue normally. Exact numerical
replay across PyTorch/CUDA versions is still not promised.

The private environment is now installed and CUDA-tested. No reinstallation
is needed for this error. Submit `sbatch run_retrain_selected_4000.sh` again;
look for the conversion message followed by resume at update 252064.
CPU regression tests and a read-only conversion check on the real checkpoint
cover this fix; the assistant did not launch GPU training or rewrite checkpoints.

## October 8: CUDA failure when resuming

Job 47166 exited cleanly at 47h54m34s, shortly before its 48-hour limit, and
saved update 252064. Job 47303 failed during CUDA initialization, before loading
the checkpoint. Both jobs ran on dgx1; it is the only node listed in longq.
The latest model and optimizer tensors are finite. Best validation so far is
69.282% at 185000; the latest validation is 65.688% at 250000.

The shared pytorch_gpu1 environment currently contains PyTorch 2.3.1+cu121
(CUDA 12.1), while the failure reports CUDA driver API version 11040 (11.4).
PyTorch package files have October 7 timestamps, suggesting an environment
change while the older process was still running. The previous exact package
versions were not recorded, so the update history is not conclusively known.

Use the prepared private environment instead of altering the shared install:

```bash
cd /scratch/ajin/Autodeeplab/modified
bash setup_retrain_env.sh
sbatch run_retrain_selected_4000.sh
```

Setup downloads PyTorch 2.3.1+cu118, NumPy 1.26.4 and Pillow 10.4.0 into
`.venvs/retrain_cu118`, and records installed dependencies there. Run it once on
the login node with package-download access; it does not train or submit jobs.
Setup disables inherited pip index configuration within its own process, uses
PyPI plus the official PyTorch CUDA 11.8 index, and requires binary wheels. It
pins typing_extensions 4.12.2 to avoid the observed source-package/flit-core
build failure and the unreachable inherited NVIDIA NGC index. A failed setup
can be rerun in the same venv. Submit only after it prints `Setup complete`:

```bash
bash setup_retrain_env.sh && sbatch run_retrain_selected_4000.sh
```

The launcher requires a completion marker before requesting CUDA execution.
GPU nodes need no internet: all runtime packages are installed on shared scratch
by the login-node setup. This does not install a system CUDA toolkit or driver.
The launcher uses this private Python, checks GPU initialization and a tiny AMP
convolution/backward pass inside the allocation, then resumes the existing
252064-update checkpoint. No model, optimizer or schedule reset is requested.

The CUDA 11.8 build is provided by [PyTorch](https://pytorch.org/get-started/previous-versions/).
CUDA 11.x supports older drivers through
[minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
subject to feature limitations. Actual operation on dgx1 still needs the job's
startup check. If it fails, provide that output and nvidia-smi driver information
to the cluster administrator. Loading a toolkit module alone does not replace
PyTorch's compiled CUDA runtime or repair the kernel driver.

The assistant checked checkpoint integrity, Python/shell syntax and scheduler
history. It did not download this new environment or run GPU code. The existing
pilot discussion below is retained as experiment history.

The completed 20,000-update pilots are not a clean validation of the training
recipe. Reference validation mIoU at 5k/10k/15k/20k was
4.338/1.982/1.982/22.906%; selected-cell mIoU was
4.050/2.202/14.699/45.695%. Both had low late training loss and finite saved
weights and buffers. The selected model currently looks stronger, but the
reference's collapse means this comparison does not establish a reliable NAS gain.

## Do this first: diagnose the saved pilots

```bash
cd /scratch/ajin/Autodeeplab/modified
sbatch run_diagnose_pilots.sh
```

This submits two GPU tasks, one per pilot. To run them sequentially instead,
use `sbatch --array=0-1%1 run_diagnose_pilots.sh`. Each task compares full-resolution
FP32 validation with stored BatchNorm (BN) statistics against FP32 validation
after train-only BN recalibration. Calibration uses 128 batches of 16 randomly
augmented training images, frozen weights, and disabled dropout/drop path.
The same calibration seed is used for both pilots. Validation images and labels
are never used to update BN. Original checkpoints are not overwritten.

Outputs: `pilot_bn_<jobid>_0.out`, `pilot_bn_<jobid>_1.out`, and
`bn_diagnostic.json` in each pilot directory. A report is finished only when its
`complete` field is true. The diagnostic runs inference only, not another pilot
optimization. It also separates an FP16 evaluation problem from a BN problem
by measuring stored statistics in FP32 first.

BN momentum 0.0003 is the correct conversion of TensorFlow decay 0.9997,
not a typo. However, its exponential averaging window is roughly 3,333 batches;
in a deep network, stale statistics can substantially distort inference when
weights change. This is a hypothesis for these results, not a confirmed cause.
See [PyTorch BN semantics](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
and the [reference NAS settings](https://github.com/tensorflow/models/blob/master/research/deeplab/core/nas_network.py).

If calibration substantially improves both models, compare their calibrated
scores before choosing the final cell. If stored-stat FP32 alone improves the
score, investigate evaluation precision instead. If neither helps, investigate
the model/data/training mismatch before spending the long-run budget.
Do not treat a calibration improvement as proof that any new training recipe
will reach 80%.

## Prepared long-run recipe

`run_retrain_selected_4000.sh` uses `architecture_seed0_halfres_reranked.json`,
F32 (19,481,587 parameters), physical batch 16, crop 769, seed 0, fine training
labels, SGD LR 0.05, weight decay 0.0001, 1,000-update warmup, polynomial power
0.9, and scheduled drop path 0.1. It starts from scratch in
`retrain_selected_seed0_4000_bn001`; it never resumes either short pilot.

The proposed new recipe uses BN momentum **0.01** (roughly a 100-batch averaging
window), saved in the model/checkpoint config and checked on resume. This is an
explicit change from the reference-informed recipe, intended to reduce running
statistics lag; its accuracy benefit is not yet measured. Old checkpoint and
search behavior remain unchanged. Confirm the diagnostic before choosing this
recipe for the long run.

With 2,975 train images and drop_last=True:

```
batches/epoch = floor(2975 / 16) = 185
successful updates = 4000 * 185 = 740000
```

Both LR and drop path use that horizon. This is not the paper's 1.5-million-update
schedule. `--epochs` derives the budget from the actual dataset; it cannot be
combined with `--max-steps`. Accumulation must divide the number of batches per
epoch for an epoch-based run. AMP overflows retry the same microbatches and
restore buffers before retrying, so they do not consume extra data epochs.

After the diagnostic supports proceeding, first check the new recipe at 5k:

```bash
sbatch run_retrain_selected_4000.sh --stop-after-steps 5000
```

This uses the full 740k-update schedule, not a compressed pilot schedule.
If loss, validation and runtime are healthy, continue that same run:

```bash
sbatch run_retrain_selected_4000.sh
```

Each allocation lasts at most 48 hours. Resubmit the same command after it ends;
there is no automatic job chain. Do not run two jobs on the same output directory.
Checkpoints retain optimizer/scaler/RNG and data progress, with latest every
1k updates, full 500-image validation every 5k, best validation checkpoint,
and snapshots every 50k. Keep the schedule and BN settings fixed on resume.

The selected pilot reported about 0.74 seconds/update near completion: 740k
updates at that rate is about 6.3 days before differences in validation/checkpoint
overhead, GPU availability and the revised loop. Use the new run's measured
throughput; the previous 20-day estimate does not transfer automatically.

If the diagnostic favors the reference instead, use a fresh RUN_DIR and the
reference script with MAX_STEPS=740000, --seed 0 and --bn-momentum 0.01, provided
the BN diagnosis supports that setting. Matching pilot comparisons with the
revised recipe may still be needed if the diagnostic is ambiguous.

Only small CPU checks have been run while preparing this code. GPU diagnosis,
training behavior and final accuracy remain to be measured by your submissions.
