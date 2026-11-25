# Candidate 4: completed 500k comparison

Checked November 10, 2026, from each run's `validation.jsonl` and `config.json`.
Both runs use F20, seed 0, 500k successful optimizer updates, all 2975 training
images, batch 16, crop 769, initial LR .05, warmup 1000, polynomial power .9,
BN momentum .01 and no scheduled drop path. These logged scores use stored BN,
single-scale/no-flip AMP evaluation on all 500 official validation images.

| Measurement | Published-cell reproduction | Candidate 4 | Delta (points) |
| --- | ---: | ---: | ---: |
| Best validation checkpoint mIoU | 73.942247 (475k) | 74.284914 (490k) | +0.342667 |
| Final 500k checkpoint mIoU | 73.116622 | 74.132372 | +1.015750 |
| Worst-20% image IoU mean, at best-mIoU checkpoint | 55.365474 | 54.997605 | -0.367869 |
| Worst-20% image IoU mean, final checkpoint | 55.868630 | 54.879324 | -0.989305 |
| Parameters | 10,147,427 | 10,121,187 | -26,240 |

The tail metric is the mean of the lowest 20% per-image IoUs. Each image IoU
averages over classes present in that image's ground truth, ignoring void.
It is different from dataset-wide mIoU. Report both for a CVaR-based method.

This is a reportable single-seed improvement over this reproduction. It does
not establish statistical significance, a repeatable search-method advantage,
or an improvement caused by LoRA or CVaR. The tail results currently do not
support improved worst-image performance. Independent search/retraining seeds,
matched-budget random-search controls and component ablations would be needed
to substantiate those claims. Repeated checkpoints of one run are not seeds.

The paper's Table 3 reports 75.20 for Auto-DeepLab-S at 500k without SDP, in its
single-scale Cityscapes validation experiments. Candidate 4 is 0.915086 points
below that value at its best checkpoint. Keep the published number and the
reproduction as separate rows. Source:
https://chenxi116.github.io/papers/auto_deeplab_cvpr19.pdf

## Next: evaluate the saved models without optimization

```bash
cd /scratch/ajin/Autodeeplab/modified
sbatch run_posteval_candidate4.sh
```

The array runs four jobs sequentially on one GPU at a time:

| Array task | Architecture | Checkpoint |
| --- | --- | --- |
| 0 | Reference | `best.pth`, 475k |
| 1 | Candidate 4 | `best.pth`, 490k |
| 2 | Reference | `step_500000.pth` |
| 3 | Candidate 4 | `step_500000.pth` |

Each job evaluates stored BN and then temporarily recalibrates BN using 128
batches of 16 **training** images with the same calibration seed and recipe.
Both evaluations are FP32, full-resolution, single-scale, no-flip. Calibration
does not use validation images or labels. No optimizer runs; original checkpoint
files are not modified. The saved reports retain both outcomes even if
calibration hurts. Outputs: `posteval_candidate4_500k/bn/<label>.json`.

The previously completed reference-best diagnostic measured 73.943427 with
stored BN and 73.908546 after the same 128-batch calibration. Therefore a large
gain from this check is not assured. FP32 verifies numerical consistency; it is
not an accuracy-improvement promise.

For a separate inference-augmentation experiment on both existing best-mIoU
checkpoints:

```bash
POSTEVAL_MODE=tta sbatch --array=0-1%1 run_posteval_candidate4.sh
```

This uses stored BN, AMP, scales 0.5/0.75/1/1.25/1.5 and horizontal flips:
ten forward passes per validation image, with no gradient updates. Both models
receive the same protocol. Outputs are under
`posteval_candidate4_500k/tta/<label>/scales_0.5_0.75_1_1.25_1.5_flip_1_amp_1_cvar_0.2/`.
TTA can help or hurt. A score above 75.20 under this protocol is not a matched
single-scale win over the paper. Report it separately with inference cost.
Use `--array=0-3%1` instead if also evaluating both final checkpoints with TTA.

Do not pick the better BN/TTA treatment independently for each model and then
present the comparison as one shared protocol. Disclose all evaluated settings
and checkpoint-selection rules. Continued fine-tuning would be extra training,
even if initialized from these checkpoints, and must be accounted for separately.

Commands can be inspected without GPU work with
`SLURM_ARRAY_TASK_ID=1 bash run_posteval_candidate4.sh --dry-run`.
