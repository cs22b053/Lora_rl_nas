# Audit of search job 46703

## Completion and numerical checks

The full seed-0 run completed and exported `search_lora_cvar_seed0/architecture.json`.
Both the log and `latest.pth` report epoch 60, 17,820 successful weight updates,
and 40 PPO rollout updates. The policy optimizer records 160 Adam steps (four
per rollout). The error log is empty. All saved floating-point model and policy
tensors are finite, and the weight optimizer contains momentum state.

The 1,785 / 595 / 595 image subsets are disjoint. Final group learning rates are
approximately 0.001 for ordinary weights/BN and 0.002 for LoRA. The controller
history contains 640 actions, including 45 over-budget rejections and 595 unique
valid cells. Rejections logged as zero mIoU are not measured failed predictions;
those cells receive reward -1 without an evaluation.

These checks support successful execution of this implementation. They do not
establish that its ranking predicts independent retraining performance.

## Comparison with the publication

| Setting | Published search | This completed run |
| --- | --- | --- |
| Search scope | Joint cell/network | Cell only, fixed published network path |
| Epochs / batch | 40 / 2 | 60 / 6 |
| Train split | Half weights, half architecture | 60% weights, 20% controller, 20% selection |
| Architecture start | After epoch 20 | After epoch 20 |
| Weight updates | Approximately 29.7K, derived | 17,820, measured |
| Warmup weight updates | Approximately 14.9K, derived | 5,940 |
| Crop / filter multiplier | 321 / 8 | 321 / 8 |
| Architecture optimizer | Adam on alpha/beta | Discrete PPO |

Source: [paper, sections 4.2 and 5.1](https://chenxi116.github.io/papers/auto_deeplab_cvpr19.pdf).
The approximate counts assume one pass over half of 2,975 images per epoch;
rounding and treatment of incomplete batches prevent assigning an exact count
without the original search loop. This is a paper-budget comparison, not a claim
to have executed and matched that loop. The [public TensorFlow README](https://github.com/tensorflow/models/tree/master/research/deeplab)
documents the found HNASNet backbone and training/evaluation tools.

The current run has about 60% as many optimizer updates but more image
presentations, because it uses a larger batch and training subset. Dense mixed
operation search and single-path search also update different numbers of
operations per step. Matching the update count cannot make these algorithms
equivalent. The retained reference decoder/ASPP and discrete skip/parameter
constraints are further method differences.

## Quality of the current selection

The selected cell has four pooling edges, one skip and five convolution edges.
Its final F32 model has 9,576,435 parameters, versus 21,622,451 for the reference.
F32 therefore does not imply equal model capacity between these cells.

On 128 cropped held-out training images:

- Selected aggregate proxy mIoU: 21.2081%.
- Reference aggregate proxy mIoU: 20.5088%.
- Difference: +0.6993 percentage points.
- Best versus second-best aggregate proxy difference: about 0.1454 points.
- Selected image-tail score: 16.4945%; reference: 15.7462%.

An exploratory paired image bootstrap (2,000 resamples, seed 0) gives a
per-image mean-IoU difference of +0.443 points and a 95% interval of approximately
[-0.418, +1.335]. This is NOT a confidence interval for aggregate dataset mIoU,
and it does not account for selecting the winner from several candidates. It
illustrates why the small proxy margin should not be treated as reliable proof
of a better fully trained architecture.

## Issues corrected after this audit

1. Search weight crops previously used the retraining transform: random scaling
   of original-resolution images. Reward crops used approximately half-resolution
   views. New search runs default to `--train-short-side 512`, with fixed resizing
   before random crop/flip, and use the same base size for reward views. Final
   retraining augmentation is unchanged. Old checkpoints require explicit
   `--train-short-side 0` to preserve their recipe; changing preprocessing on
   resume is rejected. Existing checkpoints and results were not overwritten.
2. Final selection considered only 16 cells, despite the larger visited pool.
   `select_search_architecture.py --all-visited` now re-evaluates all valid visited
   cells, all original finalists and the reference at the same saved weights.
   It also supports larger held-out panels and BN calibration sets. Legacy runs
   retain their original reward-view resizing when re-ranked.

## Recommended next steps

Do not commit the only long retraining run based solely on the current proxy
winner. First use the saved model for a more thorough comparison:

```bash
sbatch run_select_search.sh --search-dirs search_lora_cvar_seed0 \
  --all-visited --rank-images 595 --calibration-batches 32 \
  --output architecture_seed0_reranked.json
```

This evaluates candidates on all 595 selection images, still as cropped-image
proxies, and recalibrates BN on 192 weight-training images. It does not resume
search, touch official validation, or train a child model. It requires GPU time;
it was not launched during the audit.

For a follow-up search with aligned input resizing and a weight-update budget
close to the paper-derived count, use a NEW directory:

```bash
SEARCH_SEED=0 SEARCH_DIR=search_lora_cvar_seed0_halfres_30k \
  sbatch run_search_ppo.sh --epochs 100 --warmup-epochs 50 --train-short-side 512
```

At the existing batch/split sizes this gives 29,700 weight updates and 14,850
warmup updates. It is a controlled budget comparison, not a claim that 100
epochs will perform best. It still uses cell-only PPO, not the published search
algorithm. Do not combine this new recipe with old runs in the cross-seed
selector. Its checkpoints require the same arguments when resubmitted.

The strongest next evidence would be equal-budget, independent short child
training trials for the top few cells and the reference, followed by one long
run for the strongest candidate. Extra search seeds help measure stability, but
do not replace that ranking check. No finite search or this audit can certify
the globally best architecture or guarantee 80% validation mIoU.
