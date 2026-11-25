# Reference-guided search and independent F20 selection

The completed F20/500k BN diagnostics confirm an architecture gap:

| Best checkpoint | Stored BN, FP32 | Train-only recalibrated BN |
| --- | --- | --- |
| Published reference | 73.943% | 73.909% |
| Previously selected cell | 71.285% | 71.323% |

The approximately 2.59-point gap remains after recalibration. These diagnostics
do not update the saved checkpoints. The next workflow addresses architecture
selection; no improvement over the reference is guaranteed.

## What is implemented

1. A fixed library of 128 unique cells: the reference and one/two edge changes
   around it. Only separable/atrous convolutions are allowed in this experiment;
   the macro path stays fixed. Candidates have at most the reference's
   10,147,427 parameters at F20. This is a parameter cap, not a measured FLOP cap.
   Input choices remain valid and distinct within each node. Swapping the two
   branches of an addition does not count as a different architecture.
2. The corrected single-path supernet, input-specific LoRA rank 4, and measured
   mean/CVaR reward remain. A categorical PPO policy chooses from the explicit
   local library. This is a new search space and policy, version
   `adl_reference_local_v1`; old checkpoints cannot resume into it.
3. Search retains F8/321 crops, 100 epochs, 50 warmup, batch 6 and 29,700 weight
   updates. It trains from scratch. The reference gets a weight-training anchor
   every tenth minibatch, as before. No old child weights are loaded.
4. Every library cell is re-ranked at the final supernet weights on the same
   595 held-out training images/crops, with 32 common BN-calibration batches.
   The shortlist contains reference ID 0 plus four challengers. It includes the
   mean leader, another high-reward candidate, and diverse high-scoring cells.
   No proxy winner is treated as a demonstrated full-training winner.
5. Finalists train independently at F20/crop769, batch16, seed0, BN .01 and no
   SDP. Their LR horizon is always 500k; stage boundaries are absolute update
   counts. Selection averages the last three scheduled measurements, rather
   than choosing the highest noisy checkpoint.

The convolution-only local search is a hypothesis motivated by the earlier
pool/skip-heavy winner. It does not establish that those operations are bad in
general. LoRA/CVaR effectiveness still needs ablations; any gain here could
also come from the changed search space or independent selection.

## Data separation

Search uses the existing 1785 weight / 595 controller / 595 selection partition
of Cityscapes **train**. Independent probes train on the combined first two
partitions (2380 images), leaving the same 595 selection images out of gradient
updates and BN calibration. Probe scoring uses full-resolution selection images.
The selection images already influence proxy shortlisting; this is a reused
development set, not an untouched final test set.

Every probe evaluation first temporarily recalibrates BN with 32 batches of 16
training images, using the same seeded calibration views for all candidates
and checkpoints. It then restores the stored BN statistics and training RNG.
Selection reports record this protocol. A probe's `best.pth` retains ordinary
training BN buffers; its selection score describes recalibrated evaluation.

Official Cityscapes validation is not loaded during local search or independent
probes. The ordinary evaluator refuses probe checkpoints. Final retraining is
fresh on all 2975 training images and uses official validation normally. A probe
checkpoint cannot be resumed as full-data training. Final single-scale scores
remain comparable to the existing F20/500k/no-SDP recipe, subject to seed variance.

## Stage 1: GPU smoke check, then real search

The private CUDA 11.8 environment is already installed. No GPU-node downloads
or shared-conda modifications are required.

```bash
cd /scratch/ajin/Autodeeplab/modified
SEARCH_SEED=0 sbatch run_search_local.sh --smoke-test
```

Inspect `local_search_<jobid>.out/.err`. Success must include
`Successful weight updates: 4; PPO updates: 2` and `Search complete`.
Smoke output goes to `search_local_v1_seed0_smoke` and cannot enter selection.

### Recovery from smoke job 47972

This job completed all four weight updates and its first PPO update, then
stopped on non-finite candidate logits during the second rollout. Its saved
model and policy tensors were finite. The smoke model is F2/crop65 with only
four calibration images, unlike the real F8/crop321 search with 48 calibration
images. AMP forward overflow or unstable calibration is consistent with this
failure; the log alone does not isolate the exact operation that overflowed.

Scoring now detects non-finite calibration outputs/buffers as well as evaluation
outputs. A failed AMP attempt restores state, resets/recalibrates BN, and retries
the entire candidate on the identical cached images in FP32. Invalid results are
never converted into rewards; an FP32 failure still stops the job. Candidate
records include `scoring_precision`, `amp_fallback` and `scoring_protocol`.
Finite AMP scores retain the existing calculation.

Resubmit the same smoke command to resume its four-update checkpoint:

```bash
SEARCH_SEED=0 sbatch run_search_local.sh --smoke-test
```

Do not add `--no-amp` to that resume: it changes the saved training recipe.
The scoring fallback does not change training AMP or checkpoint tensor layout.
After this patch, 31 lightweight CPU tests passed; the actual GPU retry remains
to be verified by this resubmission.

Job 47973 was advancing through epoch 64 with 13 PPO updates and an empty error
log when checked. It does not need cancellation because the separate smoke job
failed. An already running Python process keeps its imported scoring code; the
patch is loaded by newly started jobs. If the real job later stops at this same
scoring error, resubmit its original command once it has stopped; the saved
weights and optimizer state can be resumed, with an incomplete rollout replayed.
Never launch two real jobs against `search_local_v1_seed0` simultaneously.

For a new experiment, submit the real search after smoke succeeds. Skip this
submission while job 47973 (or another job in the same directory) is running:

```bash
SEARCH_SEED=0 sbatch run_search_local.sh
```

Output: `search_local_v1_seed0`. Resubmit the identical command after each
48-hour allocation until `Search complete` appears (29,700 updates / 50 PPO
updates). Do not run two jobs simultaneously in the same directory. The
100-epoch schedule is fixed at the first submission; changing it on resume
changes the experiment and is rejected.

## Stage 2: re-rank and prepare independent candidates

After the real search finishes:

```bash
SEARCH_SEED=0 sbatch run_select_local.sh
```

This evaluates all 128 library cells, then writes
`local_candidates_v1_seed0/manifest.json`, `candidate_0.json` through
`candidate_4.json`, and `probe_split.json`. Logs: `local_select_<jobid>.out/.err`.
The manifest fixes the recipe, candidate identities and partition hash.
Preparation is idempotent for identical contents and refuses to replace a
different existing shortlist. Preserve this directory for all later stages.

## Stage 3: five independent 50k probes

```bash
sbatch run_probe_local.sh
```

The default array is `0-4%1`: one GPU job at a time. Use `--array=0-4%2` only if
two GPUs are available. Candidate 0 is always the reference. No 500k training
run is launched by this command. Each job stops at 50k successful updates with
a 500k LR horizon, writes under `local_candidates_v1_seed0/probes/`, and can be
resubmitted unchanged. A completed stage does not run extra updates.

Once every candidate has reached 50k, run this small CPU reporting command on
the login node (it reads JSON, not checkpoints or images):

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 \
  .venvs/retrain_cu118/bin/python local_experiments.py rank \
  --manifest local_candidates_v1_seed0/manifest.json --stage-step 50000
```

The report averages measurements at 30k/40k/50k and prints suggested next array
indices: reference 0 and the two strongest challengers. This is a provisional
screen. Early rankings can reverse, as the old 20k and 500k comparison showed.
When candidates are close or curves cross, retain more than two challengers
rather than interpreting a small 50k difference as a reliable result.

## Stage 4: 150k and an independent training seed

Replace the example IDs **1,2** below with the actual retained challengers.
All retained candidates, including the reference, must complete both seeds.

```bash
STOP_AT_STEP=150000 PROBE_SEED=0 sbatch --array=0,1,2%1 run_probe_local.sh
STOP_AT_STEP=150000 PROBE_SEED=1 sbatch --array=0,1,2%1 run_probe_local.sh
```

Seed0 continues from its 50k checkpoints. Seed1 starts independently from scratch.
Neither changes the 500k LR horizon. An optional 100k stop is supported, but
150k and two seeds are required for a final export. At 150k, comparison averages
130k/140k/150k and pairs each candidate against the same-seed reference.

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 \
  .venvs/retrain_cu118/bin/python local_experiments.py rank \
  --manifest local_candidates_v1_seed0/manifest.json --stage-step 150000 \
  --candidate-ids 0 1 2 --seeds 0 1 \
  --export local_candidates_v1_seed0/winner.json
```

The predefined selection rule requires a challenger to beat the reference in
both seeds and by at least 0.5 mean-IoU points averaged across seeds and the last
three measurements. This margin is a practical screening rule, **not a
statistical significance test or a guarantee of the final 500k ranking**.
Image CVaR is reported alongside mean IoU; final promotion prioritizes mean IoU.
Do not lower the margin after seeing scores simply to force a winning cell.

If no challenger qualifies, the export retains the reference and the full-run
launcher refuses to launch another reference retrain automatically. Diagnose
the search/probe results before spending the final budget.

## Stage 5: fresh full retraining of a qualified challenger

Only after an independently screened challenger qualifies:

```bash
sbatch run_retrain_local_winner.sh
```

This uses `winner.json` and a new directory
`retrain_local_winner_f20_500k_seed0`. It initializes fresh weights, trains on all
2975 images for 500k updates, and validates on official Cityscapes val.
Resubmit the same command after each allocation to resume. No LoRA adapters or
PPO updates are used in child training. For another final seed, set BOTH a new
`RUN_DIR` and `FINAL_SEED`.

After completion:

```bash
RUN_DIR=retrain_local_winner_f20_500k_seed0 sbatch run_test_baseline.sh \
  --results-dir log_local_winner_f20
```

Report single-scale/no-flip results against the reference's best 73.942% and
the paper's 75.20%, with the respective protocols stated. Final repeated-seed
and LoRA/CVaR ablations remain necessary for attributing an improvement.

## Time-limited route: candidate 4 without the second probe seed

On November 4 the user chose to prioritize a full 500k run because there was
not enough time for more probe seeds. Candidate 4 had a +0.608-point average
advantage at 130k/140k/150k in training seed 0, driven by the final checkpoint;
this is provisional evidence, not two-seed confirmation.

```bash
sbatch run_retrain_candidate4_500k.sh
```

This launcher uses the existing candidate-4 architecture directly, preserving
the two-seed rule in the separate automatic winner-export workflow. It starts
fresh on all 2975 training images, with F20, 500k updates, seed0, batch16,
crop769, BN .01 and no SDP, matching the reproduced reference recipe. It does
not continue the 150k probe, which trained on only 2380 images.

Output is `retrain_candidate4_f20_500k_seed0_nosdp`; logs are
`candidate4_500k_<jobid>.out/.err`. Resubmit the same command after each 48-hour
allocation. The launcher automatically resumes its own latest checkpoint.
Keep its default fresh run directory; do not redirect it to a probe directory.

At the observed roughly .83 seconds/update, 500k updates take about 4.8 days;
allow approximately 5–6 days plus queue delays, depending on actual throughput.
After completion, evaluate the best checkpoint with:

```bash
RUN_DIR=retrain_candidate4_f20_500k_seed0_nosdp sbatch run_test_baseline.sh \
  --results-dir log_candidate4_f20_500k
```

A single final run can establish its measured validation score. It cannot
establish repeatability or isolate a LoRA/CVaR contribution. No accuracy gain
is guaranteed by the provisional selection.

The candidate-4 500k run is now complete. See `CANDIDATE4_RESULTS.md` for the
matched results, the lower tail-IoU result, and evaluation-only submission
commands for the saved reference and candidate checkpoints.

## Cost and verification

Default independent training consumes 1M probe updates in total:
5*50k, then 3*100k more for seed0, plus 3*150k for seed1. At the observed roughly
0.7–0.8 seconds/update, that is about 8–10 GPU-days of optimizer work, plus
calibration/selection overhead. Search and final retraining are additional.
Concurrency shortens wall-clock time but not GPU-days. Actual speeds depend on
the candidate. The stages let you review evidence before spending more.

The assistant ran 29 lightweight CPU tests, including actual tiny CPU
optimization/checkpoint/resume with mocked GPU availability, local-policy PPO,
candidate constraints, disjoint partitions, protocol checks and reference
fallback. Shell/Python syntax checks passed. No GPU jobs or Cityscapes training
were run by the assistant; GPU performance remains to be checked by the smoke job.
