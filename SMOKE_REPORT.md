# M1 smoke report — EMBER Domain-IL (2 tasks, 1 epoch, seed 42, budget 10K, CPU)

Date: 2026-11-10. Box: Kubuntu 26.04, Intel i5 HX, RTX 4050 (GPU **not enumerable** —
`nvidia-smi`: "No devices were found", so `torch.cuda.is_available()` is False;
all numbers below are **CPU**). Env: `requirements.txt` (torch 2.12.1+cpu for smoke;
cu126 CUDA-12.x index pinned for M2+). Data: `data/processed/ember_domain/`
(extracted from `EMBER2018_Domain_with_FamilyLabels.tar.bz2`; Jan-2018 train+test =
61914 rows matches paper Table I; keys/dtypes verified float32 2381-d).

Scripts: `src/ember_domain/GRS.py` / `MADAR_IFS.py` (copies of official code, minimal
smoke patches — see DEVIATIONS.md items 4–10), run with `--data_dir
data/processed/ember_domain/ --max_tasks 2 --num_epoch 1 --seed 42`.
Official training/replay/eval logic untouched.

## Measured results (1 epoch)

| Method | Budget | Task | n_train | n_test | Wall-clock (min) | Train (min) | IF select (s) | Peak RSS (GB) | Peak VRAM | acc | AP | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 2018-01 | 55722 | 6192 | 0.29 | 0.18 | — | 2.2 | n/a (CPU) | 0.533 | 0.787 | 0.747 |
| none | 0 | 2018-02 | 48723 | 11606 | 0.31 | 0.17 | — | 2.7 | n/a | 0.566 | 0.844 | 0.797 |
| grs | 10K | 2018-01 | 55722 | 6192 | 0.28 | 0.18 | — | 2.2 | n/a | 0.533 | 0.787 | 0.747 |
| grs | 10K | 2018-02 | 58723 | 11606 | 0.35 | 0.20 | — | 2.9 | n/a | 0.572 | 0.847 | 0.802 |
| madar-r | 10K | 2018-01 | 55722 | 6192 | 0.45 | 0.32 | 0.0 | 2.7 | n/a | 0.677 | 0.875 | 0.854 |
| madar-r | 10K | 2018-02 | 59644 | 11606 | 1.26 | 0.35 | 44.7 | 3.4 | n/a | 0.774 | 0.898 | 0.873 |

Notes:
- Task-1 none/grs/madar-r and task-1 grs==none (no replay on first task, same seed →
  bit-identical) confirm determinism; MADAR-R task 1 differs only by batch size
  (official default 2000 vs GRS 6000).
- MADAR-R task 2 replayed 5000 goodware + 5921 malware = 10921 > 10K budget: expected
  official behavior (per-family `ceil` in ratio mode overshoots).
- "AP" = `average_precision_score` on raw sigmoid outputs (official code only reports
  metrics on rounded predictions, so AP was added by the smoke patch).
- Raw per-task records: `results/smoke_ember-domain_{none,grs,madarr}_*.json`
  (+ `smoke_out_*/smoke_*.json` + `smoke_*.log` on disk, not committed).
- 1-epoch accuracy is far below paper Table II values (None 93.1 / Joint 96.4 at full
  training) — expected; smoke measures cost, not convergence.

## Fixes the smoke forced (all in DEVIATIONS.md)
`torch` from cu126 index (PyPI default = CUDA-13); modern numpy/pandas/sklearn/scipy
pins; `np.Inf`→`np.inf`; DataLoader `num_workers` 1→0 (forkserver crash); `--seed`,
`--max_tasks`, `--out_dir` (official hard-coded `../GRS_SavedModel`, `../../IFS_*`,
`./Results_Domain/`).

## M2 full-grid extrapolation (12 tasks × 7 methods × 7 budgets × 3 seeds)

Model fit to the table above: GRS/none train ≈ 0.18 min per 56K samples/epoch +
0.12 min overhead; MADAR train ≈ 0.33 min per 56K (batch 2000) + 0.16 min overhead;
IsolationForest selection ≈ 45 s at 55K pool, linear in pool size (budget-independent —
IF fits on the full pool). Official epoch defaults: GRS/Joint/None `num_epoch=100`
with early stopping (patience 5, assume ~10 effective epochs); MADAR-IFS
`num_epoch=1`.

- **CPU-only (as smoked): ~92 h at 1-epoch-equivalent; ~160–170 h (~7 days) with the
  ×10 early-stopping allowance for the GRS family.** MADAR runs are IF-bound
  (~47 min IF per 12-task run, single-threaded sklearn, unaffected by GPU).
- **With a working GPU (NVML fix + cu126 torch): MLP training ~3–5× faster, IF
  unchanged → ~3 days, IF-bound.** Biggest available speedup: `n_jobs=-1` (or
  per-family parallel) IsolationForest, same math, same seeds.
- **Memory warning:** peak RSS 3.4 GB at 2 tasks; late tasks hold ~600K×2381 float32
  (~6 GB) plus scaler copies, while the box currently has only ~4–8 GB free of
  24 GB. Run M2 with nothing else resident (or spill to memmap) and watch RSS;
  restructure before swap.

Verdict: pipeline (extract → official scripts → metrics → JSON) works end to end.
**Do NOT start the M2 grid** until (a) the NVML/GPU issue is fixed, (b) a late-task
memory trial (e.g. single 12-task joint run) clears the RSS budget.
