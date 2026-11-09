# Deviations from paper / official code

Ground rule: feature vectors only; never tune on test sets; every departure logged here.

## M1 — environment
1. **torch 2.12.1 source (CUDA 12.x).** The official `requirements.txt` pins
   `torch==2.12.1` with `torchvision==0.27.1`, and `environment.yml` pairs it with
   `pytorch-cuda=12.4`. A plain PyPI install of `torch==2.12.1` today resolves to a
   **CUDA-13** wheel, not CUDA 12.x. Our `requirements.txt` therefore keeps the
   official version numbers but installs them from the PyTorch **cu126** wheel index
   (`--index-url https://download.pytorch.org/whl/cu126`), which ships a real
   CUDA-12.x build (`torch 2.12.1+cu126`, `torchvision 0.27.1+cu126`).
2. **NumPy / pandas / scikit-learn / scipy / matplotlib pins raised.** The official pins
   (`numpy==1.25.2`, `pandas==2.1.1`, `scikit-learn==1.5.0`, `scipy==1.10.0`) have no
   wheels for the box's Python 3.14 and will not build. Pinned to tested modern
   equivalents instead: `numpy==2.5.4`, `pandas==3.1.1`, `scikit-learn==1.9.1`,
   `scipy==1.18.1`, `matplotlib==3.11.2`, `seaborn==0.13.2`. No API used by the
   repro path changed behavior (StandardScaler, IsolationForest, DataLoader paths
   verified by the smoke runs).
3. **M1 smoke ran on CPU torch (`torch==2.12.1+cpu`).** The box has an RTX 4050 and
   driver 595.99 loaded (`/proc/driver/nvidia` lists the GPU, not excluded), but the
   NVML enumeration layer reports "No devices were found" (`nvidia-smi` exit 6), so
   `torch.cuda.is_available()` is False for any build. The smoke therefore measures
   CPU wall-clock/RAM; peak-VRAM is reported as unavailable. The M2 grid needs the
   driver/NVML issue fixed first (or a CUDA-12.x torch reinstall once the GPU is
   enumerable) — timings below are explicitly CPU-based and the extrapolation states
   the GPU assumption separately.

## M1 — code fixes (all applied to copies under `src/ember_domain/`, never to `madar-official/`)
4. **`--data_dir` honored; stale hard-coded paths redirected.** Both `GRS.py` and
   `MADAR_IFS.py` accept `--data_dir` but write to hard-coded relative paths
   (`../GRS_SavedModel`, `../GRS_SavedResults_`, `../../IFS_SavedModel`,
   `../../IFSS_SavedModel` (note the `IFSS` typo, kept as-is upstream),
   `../../IFS_SavedResults_`, and `./Results_Domain/` which is never created and
   would crash). Added `--out_dir`; all writes go under it (including creating
   `<out_dir>/Results_Domain/`).
5. **Task slicing for smoke.** Added `--max_tasks` (smoke uses 2: `2018-01`, `2018-02`).
   No training logic touched.
6. **Deterministic seed.** Official code draws `exp_seeds` from `random.randint` and
   never seeds `random`/`numpy` (only `torch.manual_seed`). Added `--seed`
   (default 42) with `exp_seeds = [seed + i ...]`, plus `random.seed(exp)` /
   `np.random.seed(exp)` per experiment.
7. **Added measurements only.** Per-task JSON (`smoke_*.json`): wall-clock minutes,
   `training_early_stopping` minutes, peak RSS (`resource.getrusage`), peak CUDA MB
   (or null), IsolationForest replay-selection seconds (MADAR only), and AP
   (`sklearn.metrics.average_precision_score` on raw sigmoid outputs) + ROC-AUC
   alongside the official acc/precision/recall/F1. Training, replay, eval logic
   untouched.
8. **`np.Inf` → `np.inf` in `src/ember_domain/ember_model.py` (EarlyStopping).**
   `np.Inf` was removed in NumPy 2.0; the official code crashes at first use.
   One-token fix, no behavior change.
9. **Kept official `X_test`-as-validation behavior.** `training_early_stopping` is
   called with the test split as `X_valid` in both scripts (i.e. early stopping
   *looks at* test loss). With smoke's `--num_epoch 1`, early stopping cannot trigger
   (patience 5), so no test-based model selection occurs here; the M2 runner must
   decide whether to preserve or fix this (log as a method deviation if changed).
10. **DataLoader `num_workers=1` → `0` in `src/ember_domain/ember_utils.py`.**
    With `num_workers=1`, loader workers die with
    `ConnectionResetError` (forkserver) on this box's Python 3.14. `num_workers=0`
    loads the in-RAM tensors directly; no semantics change.

## Pending (for M2+)
- Seeds: paper used 10–15; we use 3 (42, 43, 44) — matches README plan.
- MADARθ BN quirk, `--iters` semantics, γ splits: per README_PROJECT.md, unchanged.
