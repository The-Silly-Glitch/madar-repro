# MADAR Reproduction — Continual Learning for Malware Analysis

Reproduction of **Rahman, Coull, Yu & Wright, "MADAR: Efficient Continual Learning for Malware Analysis
with Distribution-Aware Replay"** (arXiv:2502.05760v2, CAMLIS 2025) on a consumer laptop
(Kubuntu 26.04, Intel i5 HX, RTX 4050 6 GB, 24 GB RAM).
Paper environment for reference: CentOS-7, 40 CPU cores, 128 GB RAM, 4× RTX 2080Ti.

> **Status:** M0 (recon) and M1 (smoke test) are done and committed. M2+ full grids are pending —
> see `SMOKE_REPORT.md` for the runtime estimate and blockers. Result tables below show the paper's
> reference numbers; ours will be filled in from `results/*.json` as grids complete.

## The paper in brief

Antivirus vendors see ~450K new malware samples a day. Retraining from scratch is expensive, but training
only on new samples causes **catastrophic forgetting**, letting attackers reintroduce old malware
(**Retrograde Malware Attack**). MADAR keeps a small replay budget of old samples chosen to cover malware
heterogeneity: stratify by **family**, then per family pick half **representative** (inlier) and half
**anomalous** (outlier) samples via Isolation Forest (contamination 0.1, `α = 0.5`). **Ratio (-R)**
quotas scale with family size (best for Domain-IL); **Uniform (-U)** quotas are equal per family (best for
Class-/Task-IL). **MADARθ** runs the same selection on penultimate-layer activations (128-d, no-BN copy).

## Repository contents (everything needed to recreate from scratch)

```
├── README.md               # this file
├── requirements.txt        # pinned env (torch via cu126 CUDA-12.x index — see §4)
├── recon_report.md         # M0: official-repo audit + code-vs-paper verification
├── DEVIATIONS.md           # every departure from paper/official code + reason
├── SMOKE_REPORT.md         # M1 numbers + M2 runtime extrapolation
├── scripts/
│   └── smoke_test.py       # re-runs the M1 smoke end to end
├── src/
│   └── ember_domain/       # patched copies of official EMBER_Domain scripts
│       ├── GRS.py / MADAR_IFS.py   # entry points (only these two so far)
│       ├── ember_model.py / ember_utils.py / ember_pjr_utils.py / ember_features.py
├── results/
│   └── smoke_*.json        # measured M1 per-task records (evidence, not inputs)
└── data/                   # GITIGNORED — you download this (§2); ~14 GB extracted
```

`data/`, `*.npz`, `*.parquet`, `*.tar.bz2`, checkpoints (`*.pt`), venvs, and `madar-official/`
(the pristine upstream clone, kept local-only) are never committed.

## 1. Reference code (not in this repo — clone it)

```bash
git clone https://github.com/IQSeC-Lab/MADAR madar-official
cd madar-official && git checkout ef1a771 && cd ..
```

Pin `ef1a771` — that is the commit audited in `recon_report.md`. Layout: six experiment folders
(EMBER/AZ × Domain/Class/Task-IL). Domain-IL entry points are `GRS.py` (baseline), `MADAR_IFS.py`
(MADAR), `MADAR_AWS.py` (MADARθ); Class-/Task-IL use `main.py --replay_config={grs,ifs,aws}`.
Our `src/ember_domain/` files are copies of these with only the fixes listed in `DEVIATIONS.md`
(hard-coded paths → `--out_dir`, fixed `--seed`, `np.Inf` fix, `num_workers=0`); training/replay
math is untouched. The upstream remote stays the source of truth — never edit it in place.

## 2. Datasets — what, where, how

All datasets are **pre-extracted feature vectors only** — no malware binaries are ever downloaded.
Primary source (gated — requires login + accepting terms):

- Hugging Face: **`IQSeC-Lab/MADAR`** — https://huggingface.co/datasets/IQSeC-Lab/MADAR (~20 GB total)

```bash
pip install -U huggingface_hub
hf auth login                                  # then open the URL above and accept access terms
hf download --repo-type dataset --local-dir data/hf_downloads IQSeC-Lab/MADAR \
  EMBER_Domain/EMBER2018_Domain_with_FamilyLabels.tar.bz2 \
  EMBER_Class/XY_train.npz EMBER_Class/XY_test.npz
```

Gating is `auto` but the account must still be accepted — an unauthenticated/unauthorized
download fails with `403 / GatedRepoError`; `hf auth whoami` succeeding is not sufficient.

### EMBER (Windows PE) — used in M1–M4

| HF path | Size | Contents |
|---|---|---|
| `EMBER_Domain/EMBER2018_Domain_with_FamilyLabels.tar.bz2` | 1.41 GB | 12 month folders `2018-MM/`, each `XY_train.npz` (`X_train, Y_train, Y_family_train`) + `XY_test.npz` (`X_test, Y_test, Y_family_test`), float32, 2381-d. **This is the file the Domain code needs.** |
| `EMBER_Domain/EMBER2018_Domain.tar.bz2` | 3.20 GB | Same months without family labels. Not needed — skip it. |
| `EMBER_Class/XY_train.npz` | 2.89 GB | 303,331 × 2381 float32, labels 0–99 (top-100 families). Backs **both** Class-IL and Task-IL (Task-IL just re-splits the same 100 families into 20×5; confirmed in official `EMBER_Task/data.py`). |
| `EMBER_Class/XY_test.npz` | 321 MB | 33,704 × 2381 float32. |

Sanity check after download: Jan-2018 train (55,722) + test (6,192) = 61,914 rows = paper Table I
(29,423 goodware + 32,491 malware).

```bash
mkdir -p data/processed/ember_domain
tar -xjf data/hf_downloads/EMBER_Domain/EMBER2018_Domain_with_FamilyLabels.tar.bz2 \
  -C data/processed/ember_domain   # → data/processed/ember_domain/2018-01/ ...
```

No conversion needed — keys/dtypes already match the official scripts. Class files are used in place.

### AZ (Android, AndroZoo-derived) — used in M5 (pending)

| HF path | Size | Contents |
|---|---|---|
| `AZ_Class/AZ_Class_Train.npz` | 2.52 GB | 100 families, 2,439-d Drebin features |
| `AZ_Class/AZ_Class_Test.npz` | 280 MB | Test split |
| `AZ_Domain/*_AZ_train.parquet` (2008–2016) | ~65–80 MB each | Yearly binary tasks, 9:1 goodware:malware |
| `AZ_Domain/*_AZ_test.parquet` (2008–2016) | ~0.5–9 MB each | Yearly hold-outs |

Download only when starting M5. Expected post-processing: parquet → the `{year}_Domain_AZ_{Train,Test}_Transformed.npz`
layout the AZ scripts expect (converter TBD in M5).

### Fallback (if HF access fails)

Public EMBER 2018 features (no login, ~1.1 GB): https://github.com/elastic/ember.
It lacks the paper's month/family splits, so M1+ scripts would need a splitter — not yet written.

## 3. Paper setup reference (targets we reproduce)

- **Domain-IL:** EMBER 12 monthly binary tasks; AZ 9 yearly binary tasks.
- **Class-IL:** 100 families, start 50 + 5×10 (11 tasks), only seen-family outputs active.
- **Task-IL:** same 100 families in 20×5 groups, task ID given at test time.
- **Model:** MLP + batch norm + dropout, Adam lr=0.001 (official defaults kept; Class-/Task-IL steps
  are controlled by `--iters`, default 10000, not `--epoch`).
- **MADAR:** global pool + family-freq dict; `β_G = γ·β`, `β_M = (1−γ)·β` (`γ`=0.5 EMBER, 0.9 AZ);
  per-family `ceil` quota (overshoot above budget is official behavior); IF contamination 0.1, α=0.5.
- **Metric:** APᵢ = mean accuracy over tasks ≤ i after training task i; global AP = mean over tasks, ×100.
- **Budgets:** Domain-IL 1K→400K; Class-/Task-IL 100→20K. Seeds: paper 10–15, we use 3 (42,43,44).

## 4. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Two gotchas (both in `DEVIATIONS.md`): official `torch==2.12.1` must come from the PyTorch
**cu126** index (plain PyPI serves a CUDA-13 wheel), and numpy/pandas/sklearn pins were raised
because the official ones have no Python 3.14 wheels. M1 ran on CPU torch because the box's GPU
was not enumerable (`nvidia-smi`: no devices) — fix NVML before M2 (see `SMOKE_REPORT.md`).

## 5. Reproduce

```bash
# M1 smoke (2 tasks, 1 epoch, None+GRS+MADAR-R @10K, seed 42) — re-runnable any time:
python scripts/smoke_test.py
```

M2+ full grids: pending GPU fix + late-task memory trial. Runner + exact commands will be
documented here before launch; the estimate is **~7 days CPU-only / ~3 days with GPU**
(12 tasks × 7 methods × 7 budgets × 3 seeds; IsolationForest is the CPU bottleneck).

## 6. Key results so far

M1 is a cost probe (1 epoch), not convergence — 1-epoch APs: None .787/.844, GRS .787/.847,
MADAR-R .875/.898 (tasks 01/02). Paper targets for later: EMBER-Domain Joint 96.4 / MADAR-R 96.0
@200K; EMBER-Class Joint 86.5 / MADAR-U 85.8 @20K; EMBER-Task Joint 97.0 / MADAR-U 95.8 @20K;
AZ-Task MADAR-U 98.7 @20K ≈ Joint 98.8. Claims under test: MADAR > GRS at low budgets;
Ratio wins Domain-IL, Uniform wins Class-/Task-IL.

## 7. File guide

- Need the why: `recon_report.md` (M0 audit + verified code-vs-paper diffs).
- Need what changed and why: `DEVIATIONS.md` (10 items; e.g. seed fix, BN-quirk finding, GRS/γ
  suspected deviations that turned out to match the paper).
- Need costs: `SMOKE_REPORT.md` + `results/smoke_*.json`.
- Run something: `scripts/smoke_test.py` (usage in its docstring).
