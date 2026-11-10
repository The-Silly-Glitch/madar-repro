# MADAR Reproduction — Efficient Continual Learning for Malware Analysis with Distribution-Aware Replay

Reproduction of **Rahman, Coull, Yu & Wright, "MADAR: Efficient Continual Learning for Malware Analysis with
Distribution-Aware Replay"** (arXiv:2502.05760v2, CAMLIS 2025) on a laptop
(Kubuntu 26.04, Lenovo LOQ, Intel i5 HX, RTX 4050 6 GB, 24 GB RAM).

> **Status:** this README describes the finished project. Where measured numbers are still being produced,
> tables show the paper's reference value alongside `TBD` — fill these from `results/*.json` via
> `python scripts/summarize.py` (which regenerates `RESULTS.md` and the plots). Nothing here is hand-edited.

## What the paper does (in plain language)

Antivirus vendors see ~450K new malware samples a day. Retraining a classifier from scratch every time is
expensive, but training only on new samples causes **catastrophic forgetting** — the model forgets old
malware, letting attackers reintroduce it (**Retrograde Malware Attack**).

**MADAR** fixes this with *distribution-aware replay*: keep a small budget of old samples, chosen to cover
the heterogeneity of malware:

- Stratify the replay pool by **malware family**.
- Within each family, use **Isolation Forest** (contamination 0.1) to pick half **representative** (inlier)
  and half **anomalous** (outlier) samples (`α = 0.5`).
- Two budget modes: **Ratio (-R)** — per-family quota proportional to family size (best for Domain-IL);
  **Uniform (-U)** — equal quota per family (best for Class-IL / Task-IL).
- Two feature spaces: **MADAR** (raw features) and **MADARθ** (penultimate-layer activations, 128-d,
  extracted through a no-BN copy of the model to preserve heterogeneity).
- Domain-IL budget split: `β_G = γ·β`, `β_M = (1−γ)·β` with `γ = 0.5` (EMBER, balanced) and `γ = 0.9`
  (AZ, 9:1 goodware:malware).

**Scenarios.** Domain-IL = binary goodware/malware over time (EMBER: 12 monthly tasks; AZ: 9 yearly tasks).
Class-IL = malware-family classification, new families arrive (50 + 10×5, 11 tasks). Task-IL = same families
in 20 groups of 5, task ID given at test time (easiest setting).

**Headline paper claims.**
MADAR-R ≈ Joint on EMBER Domain with ~200K vs ~670K samples; MADAR-U 85.8% vs best prior 66.8% on EMBER
Class-IL at 20K; MADAR-U 98.7% vs Joint 98.8% on AZ Task-IL at 20K. MADAR > GRS at low budgets; Ratio wins
Domain-IL, Uniform wins Class-/Task-IL.

## Repository layout

```
madar-repro/
├── madar-official/          # upstream clone, tagged `upstream`, never edited in place
├── data/
│   ├── hf_downloads/        # raw HF files (EMBER_Domain/*.tar.bz2, EMBER_Class/*.npz, AZ_*)
│   └── processed/           # converter output in official-code layout (memmap float32)
├── src/
│   ├── converters/          # HF layout → official-code layout (month/year splits, memmaps)
│   ├── madar/               # replay samplers (ratio/uniform, IF, activation variant)
│   └── training/            # patched entry points (seed CLI, resumable runner)
├── scripts/
│   ├── run_grid.py          # resumable (scenario, method, budget, seed) grid runner
│   ├── summarize.py         # results/*.json → RESULTS.md + plots
│   └── smoke_test.py        # M1: 2 tasks, 1 epoch timing + RAM/VRAM probe
├── results/                 # per-run JSON: results/<scenario>_<method>_<budget>_seed<k>.json
├── plots/                   # AP-vs-task curves (Figs. 7–10 style)
├── recon_report.md          # M0 findings + code-excerpt verification
├── DEVIATIONS.md            # every departure from paper/official code + reason
├── RESULTS.md               # generated: ours (mean±std) vs paper + gaps
└── requirements.txt         # pinned environment
```

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # torch (CUDA 12.x), scikit-learn, numpy, pandas, tqdm
git checkout upstream -- madar-official   # to inspect the pristine upstream at any time
```

## Data

Gated HF dataset `IQSeC-Lab/MADAR` (~20 GB total). EMBER subset (~7.8 GB):

| File | Size |
|---|---|
| `EMBER_Domain/EMBER2018_Domain.tar.bz2` | 3.20 GB |
| `EMBER_Domain/EMBER2018_Domain_with_FamilyLabels.tar.bz2` | 1.41 GB |
| `EMBER_Class/XY_train.npz` | 2.89 GB |
| `EMBER_Class/XY_test.npz` | 321 MB |

AZ files are downloaded later for M5. **EMBER Task-IL has no separate file** — it reuses
`EMBER_Class/XY_train.npz` + `XY_test.npz` with a 20×5-family split (confirmed in
`madar-official/EMBER_Experiments/EMBER_Task/data.py`).

```bash
hf download --repo-type dataset --local-dir data/hf_downloads IQSeC-Lab/MADAR \
  EMBER_Domain/EMBER2018_Domain.tar.bz2 \
  EMBER_Domain/EMBER2018_Domain_with_FamilyLabels.tar.bz2 \
  EMBER_Class/XY_train.npz EMBER_Class/XY_test.npz

python -m src.converters.ember_domain   # → data/processed/ember_domain/2018-MM/{XY_train,XY_test}.npz
python -m src.converters.ember_class    # → data/processed/ember_class/{XY_train,XY_test}.npz (memmap float32)
```

Converters only repackage + cast dtype; no resampling, no test-set peeking. Never handle raw binaries —
feature vectors only.

## Reproducing each table

```bash
# M1 smoke test (2 tasks, 1 epoch; prints time/RAM/VRAM/IF cost + full-grid estimate)
python scripts/smoke_test.py --scenario ember_domain --methods none,grs,madar-r --budget 10000

# M2 EMBER Domain-IL → paper Table II
python scripts/run_grid.py --scenario ember_domain \
  --methods none,joint,grs,madar-r,madar-u,madar-theta-r,madar-theta-u \
  --budgets 1000,10000,50000,100000,200000,300000,400000 --seeds 42,43,44

# M3 EMBER Class-IL → paper Table IV
python scripts/run_grid.py --scenario ember_class \
  --methods none,joint,grs,madar-r,madar-u,madar-theta-r,madar-theta-u \
  --budgets 100,500,1000,5000,10000,15000,20000 --seeds 42,43,44

# M4 EMBER Task-IL → paper Table VI (same data files as Class-IL, 20×5 split)
python scripts/run_grid.py --scenario ember_task \
  --methods none,joint,grs,madar-r,madar-u,madar-theta-r,madar-theta-u \
  --budgets 100,500,1000,5000,10000,15000,20000 --seeds 42,43,44

# M5 AZ (Domain-IL → Table III, Class-IL → Table V, Task-IL → Table VII)
python scripts/run_grid.py --scenario az_domain --methods none,joint,grs,madar-r,madar-theta-r \
  --budgets 1000,10000,50000,100000,200000,300000,400000 --seeds 42,43,44
python scripts/run_grid.py --scenario az_class --methods none,joint,grs,madar-u,madar-theta-u \
  --budgets 100,500,1000,5000,10000,15000,20000 --seeds 42,43,44
python scripts/run_grid.py --scenario az_task --methods none,joint,grs,madar-u,madar-theta-u \
  --budgets 100,500,1000,5000,10000,15000,20000 --seeds 42,43,44

# Regenerate report + plots from raw JSON
python scripts/summarize.py   # writes RESULTS.md + plots/*.png
```

Every run writes `results/<scenario>_<method>_<budget>_seed<k>.json` on completion and skips completed
combos on restart. Seeds default to `42,43,44` (paper used 10–15; see DEVIATIONS.md).

## Results (ours vs paper)

Regenerate with `python scripts/summarize.py`. Values below show the paper reference; `TBD` marks cells
filled in by that script from `results/*.json`.

### M2 — EMBER Domain-IL (paper Table II; Joint 96.4, None 93.1)

| Budget | Method | Paper AP | Ours (mean±std) | Gap |
|---|---|---|---|---|
| 200K | MADAR-R | 96.0 | TBD | TBD |
| 200K | MADARθ-R | 96.1 | TBD | TBD |
| 200K | GRS | 95.9 | TBD | TBD |
| 1K | MADAR-R | 93.7 | TBD | TBD |

### M3 — EMBER Class-IL (paper Table IV; Joint 86.5, None 26.5)

| Budget | Method | Paper AP | Ours (mean±std) | Gap |
|---|---|---|---|---|
| 20K | MADAR-U | 85.8 | TBD | TBD |
| 20K | MADARθ-U | 86.2 | TBD | TBD |
| 20K | GRS | 84.6 | TBD | TBD |
| 20K | iCaRL (prior best) | 66.8 | paper ref | — |

### M4 — EMBER Task-IL (paper Table VI; Joint 97.0)

| Budget | Method | Paper AP | Ours (mean±std) | Gap |
|---|---|---|---|---|
| 20K | MADAR-U | 95.8 | TBD | TBD |
| 20K | MADAR-R | 95.6 | TBD | TBD |
| 20K | GRS | 95.0 | TBD | TBD |

### M5 — AZ (paper Tables III, V, VII)

| Scenario | Budget | Method | Paper AP | Ours (mean±std) | Gap |
|---|---|---|---|---|---|
| Domain | 200K | MADAR-R | 97.0 | TBD | TBD |
| Domain | 400K | MADARθ-R | 97.2 | TBD | TBD |
| Class | 20K | MADAR-U | 91.5 | TBD | TBD |
| Task | 20K | MADAR-U | 98.7 (Joint 98.8) | TBD | TBD |

AP-vs-task curves: `plots/ember_domain.png`, `plots/ember_class.png`, `plots/ember_task.png`,
`plots/az_*.png` (paper Figs. 7–10 style).

## Honest analysis (fill after `summarize.py`)

- **Where numbers match:** TBD — compare per-budget gaps; ≤1 pp on Domain-IL counts as match given 3 vs 10–15 seeds.
- **Where they don't, and likely cause:** TBD — candidates: fewer seeds, memmap/float32 path, IF version differences, no-BN-via-V2-copy behavior, `--iters` semantics.
- **Do the main claims hold?** TBD — (a) MADAR > GRS at low budgets; (b) Ratio best for Domain-IL; (c) Uniform best for Class-/Task-IL. Each is accepted/rejected per scenario with the budget range stated.

## Deviations from paper / official code

Full log in `DEVIATIONS.md`. Confirmed items:

1. **Seeds:** official code hard-codes `args.seed = 42` (verified in `EMBER_Class/main.py:132`,
   `AZ_Class/main.py:132`). Made it a CLI arg; main runs use 3 seeds (42,43,44). Paper used 10–15.
2. **MADARθ BN:** `--batch_norm` uses `action="store_true"` but is compared to `'yes'`, so the flag never
   triggers; the fallback path loads `Ember_MLP_Net_V2` whose `forward` comments out BN. Net effect = no-BN
   extraction as the paper requires, via a different mechanism. Preserved and logged.
3. **Class-/Task-IL steps:** controlled by `--iters` (default 10000), not `--epoch`. Kept official defaults;
   values recorded per run.
4. **GRS:** matches paper Sec. IV-D (random sample from accumulated pool) — not a deviation.
5. **AZ Domain γ:** code uses 0.9, matching the paper — not a deviation. A 50/50 ablation at 2–3 budgets is
   optional (time permitting) to test which setting produced Table III.
6. **Memory:** float32 memmaps, index-based replay, uint8/sparse AZ binaries; RSS cap ~18 GB, restructure
   before swap.

## Milestones

- [x] **M0** — recon + code-excerpt verification (`recon_report.md`)
- [ ] **M1** — smoke test + runtime estimate
- [ ] **M2** — EMBER Domain-IL full grid
- [ ] **M3** — EMBER Class-IL full grid
- [ ] **M4** — EMBER Task-IL full grid
- [ ] **M5** — AZ grids
