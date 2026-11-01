# MADAR Reproduction Reconnaissance Report

**Date:** 2026-11-01
**Paper:** MADAR: Efficient Continual Learning for Malware Analysis with Distribution-Aware Replay (arXiv:2502.05760v2, accepted at CAMLIS 2025)
**Official repository:** https://github.com/IQSeC-Lab/MADAR
**Local clone:** `/home/honeysh/Projects/madar-repro/madar-official`
**HEAD:** `ef1a77176e5f73564787550eee118d99af8a3c59`

## 1. Repository structure

```
madar-official/
├── README.md                        # High-level setup, run commands, citation
├── requirements.txt                 # Python packages (torch==2.12.1, etc.)
├── environment.yml                  # Full conda env (CUDA 11.8/12.4 mix)
├── AZ-Hashes-Domain-Class/          # CSVs of Android SHA256 hashes only
│   ├── README.md
│   ├── AZ-Class-100_VT4.csv         # 285,583 rows: family labels for hashes
│   └── AZ-Domain_2008_2016.csv      # 758,447 rows: month/family/label for hashes
├── repo_imgs/                       # PNG figures used in README
├── EMBER_Experiments/
│   ├── EMBER_Domain/                # Domain-IL (12 monthly tasks)
│   │   ├── GRS.py                   # Global Reservoir Sampling baseline
│   │   ├── MADAR_IFS.py             # MADAR (raw-feature Isolation Forest)
│   │   ├── MADAR_AWS.py             # MADARθ (activation-weight IF)
│   │   ├── ember_model.py, ember_utils.py, ember_pjr_utils.py
│   │   └── *.sh job scripts
│   ├── EMBER_Class/                 # Class-IL (11 tasks: 50 + 10×5 families)
│   │   ├── main.py                  # Entry point for GRS/IFS/AWS
│   │   ├── data.py, encoder.py, linear_nets.py, aws_encoder.py
│   │   ├── train_diversity.py       # Implements IFS/AWS replay selection
│   │   ├── train.py, evaluate.py, utils.py, continual_learner.py
│   │   ├── exemplars.py, replayer.py, callbacks.py
│   │   └── param_stamp.py, param_values.py
│   └── EMBER_Task/                  # Task-IL (20 tasks, 5 families each)
│       └── Same module set as EMBER_Class
└── AZ_Experiments/
    ├── AZ_Domain/                   # 9 yearly tasks
    │   ├── GRS.py, MADAR_IFS.py, MADAR_AWS.py
    │   ├── az_model.py, az_utils.py, ember_model.py
    │   └── *.sh
    ├── AZ_Class/                    # 11 tasks
    │   └── main.py + same support modules
    └── AZ_Task/                     # 20 tasks
        └── main.py + same support modules
```

### Entry-point scripts

| Scenario | EMBER | AZ | Command (from README / .sh files) |
|---|---|---|---|
| Domain-IL GRS | `EMBER_Domain/GRS.py` | `AZ_Domain/GRS.py` | `python GRS.py --memory_budget=${B}` |
| Domain-IL MADAR (IFS) | `EMBER_Domain/MADAR_IFS.py` | `AZ_Domain/MADAR_IFS.py` | `python MADAR_IFS.py --ifs_option=${O} --memory_budget=${B}` |
| Domain-IL MADARθ | `EMBER_Domain/MADAR_AWS.py` | `AZ_Domain/MADAR_AWS.py` | `python MADAR_AWS.py --ifs_option=${O} --memory_budget=${B}` |
| Class-IL GRS/IFS/AWS | `EMBER_Class/main.py` | `AZ_Class/main.py` | `python main.py --metrics --scenario=class --replay_config=${C} --ifs_option=${O} --memory_budget=${B}` |
| Task-IL GRS/IFS/AWS | `EMBER_Task/main.py` | `AZ_Task/main.py` | `python main.py --metrics --scenario=task --replay_config=${C} --ifs_option=${O} --memory_budget=${B}` |

For Class-/Task-IL the replay mode is effectively fixed to `"offline"` in `main.py` (`--replay=offline`), and `replay_config` selects `grs`, `ifs`, or `aws` inside `train_diversity.py`.

## 2. Data expected by the code

### EMBER
- **EMBER_Domain**: expects a directory with one sub-folder per month `2018-01` … `2018-12`, each containing `XY_train.npz` (keys `X_train`, `Y_train`, `Y_family_train`) and `XY_test.npz` (keys `X_test`, `Y_test`, `Y_family_test`).
- **EMBER_Class/Task**: expects a single directory with `XY_train.npz` and `XY_test.npz` containing 100 malware families; labels are class indices 0–99.
- Default hard-coded absolute paths in `main.py` point to `/home/msrahman3/IQSeC-Datasets/EMBER_Class_Task/top_classes_100`. Users must override with `--data-dir`.

### AZ
- **AZ_Domain**: expects `Domain_Transformed/` with files `{year}_Domain_AZ_Train_Transformed.npz` and `{year}_Domain_AZ_Test_Transformed.npz` for years 2008–2016.
- **AZ_Class/Task**: expects `Family_Transformed/` with `Family_AZ_Train_Transformed.npz` and `Family_AZ_Test_Transformed.npz`.
- Default hard-coded absolute paths point to `/home/mr6564/continual_research/AZ_Data/...`.

### Feature dimensions referenced
| Dataset | Paper | Code |
|---|---|---|
| EMBER | 2,381 | 2,381 (`input_features=2381`) |
| AZ-Domain | 1,789 | 1,789 (`input_features=1789` in AZ_Domain/GRS.py) |
| AZ-Class | 2,439 | 2,439 (`orig_feats_length=2439` in AZ_Class/main.py) |

## 3. Hugging Face dataset `IQSeC-Lab/MADAR`

Probed via public HuggingFace API endpoints without authentication/accepting terms.

- **Status:** public but **gated = "auto"**. Metadata and README are readable anonymously; actual data files return **HTTP 401 Unauthorized** for anonymous download.
- **License:** MIT.
- **Total storage:** ~20.1 GB (`usedStorage`: 22,084,962,446 bytes).
- **Splits seen via `/parquet` endpoint:** `train` and `test`, each split into 9 shards (the dataset server exposes a unified view, not the per-year/per-class files the repo code reads).

### Files listed by the repository tree API

| Path | Type | Approx. size |
|---|---|---|
| `README.md` | file | 2,512 B |
| `.gitattributes` | file | 4,005 B |
| `AZ_Class/AZ_Class_Train.npz` | file | 2,523,966,748 B |
| `AZ_Class/AZ_Class_Test.npz` | file | 280,450,144 B |
| `AZ_Domain/2008_AZ_train.parquet` … `2016_AZ_train.parquet` | files | ~65 MB – ~80 MB each |
| `AZ_Domain/2008_AZ_test.parquet` … `2016_AZ_test.parquet` | files | ~0.5 MB – ~9 MB each |
| `EMBER_Class/XY_train.npz` | file | 2,890,138,390 B |
| `EMBER_Class/XY_test.npz` | file | 321,132,222 B |
| `EMBER_Domain/EMBER2018_Domain.tar.bz2` | file | 3,197,478,724 B |
| `EMBER_Domain/EMBER2018_Domain_with_FamilyLabels.tar.bz2` | file | 1,412,282,603 B |

**Note:** The HF dataset layout does **not** match the directory layouts hard-coded in the repository scripts (e.g., monthly `.npz` folders for EMBER Domain, `{year}_Domain_AZ_Train_Transformed.npz` files). The data are present in different containers, so reproducing the paper’s scripts will require reformatting/extracting the HF archives or using the authors’ private preprocessed directories.

The HF `README.md` describes the benchmark at a high level (Domain-IL / Class-IL, EMBER-Domain, AZ-Domain) but does not document the exact preprocessing steps or file schemas needed by the code.

## 4. Notable deviations between code and paper

### 4.1 Missing data-preprocessing pipeline
- The paper says features were standardized with `StandardScaler` (using `partial_fit` for incremental updates) and AZ features were reduced with `VarianceThreshold(<0.001)`.
- The repository contains **no preprocessing scripts** to generate the expected `.npz`/`.parquet` files from raw EMBER / AndroZoo data. Only hash CSVs and the HF dataset are provided.

### 4.2 Hard-coded absolute paths
- `EMBER_Class/main.py`, `EMBER_Task/main.py`, `AZ_Class/main.py`, `AZ_Domain/GRS.py`, `AZ_Domain/MADAR_IFS.py`, `AZ_Domain/MADAR_AWS.py`, and several `.ipynb_checkpoints` files contain absolute paths to authors’ local machines (`/home/msrahman3/...`, `/home/mr6564/...`). These must be overridden with `--data-dir`.

### 4.3 GRS implementation does not use true reservoir sampling
- The paper defines GRS as sampling uniformly from a growing global pool, and if `β ≥ |P|` it uses all samples.
- The code in `EMBER_Domain/GRS.py::get_GRS_data` concatenates all previous months, then if `memory_budget < len(previous_Ys)` it randomly shuffles and takes the first `memory_budget` samples. It does **not** maintain a reservoir across tasks; each task re-samples from scratch from the accumulated pool.

### 4.4 Domain-IL MADAR does not exactly follow Algorithm 1’s budget split
- Algorithm 1 computes `β_M, β_G = β·γ`, then within each family splits `β_A, β_S = β_M·Ω` (with `Ω` controlling anomalous/similar split).
- In `EMBER_Domain/MADAR_IFS.py::IFS`, the code always sets `goodware_budget = malware_budget = ceil(β/2)` (i.e., `γ = 0.5`), ignoring a configurable `γ`. The anomalous/similar split is done by `IFS_Samples` with `v_choose_split = ceil(v_choose/2)`, which is equivalent to `α = 0.5` but the `Ω` parameter is not exposed.
- For AZ Domain, the paper says `γ = 0.9` because of the 9:1 goodware/malware imbalance, but the AZ_Domain MADAR scripts also hard-code `goodware_budget = malware_budget = ceil(β/2)` (`MADAR_IFS.py` line 157, `MADAR_AWS.py` line 289), contradicting the paper.

### 4.5 Class-/Task-IL replay uses only `replay_config=ifs` semantics partially
- In `EMBER_Class/train_diversity.py`, `IFS()` and `AWS()` always allocate the full `memory_budget` to malware families; there is **no goodware replay** for Class-/Task-IL. This matches the paper’s Class-/Task-IL description (only malware), but the code path is different from Domain-IL.
- The `replay_portion` argument is mostly ignored for IFS/AWS; only `grs` path uses it in Task-IL offline mode.

### 4.6 MADARθ layer selection
- Paper: `L_Ember = act4`, `L_AZ = act5`.
- Code: `EMBER_Domain/MADAR_AWS.py` uses `--layer` default `act4` and provides `fc4`/`fc4_bn` alternatives. AZ Domain `MADAR_AWS.py` was not fully inspected, but the EMBER class/task `aws_encoder.py` defines `Ember_MLP_Net_V2` with only four layers (`act4`), so the AZ-specific `act5` requirement is not obviously present in the AZ Class/Task code.

### 4.7 Batch-normalization handling in MADARθ
- Paper says BN is omitted during the forward pass for activation extraction to preserve heterogeneity.
- In `EMBER_Domain/MADAR_AWS.py`, the default model `Ember_MLP_Net` includes BN, and the activation hook is placed on `act4` **after** the BN+ReLU path. The code does have a `--batch_norm` flag, but the logic around line 542 uses `batch_norm == 'yes'` string comparison on a boolean/flag argument, which is likely buggy (`action="store_true"` produces `True`/`False`, never the string `'yes'`).

### 4.8 Class-/Task-IL training code ignores `--epoch` for the main loop
- `main.py` passes `iters=args.iters` to `train_cl`. Inside `train_diversity.py`, `iters_to_use = (len(train_dataset)/batch_size) * args.epoch` is computed in `EMBER_Class/train.py`, but `train_diversity.py` (used by main.py) loops exactly `range(1, iters+1)`. Thus the number of optimization steps per task is controlled by `--iters` (default 10,000), not by `--epoch` (default 30), contrary to README wording.

### 4.9 Random seeds / reproducibility
- The paper reports 10–15 runs with different random initializations.
- `EMBER_Class/main.py` hard-codes `args.seed = 42` after parsing, so **every run uses the same NumPy/Torch seed** unless the line is edited. The same pattern appears in AZ Class/Task `main.py`.
- In `data.py`, class ordering for the 100 selected classes is fixed by `np.random.seed(42)`; therefore repeated runs cannot vary the class composition, only model initialization (which is also seeded to 42).

### 4.10 Model architecture mismatch with paper’s “five fully-connected layers” for AZ
- The paper says AZ uses a five-fully-connected-layer MLP.
- `AZ_Domain/GRS.py` imports `Ember_MLP_Net` from `ember_model.py`, which has four hidden layers (2381→1024→512→256→128→1). The AZ-specific `az_model.py` exists but `GRS.py` does not use it by default.

### 4.11 Dependency / environment inconsistencies
- `requirements.txt` pins `torch==2.12.1` and `torchvision==0.27.1`, which are **future/non-existent PyTorch versions** at the time of this report (latest stable is 2.6.x/0.21.x). `environment.yml` lists `pytorch=2.12.1` and `pytorch-cuda=12.4`, while also installing `cuda-libraries=11.8.0`, creating a CUDA-version mismatch.
- `Pillow==12.2.0` and `tornado==6.5.7` are also versions that do not exist in current PyPI, suggesting the files were hand-edited or contain typos.

### 4.12 Results directories are hard-coded relative to repo root
- Domain scripts write results to `../../Results_Domain/`, `../../IFS_SavedModel/`, etc., assuming the user runs from a specific nested directory. README does not document this.

## 5. Practical reproducibility blockers

1. **Cannot download data files from HF without accepting the gated-access terms.** Anonymous metadata/file-list access works; actual `.npz`/`.parquet`/`.tar.bz2` downloads return 401. The user must log into Hugging Face and accept the dataset access conditions.
2. **HF file layout ≠ code layout.** Even after download, the repository scripts expect preprocessed `.npz` files in month/year folders, while HF stores consolidated `AZ_Class/*.npz`, `EMBER_Class/*.npz`, and tarballs for EMBER Domain.
3. **No preprocessing code** is provided to bridge the gap.
4. **Multiple hard-coded paths, seed overrides, and environment-file errors** mean the scripts will not run out-of-the-box.
5. **No training was performed** for this recon report.

## 6. Summary

The MADAR repository contains complete Python implementations for all six experimental settings (EMBER/AZ × Domain/Class/Task-IL) and shell job scripts, but it is currently a **research artifact rather than a turnkey reproducibility package**. The main gaps are: missing preprocessing pipeline, HF dataset layout mismatch, hard-coded paths, seed hard-coding, environment-file errors, and several deviations from the paper’s stated algorithmic details (especially the goodware/malware budget split in AZ Domain-IL and the BN handling in MADARθ).
