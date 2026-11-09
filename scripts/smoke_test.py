"""M1 smoke test: EMBER Domain-IL, first 2 monthly tasks, 1 epoch, 1 seed.

Re-runs the exact M1 smoke (methods none + GRS + MADAR-R, budget 10K, seed 42)
through the patched official-script copies in src/ember_domain/ and prints a
summary from the per-task JSON records. Full-grid runs belong to M2 — this
script never runs more than --max_tasks tasks.

Usage:
    python scripts/smoke_test.py [--seed 42] [--budget 10000]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "ember_domain"
DATA = ROOT / "data" / "processed" / "ember_domain"


def run(script, out_dir, extra):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SRC / script), "--num_exps", "1", "--num_epoch", "1",
           "--seed", str(ARGS.seed), "--max_tasks", "2",
           "--data_dir", str(DATA) + "/", "--out_dir", str(out_dir)] + extra
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    recs = sorted(out_dir.glob("smoke_*.json"))
    for r in recs:
        d = json.loads(r.read_text())
        print(f"  {d['method']:10s} {d['task']} acc={d['acc']:.3f} ap={d['ap']:.3f} "
              f"wall={d['task_wallclock_min']:.2f}min rss={d['peak_rss_mb'] / 1024:.1f}GB", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budget", type=int, default=10000)
    ARGS = p.parse_args()
    run("GRS.py", ROOT / "smoke_out_none", ["--batch_size", "6000", "--memory_budget", "0"])
    run("GRS.py", ROOT / "smoke_out_grs10k",
        ["--batch_size", "6000", "--memory_budget", str(ARGS.budget)])
    run("MADAR_IFS.py", ROOT / "smoke_out_madarr10k",
        ["--batch_size", "2000", "--memory_budget", str(ARGS.budget),
         "--ifs_option", "ratio"])
    print("OK: 3 smoke runs x 2 tasks. See results/smoke_*.json and SMOKE_REPORT.md.")
