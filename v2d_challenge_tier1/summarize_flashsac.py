#!/usr/bin/env python3
"""Collect FlashSAC evaluation metrics for every episode of a tier into one table.

Usage (from robotic_grounding/flash_chord, venv active or any Python 3):
    python summarize_flashsac.py outputs/t2_tier1
    python summarize_flashsac.py outputs/t2_tier1 --csv tier1_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

KEYS = [
    ("AUC", "add_auc"),
    ("SP-SR", "spider_sr_uncentered"),
    ("MP-SR", "maniptrans_sr"),
    ("MPPE_cm", "mppe_cm"),
    ("chord_sr", "chord_sr"),
]


def flat(d, pre=""):
    out = {}
    for k, v in (d.items() if isinstance(d, dict) else []):
        key = f"{pre}/{k}" if pre else str(k)
        if isinstance(v, dict):
            out.update(flat(v, key))
        else:
            out[key] = v
    return out


def lookup(m, name):
    for k, v in m.items():
        if (k == name or k.endswith("/" + name)) and isinstance(v, (int, float)):
            return float(v)
    return float("nan")


def latest_metrics(ep_dir: Path) -> Path | None:
    files = list(ep_dir.glob("policy_*.safetensors.metrics.json"))
    if not files:
        return None
    return max(files, key=lambda p: int(re.search(r"policy_(\d+)", p.name).group(1)))


def status(ep_dir: Path) -> str:
    if latest_metrics(ep_dir):
        return "done"
    if list(ep_dir.glob("policy_*.safetensors")) or list(ep_dir.glob("state_*.safetensors")):
        return "training"
    return "not started"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tier_dir", help="e.g. outputs/t2_tier1")
    ap.add_argument("--csv", help="also write the table to this CSV file")
    args = ap.parse_args()

    rows = []
    for ep_dir in sorted(Path(args.tier_dir).glob("ep*")):
        row = {"episode": ep_dir.name, "status": status(ep_dir)}
        mf = latest_metrics(ep_dir)
        if mf:
            m = flat(json.loads(mf.read_text()))
            for name, key in KEYS:
                row[name] = lookup(m, key)
        rows.append(row)

    header = ["episode", "status"] + [n for n, _ in KEYS]
    print(" ".join(f"{h:>11}" for h in header))
    for r in rows:
        cells = [f"{r['episode']:>11}", f"{r['status']:>11}"]
        for name, _ in KEYS:
            v = r.get(name)
            cells.append(f"{v:>11.3f}" if isinstance(v, float) and not math.isnan(v) else f"{'-':>11}")
        print(" ".join(cells))

    done = [r for r in rows if r["status"] == "done"]
    print(f"\n{len(done)}/{len(rows)} episodes evaluated")
    if done:
        means = "  ".join(f"{n}={sum(r[n] for r in done) / len(done):.3f}" for n, _ in KEYS)
        print(f"mean over evaluated episodes: {means}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
