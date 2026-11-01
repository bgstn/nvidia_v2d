#!/usr/bin/env python3
"""Choose one final checkpoint per Tier 1 episode from several runs, by local 4096-world metrics.

Each metric difference is measured in units of a meaningful change (AUC 0.05, SP-SR and MP-SR 0.2,
MPPE 5 cm) and summed; the run with the higher total is chosen. Episodes without local metrics for
every run keep the first run.
Writes a checkpoints JSON for run_t2_tier1_eval.sh and prints the selection.

    python select_merged.py -o t2_tier1_merged_checkpoints.json t2_tier1 t2_tier1_switching20
"""

import argparse
import json
from pathlib import Path

OUTPUTS = Path("/fs04/scratch2/rz76/jzhang/v2d/video_to_data/robotic_grounding/flash_chord/outputs")
FINAL = "policy_349999104.safetensors"
# Metric: signed size of a meaningful improvement.
UNITS = {"add_auc": 0.05, "spider_sr_uncentered": 0.2, "maniptrans_sr": 0.2, "mppe_cm": -5.0}


def advantage(a: dict, b: dict) -> float:
    """How much better a is than b, in meaningful-change units."""
    return sum((a[k] - b[k]) / unit for k, unit in UNITS.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    selection = {}
    for episode in range(30):
        sequence = f"ep{episode:03d}"
        metrics = {}
        for run in args.runs:
            report = OUTPUTS / run / sequence / f"{FINAL}.metrics.json"
            if report.exists():
                metrics[run] = json.loads(report.read_text())["metrics"]
        chosen = args.runs[0]
        if len(metrics) == len(args.runs):
            for run in args.runs[1:]:
                if advantage(metrics[run], metrics[chosen]) > 0.0:
                    chosen = run
        checkpoint = OUTPUTS / chosen / sequence / FINAL
        if not checkpoint.exists():
            raise SystemExit(f"{sequence}: missing {checkpoint}")
        selection[str(episode)] = str(checkpoint)
        m = metrics.get(chosen)
        summary = (
            f"AUC {m['add_auc']:.3f}  SP-SR {m['spider_sr_uncentered']:.3f}  "
            f"MP-SR {m['maniptrans_sr']:.3f}  MPPE {m['mppe_cm']:.1f}"
            if m
            else "no local metrics"
        )
        print(f"{sequence}  {chosen:24s} {summary}")

    Path(args.output).write_text(json.dumps(selection, indent=1) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
