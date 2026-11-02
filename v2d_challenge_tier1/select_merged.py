#!/usr/bin/env python3
"""Choose one checkpoint per Tier 1 episode from several runs, by local multi-world metrics.

Candidates per run: its final policy (4096-world metrics), the earlier policies scored by
ckpt_select.sh (1024 worlds), and the final policy of its <run>_objft fine-tune (4096 worlds).
Each metric is measured in units of a meaningful change (AUC 0.05, SP-SR and MP-SR 0.2, MPPE 5 cm)
and summed; the candidate with the highest total is chosen. Episodes without local metrics for the
first run's final policy keep it.
Writes a checkpoints JSON for run_t2_tier1_eval.sh and prints the selection.

    python select_merged.py -o t2_tier1_merged_checkpoints.json t2_tier1 t2_tier1_switching20
"""

import argparse
import json
from pathlib import Path

OUTPUTS = Path("/fs04/scratch2/rz76/jzhang/v2d/video_to_data/robotic_grounding/flash_chord/outputs")
FINAL = "policy_349999104.safetensors"
FINETUNED = "policy_449998848.safetensors"
# Metric: signed size of a meaningful improvement.
UNITS = {"add_auc": 0.05, "spider_sr_uncentered": 0.2, "maniptrans_sr": 0.2, "mppe_cm": -5.0}


def score(metrics: dict) -> float:
    """Metrics in meaningful-change units, higher is better."""
    return sum(metrics[k] / unit for k, unit in UNITS.items())


def candidates(run: str, sequence: str):
    """(checkpoint, metrics report) pairs for one run and episode."""
    yield OUTPUTS / run / sequence / FINAL, OUTPUTS / run / sequence / f"{FINAL}.metrics.json"
    for report in sorted((OUTPUTS / "ckpt_select" / run / sequence).glob("policy_*.json")):
        yield OUTPUTS / run / sequence / f"{report.stem}.safetensors", report
    finetuned = OUTPUTS / f"{run}_objft" / sequence / FINETUNED
    yield finetuned, finetuned.with_name(f"{FINETUNED}.metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    selection = {}
    for episode in range(30):
        sequence = f"ep{episode:03d}"
        checkpoint, m = OUTPUTS / args.runs[0] / sequence / FINAL, None
        for run in args.runs:
            for candidate, report in candidates(run, sequence):
                if not (candidate.exists() and report.exists()):
                    continue
                metrics = json.loads(report.read_text())["metrics"]
                if m is None and candidate != checkpoint:
                    break  # no metrics for the first run's final policy: keep it
                if m is None or score(metrics) > score(m):
                    checkpoint, m = candidate, metrics
        if not checkpoint.exists():
            raise SystemExit(f"{sequence}: missing {checkpoint}")
        selection[str(episode)] = str(checkpoint)
        summary = (
            f"AUC {m['add_auc']:.3f}  SP-SR {m['spider_sr_uncentered']:.3f}  "
            f"MP-SR {m['maniptrans_sr']:.3f}  MPPE {m['mppe_cm']:.1f}"
            if m
            else "no local metrics"
        )
        print(f"{sequence}  {str(checkpoint.relative_to(OUTPUTS)):60s} {summary}")

    Path(args.output).write_text(json.dumps(selection, indent=1) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
