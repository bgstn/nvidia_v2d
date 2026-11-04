# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score one checkpoint under fixed perturbed physics: contact friction, object mass and object size.

Runs the same evaluation as ``evaluate_policy.py``, but every world uses the given scales instead of
the checkpoint's saved physics (including any domain randomization it was trained with). Scales of 1
reproduce the nominal evaluation. For robustness studies only; leaderboard submissions use the
unmodified evaluator.

    python scripts/evaluate_robustness.py outputs/run/ep023/policy_449998848.safetensors \\
        --friction 0.5 --object-mass 1.5 --world-count 1024 -o robust/ep023_f0.5_m1.5.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from flash_chord.evaluation import harness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--friction", type=float, default=1.0, help="contact friction scale")
    parser.add_argument("--object-mass", type=float, default=1.0, help="object mass and inertia scale")
    parser.add_argument("--object-scale", type=float, default=1.0, help="isotropic object size scale")
    parser.add_argument("--world-count", type=int, default=1024)
    parser.add_argument("-o", "--output", required=True, help="metrics JSON")
    args = parser.parse_args()

    compose = harness.apply_evaluation_composition

    def compose_with_physics(config, **kwargs):
        config = compose(config, **kwargs)
        config["env"]["domain_randomization"] = {
            "_target_": "flash_chord.runtime.randomization.DomainRandomizationConfig",
            "friction_scale": [args.friction, args.friction],
            "object_mass_scale": [args.object_mass, args.object_mass],
            "seed": 0,
        }
        config["scene"]["object_scale_min"] = args.object_scale
        config["scene"]["object_scale_max"] = args.object_scale
        return config

    harness.apply_evaluation_composition = compose_with_physics
    outcome = harness.evaluate_checkpoint(Path(args.checkpoint).resolve(), world_count=args.world_count)
    scalars = harness.evaluation_scalars(outcome)
    physics = {"friction": args.friction, "object_mass": args.object_mass, "object_scale": args.object_scale}
    print(json.dumps({**physics, **scalars}, indent=2, sort_keys=True), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    harness.write_evaluation_report(output, outcome)
    output.with_suffix(".physics.json").write_text(json.dumps(physics) + "\n")


if __name__ == "__main__":
    main()
