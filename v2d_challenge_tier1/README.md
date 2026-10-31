# V2D Challenge — Track 2 Tier 1 submission

Code that produced our Track 2 Tier 1 submission: one FlashSAC policy per episode
(G1 + Dex3, `experiment=g1_recon_body_flash_sac`), evaluated with the submission kit's
unmodified `flash_chord/scripts/evaluate_policy.py` (kit revision `8c778f9`, identical to
`robotic_grounding/flash_chord/scripts/evaluate_policy.py` on this branch).

The scripts were run from a working directory one level above this checkout
(`$SCRATCH/v2d`), on a Slurm cluster with A40 GPUs; paths in them are site-specific.

## Pipeline

| Step | Script | Notes |
|---|---|---|
| Environment | `env.sh`, `install_rg_deps.sh`, `flashsac_setup.sh` | Isaac Lab 2.3.2 Apptainer image for retargeting; uv venv for FlashCHORD |
| Convert Tier 1 input | `convert_track2_to_soma.py` | LeRobot Tier 1 episodes → SOMA layout (`sequence_id=epNNN`, NNN = `episode_index`) |
| Retarget to G1 | `convert_and_retarget_tier1.sh` | runs the converter, then `scripts/retarget/soma_to_g1.py` per episode |
| Train | `flashsac_tier1.sh` | Slurm array, one episode per task: `sbatch --array=0-29%4 flashsac_tier1.sh t2_tier1` |
| Local metrics | `summarize_flashsac.py` | 4096-world diagnostics; not used for the submission |
| Submission rollouts | `run_t2_tier1_eval.sh` | kit `examples/run_policy_evaluation.py`, 1 world, final checkpoint per episode (`t2_tier1_checkpoints.json`) |

Training uses `reset.seed=42` and the default 349,999,104 environment steps; the submitted
policy for every episode is `policy_349999104.safetensors`.

## Ground-only episodes

The converter writes a support-surface stage for every episode. For episodes whose released
support surface is empty, that stage was moved to
`robotic_grounding/.../reconstructed_stage/_empty_ground_only/` and the policy was trained
with `scene.support=false` (`flashsac_tier1.sh` detects this from the missing stage file).
The remaining `epNNN_support.usda` stages are committed alongside it.

## Not committed

- Retargeted motion (`whole_body/t2_tier1/`, 1.8 GB) — regenerate with
  `convert_and_retarget_tier1.sh`.
- Checkpoints and rollouts (`flash_chord/outputs/`) — checkpoint SHA-256s are recorded in the
  submission CSV.

`flashsac_tier1.sh` has an opt-in `EVAL_NJMAX` path pointing to a local diagnostic wrapper
(not in this repo). It was not used for any submitted result.
