cd $REPO/robotic_grounding/flash_chord && source .venv/bin/activate
EP=$REPO/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/t2_tier1/sequence_id=ep013/robot_name=g1/data.parquet


python scripts/train_flash_sac.py experiment=g1_recon_body_flash_sac \
  "task.parquet='${EP}'" reset.seed=42 logging.mode=disabled \
  scene.world_count=256 training.total_environment_steps=2048000 \
  output_dir=outputs/g1_ep013_smoke
