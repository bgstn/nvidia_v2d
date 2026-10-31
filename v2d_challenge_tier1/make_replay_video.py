src = "scripts/replay_motion.py"
dst = "scripts/replay_motion_video.py"
s = open(src).read()

def sub(old, new):
    global s
    assert s.count(old) == 1, f"anchor not unique/found: {old[:60]!r} - tell Claude"
    s = s.replace(old, new, 1)

# 1) CLI flags; recording forces cameras on
sub("AppLauncher.add_app_launcher_args(parser)\nargs_cli = parser.parse_args()\n",
"""parser.add_argument("--video_out", type=str, default=None,
                    help="Write an MP4 of the replay here (implies --no-loop).")
parser.add_argument("--video_fps", type=float, default=30.0, help="Output video fps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video_out:
    args_cli.enable_cameras = True
    args_cli.no_loop = True
""")

# 2) Frame the camera on the motion before building the env
sub("    cfg.scene.num_envs = args_cli.num_envs\n",
"""    cfg.scene.num_envs = args_cli.num_envs
    if args_cli.video_out:
        try:
            import os as _os
            sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "rsl_rl"))
            from viewer_utils import autoframe_viewer
            autoframe_viewer(cfg, args_cli.motion_file)
        except Exception as _e:
            print(f"[WARN] autoframe_viewer failed ({_e}); using default camera")
""")

# 3) Attach an RGB annotator to the viewer camera
sub("    env.reset()\n    device = env.device\n",
"""    env.reset()
    device = env.device
    _frames, _annot, _every = [], None, 1
    if args_cli.video_out:
        import numpy as _np
        import omni.replicator.core as rep
        _rp = rep.create.render_product(cfg.viewer.cam_prim_path, cfg.viewer.resolution)
        _annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        _annot.attach([_rp])
        _every = max(1, round((1.0 / cfg.sim.dt) / args_cli.video_fps))
        print(f"[INFO] Recording every {_every} sim steps -> {args_cli.video_out}")
""")

# 4) Grab a frame after each render
sub("        env.scene.update(dt=env.physics_dt)\n",
"""        env.scene.update(dt=env.physics_dt)
        if _annot is not None and step_idx % _every == 0:
            _d = _annot.get_data()
            _d = _np.frombuffer(_d, dtype=_np.uint8).reshape(*_d.shape)
            if _d.size:
                _frames.append(_d[:, :, :3].copy())
""")

# 5) Write the MP4
sub("    env.close()\n\n    print(\"[INFO] Kinematic",
"""    if args_cli.video_out and _frames:
        import os as _os
        import imageio
        _os.makedirs(_os.path.dirname(_os.path.abspath(args_cli.video_out)), exist_ok=True)
        imageio.mimwrite(args_cli.video_out, _frames[1:], fps=args_cli.video_fps)
        print(f"[INFO] Wrote {len(_frames) - 1} frames to {args_cli.video_out}")
    env.close()

    print("[INFO] Kinematic""")

open(dst, "w").write(s)
print("wrote", dst)
