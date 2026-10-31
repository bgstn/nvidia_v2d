#!/usr/bin/env python3
"""Convert V2D Challenge Track 2 episodes (LeRobot v2.1) into the folder layout
that ``scripts/retarget/soma_to_g1.py`` consumes.

For each episode it writes::

    <motion_root>/whole_body/<subdir>/sequence_id=<seq>/
        soma_params.npz                      # SOMA-X body params (read_soma.py schema)
        poses.npy                            # (T, 4, 4) world_T_object, OpenCV world axes
        ground_plane.json                    # {"plane": [a, b, c, d]} (if found)
        reconstructed_mesh/output_aligned.glb  # object mesh (if found)
    <motion_root>/whole_body/reconstructed_stage/<seq>_support.usda   (if found)

Frames where ``observation.object.visible`` is False carry an all-zero pose in the
dataset; they are filled by interpolating between the nearest visible frames
(linear position, slerp rotation) and held constant at the ends.

Run inside the Isaac Lab container (needs numpy, pandas, pyarrow, scipy, trimesh).

Examples::

    # See which non-parquet asset files the tier ships (meshes, planes, supports)
    python convert_track2_to_soma.py --tier_dir <.../track_2/tier_1_multiview_caption> --list_assets

    # Check the episode -> recording -> asset matching without writing anything
    python convert_track2_to_soma.py --tier_dir <...> --episodes all --dry_run

    # Convert one episode
    python convert_track2_to_soma.py --tier_dir <...> --episodes 0 --subdir t2_tier1

    # Convert all episodes
    python convert_track2_to_soma.py --tier_dir <...> --episodes all --subdir t2_tier1
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# Column names from the Track 2 meta/info.json.
COL_POSE = "observation.human.pose"                  # (231,) = 77 joints x rotvec
COL_TRANSL = "observation.human.translation"         # (3,)
COL_IDENTITY = "observation.human.identity_coeffs"   # (45,)
COL_SCALE = "observation.human.scale_params"         # (68,)
COL_BONES = "observation.human.bone_length_flexibles"  # (6,) not used by read_soma
COL_OBJ = "observation.object.pose"                  # (7,) x y z qw qx qy qz
COL_VIS = "observation.object.visible"               # bool

# SOMA world = OpenCV world rotated 180 deg about x (see _R_CV_TO_SOMA in soma_to_g1.py).
F_CV_SOMA = np.diag([1.0, -1.0, -1.0])

MESH_EXTS = {".glb", ".gltf", ".obj", ".ply", ".stl"}
SKIP_DIRS = {"data", "videos", "meta"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_info(tier_dir: Path) -> dict:
    return json.loads((tier_dir / "meta" / "info.json").read_text())


def load_tasks(tier_dir: Path, info: dict) -> dict[int, str]:
    tasks = {}
    jl = tier_dir / "meta" / "tasks.jsonl"
    if jl.is_file():
        for line in jl.read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                tasks[int(t["task_index"])] = t["task"]
    for t in info.get("tasks", []) or []:
        if isinstance(t, dict) and "task_index" in t:
            tasks.setdefault(int(t["task_index"]), t.get("task", ""))
    return tasks


def load_episode_df(tier_dir: Path, info: dict, ep: int) -> pd.DataFrame:
    pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunk = ep // int(info.get("chunks_size", 1000))
    f = tier_dir / pattern.format(episode_chunk=chunk, episode_index=ep)
    if f.is_file():
        df = pd.read_parquet(f)
    else:  # fall back to scanning all parquet files
        frames = [pd.read_parquet(p) for p in sorted((tier_dir / "data").rglob("*.parquet"))]
        df = pd.concat(frames, ignore_index=True)
        df = df[df["episode_index"] == ep]
    if len(df) == 0:
        raise ValueError(f"episode {ep}: no rows found")
    return df.sort_values("frame_index").reset_index(drop=True)


def stack(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in df[col].to_numpy()])


def joint_names_from_info(info: dict) -> list[str]:
    names = info["features"][COL_POSE].get("names")
    if isinstance(names, dict):  # some LeRobot writers nest names
        names = next(iter(names.values()))
    joints = []
    for n in names:
        j = n.rsplit(".", 1)[0]
        if not joints or joints[-1] != j:
            joints.append(j)
    return joints


# --------------------------------------------------------------------------- #
# Object poses
# --------------------------------------------------------------------------- #
def object_poses(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (T, 4, 4) world_T_object with invisible frames filled, and the visible mask."""
    raw = stack(df, COL_OBJ)  # (T, 7)
    vis = np.array(df[COL_VIS].to_numpy(), dtype=bool).reshape(-1)
    vis &= np.linalg.norm(raw[:, 3:7], axis=1) > 1e-6  # zero quaternion = unusable
    T = len(raw)
    idx = np.flatnonzero(vis)
    if len(idx) == 0:
        raise ValueError("object is never visible in this episode")

    pos = raw[:, :3].copy()
    quat_xyzw = raw[:, [4, 5, 6, 3]]  # wxyz -> xyzw for scipy
    t = np.arange(T)
    for k in range(3):
        pos[:, k] = np.interp(t, idx, raw[idx, k])  # holds ends constant
    rot_vis = R.from_quat(quat_xyzw[idx])
    if len(idx) == 1:
        rots = R.from_quat(np.repeat(quat_xyzw[idx], T, axis=0))
    else:
        rots = Slerp(idx, rot_vis)(np.clip(t, idx[0], idx[-1]))

    M = np.tile(np.eye(4), (T, 1, 1))
    M[:, :3, :3] = rots.as_matrix()
    M[:, :3, 3] = pos
    return M, vis


# --------------------------------------------------------------------------- #
# Assets (meshes, ground planes, support surfaces)
# --------------------------------------------------------------------------- #
def list_asset_files(tier_dir: Path) -> list[Path]:
    out = []
    for p in sorted(tier_dir.rglob("*")):
        rel = p.relative_to(tier_dir)
        if p.is_file() and rel.parts[0] not in SKIP_DIRS:
            out.append(p)
    return out


DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[A-Za-z0-9_]+")
STOP = {"the", "a", "an", "to", "and", "with", "your", "you", "of", "on", "in", "from",
        "up", "it", "its", "then", "over", "both", "hands", "hand", "using", "proceed", "when",
        "within", "area", "walk", "step", "away", "complete", "place", "pick", "into", "onto"}


def load_episode_meta(tier_dir: Path) -> dict[int, dict]:
    meta = {}
    jl = tier_dir / "meta" / "episodes.jsonl"
    if jl.is_file():
        for line in jl.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                meta[int(e["episode_index"])] = e
    return meta


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def recording_names(files: list[Path]) -> list[str]:
    names = set()
    for p in files:
        m = DATE_RE.match(p.stem)
        if m:
            names.add(re.sub(r"_+(support|ground_plane|ground|plane|mesh|object)$", "",
                             m.group(0)).rstrip("_"))
    return sorted(names)


def words(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", text.lower()) if w and w not in STOP and not w.isdigit()}


def overlap(task_words: set[str], rec: str) -> int:
    """Count task words matching recording words, allowing prefix forms (sweep ~ sweeping)."""
    rw = words(rec.split("_", 2)[-1])
    return sum(any(t == r or (min(len(t), len(r)) >= 4 and (t.startswith(r) or r.startswith(t)))
                   for t in task_words) for r in rw)


def episode_tasks(tier_dir: Path, info: dict, tasks: dict, ep_meta: dict, eps: list[int]) -> dict[int, str]:
    out = {}
    for ep in eps:
        e = ep_meta.get(ep, {})
        t = e.get("tasks")
        if t:
            out[ep] = t[0] if isinstance(t, list) else str(t)
            continue
        try:
            df0 = load_episode_df(tier_dir, info, ep)
            out[ep] = tasks.get(int(df0["task_index"].iloc[0]), "")
        except Exception:
            out[ep] = ""
    return out


def assign_recordings(ep_tasks: dict[int, str], ep_meta: dict, rec_names: list[str]) -> dict[int, tuple]:
    """Map every episode to a distinct recording.

    Names found in episodes.jsonl win. The rest are solved jointly as a one-to-one
    assignment maximizing task-text / recording-name word overlap.
    """
    result, free_eps = {}, []
    for ep in sorted(ep_tasks):
        rec, how = episode_recording(ep, ep_meta, "", rec_names)
        if rec is not None:
            result[ep] = (rec, how, None)
        else:
            free_eps.append(ep)
    taken = {r for r, _, _ in result.values()}
    free_recs = [r for r in rec_names if r not in taken]
    if free_eps and free_recs:
        from scipy.optimize import linear_sum_assignment
        S = np.array([[overlap(words(ep_tasks[e]), r) for r in free_recs] for e in free_eps])
        rows, cols = linear_sum_assignment(-S)
        for i, j in zip(rows, cols):
            row = np.sort(S[i])[::-1]
            margin = int(row[0] - row[1]) if len(row) > 1 else int(row[0])
            conf = "low" if S[i, j] == 0 else ("medium" if S[i, j] < row[0] else "ok")
            result[free_eps[i]] = (free_recs[j], f"task match: {S[i, j]} words, {conf}", margin)
    for ep in free_eps:
        result.setdefault(ep, (None, "no recording left to assign", None))
    return result


def episode_recording(ep: int, ep_meta: dict, task: str, rec_names: list[str]) -> tuple[str | None, str]:
    """Return (recording name, how it was found)."""
    for sval in _strings(ep_meta.get(ep, {})):
        m = DATE_RE.search(sval)
        if m:
            name = m.group(0)
            best = [r for r in rec_names if r.startswith(name) or name.startswith(r)]
            return (best[0] if best else name), "episodes.jsonl"
    if task and rec_names:  # fall back: match the task text against recording names
        tw = words(task)
        scored = sorted(((len(tw & words(r.split("_", 2)[-1])), r) for r in rec_names), reverse=True)
        if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            return scored[0][1], f"task text ({scored[0][0]} shared words)"
        return None, "task text ambiguous: " + ", ".join(r for _, r in scored[:3])
    return None, "no recording name found"


def find_asset(files: list[Path], rec: str | None, kind: str) -> Path | None:
    if rec is None:
        return None

    def is_kind(p: Path) -> bool:
        name = str(p).lower()
        if kind == "support":
            return p.suffix.lower() in {".usda", ".usd"} and "support" in name
        if kind == "ground":
            return p.suffix.lower() == ".json" and ("ground" in name or "plane" in name)
        if kind == "mesh":
            return p.suffix.lower() in MESH_EXTS and "support" not in name
        return False

    pick = [p for p in files if is_kind(p) and (p.stem.startswith(rec) or rec in p.parent.name)]
    if not pick:
        # assets stored per object (e.g. mesh/tall_bar_stool/...): try the longest
        # prefix of the recording's descriptive words that names a file or folder
        parts = rec.split("_", 2)[-1].split("_")
        for k in range(len(parts), 0, -1):
            obj = "_".join(parts[:k])
            if len(obj) < 4:
                break
            pick = [p for p in files if is_kind(p) and
                    (p.stem == obj or p.parent.name == obj or p.stem.startswith(obj + "_")
                     or p.stem.startswith(obj + "."))]
            if pick:
                break
    if len(pick) > 1:
        print(f"  [WARN] several {kind} candidates for {rec}; using {pick[0]}")
        for p in pick[1:4]:
            print(f"         also: {p}")
    return pick[0] if pick else None


def read_plane(path: Path) -> np.ndarray | None:
    d = json.loads(path.read_text())
    if isinstance(d, dict):
        for key in ("plane", "ground_plane", "coeffs", "abcd"):
            if key in d and len(np.ravel(d[key])) == 4:
                p = np.asarray(d[key], dtype=float).ravel()
                break
        else:
            n = d.get("normal")
            off = d.get("offset", d.get("d"))
            if n is None or off is None:
                return None
            p = np.array([*np.ravel(n), float(off)], dtype=float)
    elif isinstance(d, list) and len(np.ravel(d)) == 4:
        p = np.asarray(d, dtype=float).ravel()
    else:
        return None
    return p / np.linalg.norm(p[:3])


# --------------------------------------------------------------------------- #
# Human frame check
# --------------------------------------------------------------------------- #
def choose_human_frame(transl: np.ndarray, obj_pos_cv: np.ndarray, plane: np.ndarray | None,
                       override: str) -> str:
    """Decide whether the human params are in SOMA world axes or OpenCV world axes.

    The object pose is documented as OpenCV world. soma_to_g1.py expects SOMA
    params in SOMA axes (OpenCV rotated 180 deg about x). We test both hypotheses:
    the correct one puts the pelvis on the same side of the floor as the object.
    """
    if override != "auto":
        return override
    if plane is None:
        print("  [WARN] no ground plane: cannot auto-check human frame; assuming 'soma'. "
              "Verify with the replay video (box should be near the hands).")
        return "soma"
    n, d = plane[:3], plane[3]
    up = np.sign(np.median(obj_pos_cv @ n + d)) or 1.0  # objects sit above the floor
    h_soma = np.median((transl @ F_CV_SOMA.T) @ n + d) * up  # human given in SOMA axes
    h_cv = np.median(transl @ n + d) * up                    # human given in OpenCV axes
    print(f"  frame check: pelvis height above floor if SOMA axes {h_soma:+.3f} m, "
          f"if OpenCV axes {h_cv:+.3f} m (expect roughly +0.8..+1.1)")
    err_soma, err_cv = abs(h_soma - 0.95), abs(h_cv - 0.95)
    if min(err_soma, err_cv) < 0.6 and abs(err_soma - err_cv) > 0.3:
        return "soma" if err_soma < err_cv else "cv"
    print("  [WARN] frame check inconclusive; assuming 'soma'. Verify with the replay video.")
    return "soma"


# --------------------------------------------------------------------------- #
# Main conversion
# --------------------------------------------------------------------------- #
def convert_episode(tier_dir: Path, info: dict, tasks: dict, files: list[Path], ep: int,
                    out_root: Path, subdir: str, args, assignment: dict) -> Path:
    df = load_episode_df(tier_dir, info, ep)
    T = len(df)
    task = tasks.get(int(df["task_index"].iloc[0]), "") if "task_index" in df else ""
    rec, how = (args.recording, "--recording") if args.recording else assignment[ep][:2]
    seq = f"{args.prefix}{ep:03d}"
    print(f"\n=== episode {ep} -> {seq} ({T} frames) ===")
    if task:
        print(f"  task: {task}")
    print(f"  recording: {rec}  [{how}]")
    if rec is None:
        raise ValueError("cannot tell which recording this episode is; pass --recording <name>")

    # --- human -------------------------------------------------------------
    joints = joint_names_from_info(info)
    J = len(joints)
    poses = stack(df, COL_POSE).reshape(T, J, 3)
    transl = stack(df, COL_TRANSL)
    identity = stack(df, COL_IDENTITY)
    scale = stack(df, COL_SCALE)
    for name, arr in (("identity_coeffs", identity), ("scale_params", scale)):
        if np.abs(arr - arr[0]).max() > 1e-4:
            print(f"  [WARN] {name} varies over time; read_soma uses frame 0 only")

    # --- object ------------------------------------------------------------
    obj_M, vis = object_poses(df)
    print(f"  object visible in {vis.sum()}/{T} frames "
          f"({T - vis.sum()} filled by interpolation)")

    # --- assets --------------------------------------------------------------
    mesh = Path(args.mesh) if args.mesh else find_asset(files, rec, "mesh")
    ground = Path(args.ground_plane) if args.ground_plane else find_asset(files, rec, "ground")
    support = Path(args.support) if args.support else find_asset(files, rec, "support")
    plane = read_plane(ground) if ground else None
    if ground and plane is None:
        print(f"  [WARN] could not parse ground plane {ground}; skipping it")

    frame = choose_human_frame(transl, obj_M[vis, :3, 3], plane, args.human_frame)
    if frame == "cv":  # convert human root to SOMA axes
        transl = transl @ F_CV_SOMA.T
        root = R.from_rotvec(poses[:, 0])
        poses[:, 0] = (R.from_matrix(np.broadcast_to(F_CV_SOMA, (T, 3, 3))) * root).as_rotvec()
    print(f"  human params treated as '{frame}' axes")

    # --- write -------------------------------------------------------------
    seq_dir = out_root / "whole_body" / subdir / f"sequence_id={seq}"
    seq_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        seq_dir / "soma_params.npz",
        poses=poses.astype(np.float32),
        transl=transl.astype(np.float32),
        joint_names=np.array(joints),
        identity_model_type=np.array(args.identity_model_type),
        identity_coeffs=identity.astype(np.float32),
        scale_params=scale.astype(np.float32),
        bone_length_flexibles=stack(df, COL_BONES).astype(np.float32),
        unit=np.array("meters"),
        keep_root=np.array(False),
        rotation_repr=np.array("rotvec"),
        absolute_pose=np.array(False),
    )
    np.save(seq_dir / "poses.npy", obj_M)

    if plane is not None:
        (seq_dir / "ground_plane.json").write_text(
            json.dumps({"plane": plane.tolist(), "source": str(ground), "recording": rec}, indent=2))
    else:
        print("  [WARN] no ground plane written; soma_to_g1 will fall back to the body mesh")

    if mesh:
        mdir = seq_dir / "reconstructed_mesh"
        mdir.mkdir(exist_ok=True)
        dst = mdir / "output_aligned.glb"
        if mesh.suffix.lower() == ".glb":
            shutil.copy2(mesh, dst)
        else:
            import trimesh
            trimesh.load(mesh, force="scene").export(dst)
        print(f"  mesh: {mesh}")
    else:
        print("  [ERROR] no object mesh found; pass --mesh <file>. soma_to_g1 needs "
              "reconstructed_mesh/output_aligned.glb")

    if support:
        sdir = out_root / "whole_body" / "reconstructed_stage"
        sdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(support, sdir / f"{seq}_support.usda")
        print(f"  support: {support}")
    else:
        print("  [WARN] no support surface found; the object may start with nothing under it")

    fps = 1.0 / np.median(np.diff(df["timestamp"].to_numpy())) if "timestamp" in df else info.get("fps", 30)
    print(f"  wrote {seq_dir}")
    print(f"  fps = {fps:.2f}  -> pass --fps {fps:.0f} to soma_to_g1.py")
    return seq_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier_dir", required=True, help=".../track_2/tier_1_multiview_caption")
    ap.add_argument("--episodes", default="0", help="'all', '3', or '0,4,7'")
    ap.add_argument("--subdir", default="t2_tier1", help="folder under whole_body/")
    ap.add_argument("--prefix", default="ep", help="sequence id prefix (ep -> ep000)")
    ap.add_argument("--motion_root",
                    default="/workspace/video_to_data/robotic_grounding/source/robotic_grounding/"
                            "robotic_grounding/assets/human_motion_data")
    ap.add_argument("--human_frame", choices=["auto", "soma", "cv"], default="auto")
    ap.add_argument("--identity_model_type", default="mhr")
    ap.add_argument("--mesh", help="override object mesh path (single episode)")
    ap.add_argument("--ground_plane", help="override ground plane json (single episode)")
    ap.add_argument("--support", help="override support .usda (single episode)")
    ap.add_argument("--recording", help="override recording name (single episode)")
    ap.add_argument("--list_assets", action="store_true", help="list non-data files and exit")
    ap.add_argument("--dry_run", action="store_true",
                    help="print episode -> recording -> asset matches without writing")
    args = ap.parse_args()

    tier_dir = Path(args.tier_dir).resolve()
    files = list_asset_files(tier_dir)
    if args.list_assets:
        for p in files:
            print(p.relative_to(tier_dir))
        print(f"\n{len(files)} files")
        return

    info = load_info(tier_dir)
    tasks = load_tasks(tier_dir, info)
    ep_meta = load_episode_meta(tier_dir)
    rec_names = recording_names(files)
    n = int(info.get("total_episodes", 0))
    eps = list(range(n)) if args.episodes == "all" else [int(e) for e in args.episodes.split(",")]
    if len(eps) > 1 and (args.mesh or args.ground_plane or args.support or args.recording):
        sys.exit("--mesh/--ground_plane/--support/--recording only make sense with one episode")

    all_eps = list(range(n))
    assignment = assign_recordings(episode_tasks(tier_dir, info, tasks, ep_meta, all_eps),
                                   ep_meta, rec_names)

    if args.dry_run:
        if ep_meta:
            print("episodes.jsonl, first entry:", json.dumps(ep_meta[min(ep_meta)])[:400])
        print(f"{len(rec_names)} recording names found among asset files\n")
        short = lambda p: p.relative_to(tier_dir) if p else "-"
        for ep in eps:
            rec, how, _ = assignment[ep]
            m, g, sp = (find_asset(files, rec, k) for k in ("mesh", "ground", "support"))
            print(f"ep {ep:3d}: {rec}  [{how}]")
            print(f"        mesh={short(m)}  ground={short(g)}  support={short(sp)}")
        used = [a[0] for a in assignment.values() if a[0]]
        if len(used) != len(set(used)):
            print("\n[WARN] some recordings are assigned to more than one episode")
        return

    ok = []
    for ep in eps:
        try:
            ok.append(convert_episode(tier_dir, info, tasks, files, ep,
                                      Path(args.motion_root), args.subdir, args,
                                      assignment))
        except Exception as e:  # keep going over the other episodes
            print(f"  [ERROR] episode {ep}: {type(e).__name__}: {e}")
    print(f"\nConverted {len(ok)}/{len(eps)} episodes.")


if __name__ == "__main__":
    main()
