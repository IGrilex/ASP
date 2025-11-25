#!/usr/bin/env python3
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, TYPE_CHECKING, Optional

import numpy as np

from paths import DATA_POSES_DOCKER, DATA_AIRSLAM_ROOT, DATA_2D_TEST, DATA_POSES

if TYPE_CHECKING:
    from metrics import Trajectory


# ============================================================
# Small generic helpers
# ============================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def tum_from_T(T_cw: np.ndarray) -> Tuple[List[float], List[float]]:
    """
    Convert ORB-SLAM3 pose T_cw (cam-from-world) to:
      - position t_wc (world-from-cam)
      - quaternion [qx,qy,qz,qw]
    """
    assert T_cw.shape == (4, 4)
    R_cw = T_cw[:3, :3].astype(float)
    t_cw = T_cw[:3, 3].astype(float)

    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw

    R = R_wc
    t = t_wc

    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
            qw = (R[2, 1] - R[1, 2]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
            qw = (R[0, 2] - R[2, 0]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
            qw = (R[1, 0] - R[0, 1]) / s

    return t.tolist(), [qx, qy, qz, qw]


def rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion [qx,qy,qz,qw]."""
    assert R.shape == (3, 3)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
            qw = (R[2, 1] - R[1, 2]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
            qw = (R[0, 2] - R[2, 0]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
            qw = (R[1, 0] - R[0, 1]) / s

    return qx, qy, qz, qw


def state_name(code: int) -> str:
    """Text labels for ORB-SLAM3 tracking states."""
    names = {
        -1: "SYSTEM_NOT_READY",
        0: "NO_IMAGES_YET",
        1: "NOT_INITIALIZED",
        2: "OK",
        3: "RECENTLY_LOST",
        4: "LOST",
        5: "OK_KLT",
    }
    return names.get(int(code), str(code))

# ============================================================
# Logging helpers for SLAM runners
# ============================================================

def log_phase1_config(
    algo_name: str,
    mode: str,
    seq_name: str,
    drive_name: str,
    dataroot: Path,
    saving_dir: Path,
    cfg_paths: Dict[str, str] | None = None,
    data_dirs: Dict[str, Path] | None = None,
) -> None:
    """
    Pretty-print Phase 1 configuration for a SLAM run.

    algo_name: e.g. "AirSLAM", "ORB-SLAM3"
    mode:      e.g. "STEREO", "MONO"
    """
    log("\n=======================================================")
    log(f"[PHASE1] {algo_name} ({mode}) on {seq_name} / {drive_name}")
    log("=======================================================")
    log(f"[cfg] dataroot           : {dataroot}")
    log(f"[cfg] saving_dir         : {saving_dir}")

    if cfg_paths:
        for key, val in cfg_paths.items():
            log(f"[cfg] {key:<19}: {val}")
    if data_dirs:
        log("")
        for key, path in data_dirs.items():
            log(f"[cfg] {key:<19}: {path}")


def ensure_dirs_exist(dirs: Dict[str, Path]) -> None:
    """
    Assert that all given directories exist.

    dirs: mapping from human-readable label -> Path
    """
    for name, p in dirs.items():
        assert p.is_dir(), f"{name} not found: {p}"


def log_overall_summary(
    algo_name: str,
    mode: str,
    summaries: List[object],
) -> None:
    """
    Log an overall summary table for multiple sequences.

    Each item in 'summaries' is expected to have attributes:
      seq_name, drive_name, frames_dataset, frames_traj,
      wall_time, avg_fps, ate, rpe_short, rpe_long
    (duck-typed; no strict dependency on a specific dataclass).
    """
    if not summaries:
        log(f"\n=========== OVERALL SUMMARY ({algo_name}, {mode}) ===========")
        log("[WARN] No summaries to report.")
        return

    log(f"\n=========== OVERALL SUMMARY ({algo_name}, {mode}) ===========")
    for s in summaries:
        ate_obj = getattr(s, "ate", None)
        rpe_s_obj = getattr(s, "rpe_short", None)
        rpe_l_obj = getattr(s, "rpe_long", None)

        ate_rmse = f"{ate_obj.rmse:.3f}" if ate_obj is not None else "N/A"
        rpe_s = (
            f"{rpe_s_obj.trans_rmse:.3f}m/{rpe_s_obj.rot_rmse_deg:.2f}deg"
            if rpe_s_obj is not None else "N/A"
        )
        rpe_l = (
            f"{rpe_l_obj.trans_rmse:.3f}m/{rpe_l_obj.rot_rmse_deg:.2f}deg"
            if rpe_l_obj is not None else "N/A"
        )

        log(
            f"{s.seq_name} / {s.drive_name}: "
            f"frames_dataset={s.frames_dataset}, "
            f"frames_traj={s.frames_traj}, "
            f"time={s.wall_time:.2f}s, "
            f"fps={s.avg_fps:.2f}, "
            f"ATE_rmse={ate_rmse}, "
            f"RPE_short={rpe_s}, "
            f"RPE_long={rpe_l}"
        )


# New helper: centralized SLAM run summary logger
def log_slam_summary(
    seq_name: str,
    drive_name: str,
    frames_total: int,
    tum_rows_count: int,
    num_pose_changes: int,
    elapsed: float,
    avg_fps: float,
    sys_summary: Dict[str, float],
    first_ok_frame: Optional[int],
    last_ok_frame: Optional[int],
    title: Optional[str] = None,
) -> None:
    """
    Log a concise SLAM summary (frames, poses, map/system stats).

    title: optional custom header line (if None a default is used).
    """
    if title is None:
        title = "SUMMARY (SLAM ONLY)"
    log(f"========== {title} ==========")
    log(f"[sum] seq / drive          : {seq_name} / {drive_name}")
    log(f"[sum] frames_total         : {frames_total}")
    log(f"[sum] TUM poses (st==OK)   : {tum_rows_count}")
    log(f"[sum] pose_changes (!=prev): {num_pose_changes}")
    log(f"[sum] elapsed (wall)       : {elapsed:.2f}s")
    log(f"[sum] avg_fps (wall)       : {avg_fps:.2f}")
    if sys_summary:
        log(f"[sum] cpu_avg            : {sys_summary.get('cpu_avg', -1):.1f}%")
        log(f"[sum] cpu_max            : {sys_summary.get('cpu_max', -1):.1f}%")
        log(f"[sum] ram_avg            : {sys_summary.get('ram_avg', -1):.1f}%")
        log(f"[sum] ram_max            : {sys_summary.get('ram_max', -1):.1f}%")
        if "gpu_avg" in sys_summary:
            log(f"[sum] gpu_avg          : {sys_summary.get('gpu_avg', -1):.1f}%")
            log(f"[sum] gpu_max          : {sys_summary.get('gpu_max', -1):.1f}%")
    if first_ok_frame is None:
        log("[sum] WARNING: ORB-SLAM3 never reached state=2 (OK).")
    else:
        log(
            f"[sum] OK frames from {first_ok_frame} to {last_ok_frame}, "
            f"total {last_ok_frame - first_ok_frame + 1 if last_ok_frame is not None else 0}"
        )
    log("=========================================================")


# ============================================================
# KITTI-360 generic helpers
# ============================================================

def parse_kitti360_timestamp(ts: str) -> float:
    """
    Parse KITTI-360 timestamp string to seconds since epoch.

    Example: '2013-05-28 13:52:23.123456789'
    """
    ts = ts.strip()
    if not ts:
        raise ValueError("Empty timestamp line")

    date_str, time_str = ts.split(" ")
    if "." in time_str:
        main, frac = time_str.split(".")
        frac = (frac + "000000000")[:9]
    else:
        main, frac = time_str, "000000000"

    dt = datetime.strptime(f"{date_str} {main}", "%Y-%m-%d %H:%M:%S")
    base = dt.timestamp()
    return base + int(frac) * 1e-9


def load_kitti360_cam0_to_world_as_traj(
    cam0_to_world_path: Path,
    fps: float = 10.0,  # kept for backwards compat, not really used now
) -> "Trajectory":
    """
    Load KITTI-360 cam0_to_world.txt into a Trajectory.

    Timestamps are loaded from:
      1) <drive>/timestamps.txt next to cam0_to_world.txt
      2) fallback: data_2d_test_slam/test_*/<drive>/image_00/timestamps.txt
    """
    from metrics import Trajectory  # local import to avoid circular deps

    cam0_dir = cam0_to_world_path.parent
    drive_name = cam0_dir.name  # e.g. '2013_05_28_drive_0008_sync'

    ts_path = cam0_dir / "timestamps.txt"

    if not ts_path.exists():
        # fallback to 2D test layout
        try:
            from paths import DATA_2D_TEST  # type: ignore
        except ImportError:
            DATA_2D_TEST = None  # type: ignore

        if DATA_2D_TEST is not None:
            for seq_dir in DATA_2D_TEST.glob("test_*"):
                cand = seq_dir / drive_name / "image_00" / "timestamps.txt"
                if cand.exists():
                    ts_path = cand
                    break

    if not ts_path.exists():
        raise FileNotFoundError(
            f"Pose timestamps not found next to {cam0_to_world_path} "
            f"or in data_2d_test_slam/*/{drive_name}/image_00/timestamps.txt"
        )

    # Load absolute timestamps
    ts_abs: List[float] = []
    with ts_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ts_abs.append(parse_kitti360_timestamp(line))

    if not ts_abs:
        raise RuntimeError(f"No timestamps in {ts_path}")

    # Relative times
    t0 = ts_abs[0]
    ts_rel = [t - t0 for t in ts_abs]

    # Read cam0_to_world
    data: List[Tuple[float, float, float, float]] = []
    with cam0_to_world_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 17:
                continue  # frame_idx + 16 values

            frame_idx = int(parts[0])

            # support both 0-based and 1-based indices
            if frame_idx < len(ts_rel):
                t = ts_rel[frame_idx]
            elif frame_idx - 1 < len(ts_rel):
                t = ts_rel[frame_idx - 1]
            else:
                continue

            vals = list(map(float, parts[1:]))
            T = np.array(vals, dtype=float).reshape(4, 4)
            xyz = T[:3, 3]

            data.append((t, xyz[0], xyz[1], xyz[2]))

    if not data:
        raise RuntimeError(f"No valid cam0_to_world data in {cam0_to_world_path}")

    arr = np.array(data, dtype=float)
    t_arr = arr[:, 0]
    xyz_arr = arr[:, 1:4]

    # Build quaternions from rotation matrices
    quat_list: List[List[float]] = []
    with cam0_to_world_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 17:
                continue
            vals = list(map(float, parts[1:]))
            T = np.array(vals, dtype=float).reshape(4, 4)
            R = T[:3, :3]
            qx, qy, qz, qw = rot_to_quat(R)
            quat_list.append([qx, qy, qz, qw])

    if len(quat_list) != xyz_arr.shape[0]:
        quat_arr = np.zeros((xyz_arr.shape[0], 4), dtype=float)
        quat_arr[:, 3] = 1.0
    else:
        quat_arr = np.array(quat_list, dtype=float)

    return Trajectory(t=t_arr, xyz=xyz_arr, quat=quat_arr)


# ============================================================
# Generic IO / geometry helpers
# ============================================================

def load_ply_xyz(path: Path) -> np.ndarray:
    """Load PLY and return Nx3 XYZ using open3d."""
    try:
        import open3d as o3d
    except ImportError as e:
        raise RuntimeError(
            "open3d is required to load PLY files. Install with 'pip install open3d'."
        ) from e

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=float)
    if pts.size == 0:
        return np.empty((0, 3), dtype=float)
    return pts


def voxelize_points(points: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize XYZ to voxels, return unique voxel coords and counts."""
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must be (N,3+) array of XYZ")

    coords = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    unique, counts = np.unique(coords, axis=0, return_counts=True)
    return unique, counts


# ============================================================
# AirSLAM / KITTI-360 dataset helpers
# ============================================================

def build_frameidx_to_timestamp_for_seq(seq_name: str) -> Dict[int, float]:
    """
    Build mapping frame_idx -> absolute timestamp (sec) using:

      /ASP/data/airslam_kitti360/<seq_name>/cam0/data/0000001482.png
      /ASP/data/airslam_kitti360/<seq_name>/cam0/timestamps.txt
    """
    cam0_dir = DATA_AIRSLAM_ROOT / seq_name / "cam0" / "data"
    ts_path = DATA_AIRSLAM_ROOT / seq_name / "cam0" / "timestamps.txt"

    assert cam0_dir.is_dir(), f"cam0/data not found: {cam0_dir}"
    assert ts_path.exists(), f"cam0/timestamps.txt not found: {ts_path}"

    img_files = sorted(cam0_dir.glob("*.png"))
    if not img_files:
        raise RuntimeError(f"No PNG images in {cam0_dir}")

    ts_lines: List[str] = []
    with ts_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                ts_lines.append(line)

    if len(ts_lines) != len(img_files):
        raise RuntimeError(
            f"Mismatch cam0 images ({len(img_files)}) vs timestamps ({len(ts_lines)}) "
            f"for {seq_name}"
        )

    frameidx_to_ts: Dict[int, float] = {}
    for img_path, ts_str in zip(img_files, ts_lines):
        stem = img_path.stem
        try:
            frame_idx = int(stem)
        except ValueError:
            raise RuntimeError(f"Cannot parse frame index from {img_path.name}")
        ts_abs = parse_kitti360_timestamp(ts_str)
        frameidx_to_ts[frame_idx] = ts_abs

    if not frameidx_to_ts:
        raise RuntimeError(f"No frame_idx -> timestamp mapping built for {seq_name}")

    log(f"[MAP] {seq_name}: frame_idx->timestamp for {len(frameidx_to_ts)} frames (cam0)")
    return frameidx_to_ts


def load_kitti360_poses3x4_as_traj_for_seq(
    seq_name: str,
    drive_name: str,
) -> "Trajectory":
    """
    Build GT trajectory for a test_* sequence using:

      - data_poses/<drive>/poses.txt (IMU->world, 3x4 [R|t])
      - airslam_kitti360/<seq>/cam0/... for frame subset + timestamps
    """
    from metrics import Trajectory  # local import to avoid circular deps

    drive_dir = DATA_POSES_DOCKER / drive_name

    poses_path = drive_dir / "poses.txt"

    if not poses_path.exists():
        raise FileNotFoundError(f"GT poses not found: {poses_path}")

    frameidx_to_ts = build_frameidx_to_timestamp_for_seq(seq_name)

    gt_entries: List[tuple[int, np.ndarray]] = []
    with poses_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 13:
                continue  # frame_idx + 12 values

            frame_idx = int(parts[0])
            if frame_idx not in frameidx_to_ts:
                continue

            vals = np.array(list(map(float, parts[1:])), dtype=float)
            T = vals.reshape(3, 4)  # [R | t]
            gt_entries.append((frame_idx, T))

    if not gt_entries:
        raise RuntimeError(
            f"No GT poses found for subset {seq_name} in {poses_path}; "
            f"did you build airslam_kitti360/{seq_name} correctly?"
        )

    gt_entries.sort(key=lambda x: x[0])

    ts_abs_list: List[float] = [frameidx_to_ts[fi] for fi, _ in gt_entries]
    t0 = ts_abs_list[0]

    t_list: List[float] = []
    xyz_list: List[np.ndarray] = []
    quat_list: List[List[float]] = []

    for (frame_idx, T), ts_abs in zip(gt_entries, ts_abs_list):
        R = T[:, :3]
        t_vec = T[:, 3]
        t_rel = ts_abs - t0

        qx, qy, qz, qw = rot_to_quat(R)

        t_list.append(t_rel)
        xyz_list.append(t_vec)
        quat_list.append([qx, qy, qz, qw])

    t_arr = np.array(t_list, dtype=float)
    xyz_arr = np.vstack(xyz_list).astype(float)
    quat_arr = np.vstack(quat_list).astype(float)

    log(
        f"[GT] {seq_name}: built GT traj with {len(t_arr)} poses "
        f"(poses.txt filtered by cam0 subset)"
    )

    return Trajectory(t=t_arr, xyz=xyz_arr, quat=quat_arr)


def load_image_gray(p: Path) -> np.ndarray:
    """Read image as grayscale, raising on failure."""
    import cv2  # keep local to avoid heavy global deps if unused
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise RuntimeError(f"Failed to read image: {p}")
    return im


# Add helper to parse vertex count from PLY header (fast, stops at end_header)
def get_ply_vertex_count(p: Path) -> Optional[int]:
    """
    Read the ASCII PLY header and return the integer vertex count if present.
    Works even if the PLY body is binary because the header is ASCII.
    """
    try:
        with p.open("r", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                # look for: "element vertex <N>"
                if len(parts) >= 3 and parts[0].lower() == "element" and parts[1].lower() == "vertex":
                    try:
                        return int(parts[2])
                    except Exception:
                        return None
                if s.lower() == "end_header":
                    break
    except Exception:
        return None
    return None


def build_frameidx_to_timestamp_from_2d(
    seq_name: str,
    drive_name: str,
) -> Dict[int, float]:
    """
    Build mapping frame_idx -> absolute timestamp (sec) using the 2D_TEST layout:
      data/data_2d_test_slam/<seq_name>/<drive_name>/image_00/data_rect/*.png
      .../image_00/timestamps.txt
    """
    img_dir = (
        DATA_2D_TEST
        / seq_name
        / drive_name
        / "image_00"
        / "data_rect"
    )
    ts_path = (
        DATA_2D_TEST
        / seq_name
        / drive_name
        / "image_00"
        / "timestamps.txt"
    )

    assert img_dir.is_dir(), f"image_00/data_rect not found: {img_dir}"
    assert ts_path.exists(), f"image_00/timestamps.txt not found: {ts_path}"

    img_files = sorted(img_dir.glob("*.png"))
    if not img_files:
        img_files = sorted(img_dir.glob("*.jpg"))
    if not img_files:
        img_files = sorted(img_dir.glob("*.jpeg"))

    if not img_files:
        raise RuntimeError(f"No images in {img_dir}")

    ts_lines: List[str] = []
    with ts_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                ts_lines.append(line)

    if len(ts_lines) != len(img_files):
        raise RuntimeError(
            f"Mismatch images ({len(img_files)}) vs timestamps ({len(ts_lines)}) "
            f"for {seq_name}/{drive_name}"
        )

    frameidx_to_ts: Dict[int, float] = {}
    for img_path, ts_str in zip(img_files, ts_lines):
        stem = img_path.stem
        try:
            frame_idx = int(stem)
        except ValueError:
            raise RuntimeError(f"Cannot parse frame index from {img_path.name}")
        ts_abs = parse_kitti360_timestamp(ts_str)
        frameidx_to_ts[frame_idx] = ts_abs

    log(
        f"[GT-map] {seq_name}/{drive_name}: frame_idx->timestamp for "
        f"{len(frameidx_to_ts)} frames (image_00)"
    )
    return frameidx_to_ts


def load_kitti360_poses3x4_as_traj_for_2d(
    seq_name: str,
    drive_name: str,
) -> "Trajectory":
    """
    Build GT trajectory using:
      - data/data_poses/<drive_name>/poses.txt (IMU->world, 3x4 [R|t])
      - frame_idx/time subset from DATA_2D_TEST image_00 for <seq_name>/<drive_name>
    Returns a metrics.Trajectory (local import to avoid circular import).
    """
    from metrics import Trajectory  # local import to avoid circular deps

    poses_path = (DATA_POSES / drive_name / "poses.txt").resolve()
    if not poses_path.exists():
        raise FileNotFoundError(f"GT poses not found: {poses_path}")

    frameidx_to_ts = build_frameidx_to_timestamp_from_2d(seq_name, drive_name)

    gt_entries: List[tuple[int, np.ndarray]] = []
    with poses_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 13:
                continue  # frame_idx + 12 values [3x4]

            frame_idx = int(parts[0])
            if frame_idx not in frameidx_to_ts:
                continue

            vals = np.array(list(map(float, parts[1:])), dtype=float)
            T = vals.reshape(3, 4)
            gt_entries.append((frame_idx, T))

    if not gt_entries:
        raise RuntimeError(
            f"No GT poses matched image subset {seq_name} in {poses_path}"
        )

    gt_entries.sort(key=lambda x: x[0])

    ts_abs_list: List[float] = [frameidx_to_ts[fi] for fi, _ in gt_entries]
    t0 = ts_abs_list[0]

    t_list: List[float] = []
    xyz_list: List[np.ndarray] = []
    quat_list: List[List[float]] = []

    for (frame_idx, T), ts_abs in zip(gt_entries, ts_abs_list):
        R = T[:, :3]
        t_vec = T[:, 3]
        t_rel = ts_abs - t0

        qx, qy, qz, qw = rot_to_quat(R)

        t_list.append(t_rel)
        xyz_list.append(t_vec)
        quat_list.append([qx, qy, qz, qw])

    t_arr = np.array(t_list, dtype=float)
    xyz_arr = np.vstack(xyz_list).astype(float)
    quat_arr = np.vstack(quat_list).astype(float)

    log(
        f"[GT] {seq_name}/{drive_name}: built GT traj with {len(t_arr)} poses "
        f"from poses.txt (image_00 subset)"
    )

    return Trajectory(t=t_arr, xyz=xyz_arr, quat=quat_arr)
