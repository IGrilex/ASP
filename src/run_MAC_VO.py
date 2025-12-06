#!/usr/bin/env python3
from __future__ import annotations

import time
import subprocess
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from paths import (
    DATA_2D_TEST,
    DATA_POSES,
    OUTPUT_ROOT,          # ASP/output
    TEST_DRIVES,
    TEST_SCENARIOS,
)

from file_creator import (
    log,
    log_phase1_config,
    log_overall_summary,
    log_slam_summary,
)

from utils import (
    tum_from_T,
    parse_kitti360_timestamp,
    ensure_dirs_exist,
    load_kitti360_poses3x4_as_traj_for_2d,
)

from metrics import (
    ATEResult,
    RPEResult,
    Trajectory,
    load_tum_trajectory,
    compute_ate,
    compute_rpe,
    SystemSample,
    append_system_sample,
    summarize_system_metrics,
)

from file_creator import (
    write_ate_report,
    write_ate_detailed,
    write_rpe_report,
    write_rpe_detailed,
    write_dashboard,
)

# ============================================================
# Constants for MAC-VO paths (inside Docker)
# ============================================================

# MAC-VO repo root inside the container
# With docker run: -v ~/ASP-SLAM:/workspace
# this is /workspace/MAC-VO
MACVO_ROOT = Path("/workspace/MAC-VO")

# MAC-VO odometry config (Fast mode)
MACVO_ODOM_CFG = MACVO_ROOT / "Config" / "Experiment" / "MACVO" / "MACVO_Fast.yaml"

# Base / template GeneralStereo data config for KITTI-360
# We will CLONE + MODIFY this per sequence (test_0, test_1, ...)
MACVO_DATA_TEMPLATE = Path("/workspace/ASP/configs/MAC_VO.yaml")

# Output root for MAC-VO results in ASP project
# This will be /workspace/ASP/output/MAC_VO inside the container
OUT_MACVO_ROOT = OUTPUT_ROOT / "MAC_VO"


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class RunInfo:
    seq_name: str
    drive_name: str
    out_dir: Path
    frames_dataset: int
    frames_traj: int
    wall_time: float
    avg_fps: float
    num_map_points: Optional[int]
    sys_summary: Dict[str, float]


@dataclass
class RunSummary:
    seq_name: str
    drive_name: str
    frames_dataset: int
    frames_traj: int
    wall_time: float
    avg_fps: float
    num_map_points: Optional[int]
    ate: Optional[ATEResult]
    rpe_short: Optional[RPEResult]
    rpe_long: Optional[RPEResult]
    sys_summary: Dict[str, float]


# ============================================================
# Helpers
# ============================================================

def ensure_monitoring_deps() -> None:
    """
    Ensure that psutil and pynvml are installed in the current Python environment.

    Call this early in main(), e.g.:

        if __name__ == "__main__":
            ensure_monitoring_deps()
            main()

    so that system monitoring (CPU/RAM/GPU) works without manual setup.
    """
    import importlib
    import subprocess
    import sys

    required_pkgs = [
        ("psutil", "psutil"),
        ("pynvml", "pynvml"),
    ]

    for pkg_name, import_name in required_pkgs:
        try:
            importlib.import_module(import_name)
            print(f"[deps] {pkg_name} already installed.", flush=True)
        except ImportError:
            print(f"[deps] {pkg_name} not found. Installing with pip...", flush=True)
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg_name]
                )
                print(f"[deps] Successfully installed {pkg_name}.", flush=True)
            except Exception as e:
                print(
                    f"[deps][WARN] Failed to install {pkg_name} via pip: {e}",
                    flush=True,
                )

def _make_seq_data_cfg(seq_name: str, base_data_dir: Path, out_dir: Path) -> tuple[Path, str]:
    """
    Create a per-sequence MAC-VO data config based on MACVO_DATA_TEMPLATE.

    - seq_name: "test_0", "test_1", ...
    - base_data_dir: /workspace/ASP/data/data_2d_test_slam/test_X/<drive>/

    We will:
      - set cfg["name"] = f"kitti360_{clean_seq}" where clean_seq="test0", "test1", ...
      - set cfg["args"]["root"] = str(base_data_dir)

    Returns:
      (seq_cfg_path, dataset_name)
      where dataset_name is what MAC-VO uses in its Results folder:
        Results/MACVO-Fast@<dataset_name>/
    """
    assert MACVO_DATA_TEMPLATE.exists(), f"Template MAC-VO data config not found: {MACVO_DATA_TEMPLATE}"

    with MACVO_DATA_TEMPLATE.open("r") as f:
        cfg = yaml.safe_load(f)

    # "test_0" -> "test0", so name becomes "kitti360_test0"
    clean_seq = seq_name.replace("_", "")
    dataset_name = f"kitti360_{clean_seq}"

    cfg["name"] = dataset_name

    args = cfg.get("args", {})
    args["root"] = str(base_data_dir)
    cfg["args"] = args

    # Save per-sequence config in this sequence's output dir (nice and tidy)
    seq_cfg_path = out_dir / f"MAC_VO_{seq_name}.yaml"
    seq_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with seq_cfg_path.open("w") as f:
        yaml.safe_dump(cfg, f)

    log(f"[cfg] Wrote per-sequence MAC-VO cfg -> {seq_cfg_path} (name={dataset_name})")

    return seq_cfg_path, dataset_name


def _find_latest_macvo_result(exp_prefix: str) -> Path:
    """
    Find the latest MAC-VO experiment folder under Results/exp_prefix/*.
    exp_prefix example: "MACVO-Fast@kitti360_test0"
    """
    base = MACVO_ROOT / "Results" / exp_prefix
    assert base.exists(), f"MAC-VO Results base not found: {base}"

    candidates = [p for p in base.iterdir() if p.is_dir()]
    assert candidates, f"No runs found under {base}"

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def _poses_npy_to_tum(poses_npy: Path, timestamps: List[float]) -> List[str]:
    """
    Convert MAC-VO poses.npy into TUM txt lines using provided timestamps.

    Supported formats:
      - (N, 4, 4): homogeneous matrices T_w_c
      - (N, 7): [tx, ty, tz, qx, qy, qz, qw]
      - (N, 8): [id_or_time, tx, ty, tz, qx, qy, qz, qw]

    TUM format line:
      t tx ty tz qx qy qz qw
    """
    poses = np.load(poses_npy)

    tum_rows: List[str] = []

    # Case 1: full 4x4 matrices
    if poses.ndim == 3 and poses.shape[1:] == (4, 4):
        n = poses.shape[0]
        assert n <= len(timestamps), (
            f"More poses ({n}) than timestamps ({len(timestamps)}); "
            "you may need to slice or adjust."
        )

        for i in range(n):
            T = poses[i]
            t_xyz, q_xyzw = tum_from_T(T)
            ts = float(timestamps[i])
            tum_rows.append(
                f"{ts:.9f} "
                f"{t_xyz[0]:.6f} {t_xyz[1]:.6f} {t_xyz[2]:.6f} "
                f"{q_xyzw[0]:.6f} {q_xyzw[1]:.6f} {q_xyzw[2]:.6f} {q_xyzw[3]:.6f}"
            )
        return tum_rows

    # Case 2: (N,7) pose vectors -> [tx, ty, tz, qx, qy, qz, qw]
    if poses.ndim == 2 and poses.shape[1] == 7:
        n = poses.shape[0]
        assert n <= len(timestamps), (
            f"More poses ({n}) than timestamps ({len(timestamps)}); "
            "you may need to slice or adjust."
        )

        for i in range(n):
            tx, ty, tz, qx, qy, qz, qw = poses[i]
            ts = float(timestamps[i])
            tum_rows.append(
                f"{ts:.9f} "
                f"{float(tx):.6f} {float(ty):.6f} {float(tz):.6f} "
                f"{float(qx):.6f} {float(qy):.6f} {float(qz):.6f} {float(qw):.6f}"
            )
        return tum_rows

    # Case 3: (N,8) pose vectors -> [id_or_time, tx, ty, tz, qx, qy, qz, qw]
    if poses.ndim == 2 and poses.shape[1] == 8:
        n = poses.shape[0]
        assert n <= len(timestamps), (
            f"More poses ({n}) than timestamps ({len(timestamps)}); "
            "you may need to slice or adjust."
        )

        for i in range(n):
            vec = poses[i]
            # vec[0] = frame index or time in ms -> we ignore it
            tx, ty, tz = vec[1:4]
            qx, qy, qz, qw = vec[4:8]
            ts = float(timestamps[i])
            tum_rows.append(
                f"{ts:.9f} "
                f"{float(tx):.6f} {float(ty):.6f} {float(tz):.6f} "
                f"{float(qx):.6f} {float(qy):.6f} {float(qz):.6f} {float(qw):.6f}"
            )
        return tum_rows

    # Otherwise: unknown format
    raise ValueError(
        f"Unexpected poses.npy shape {poses.shape}, expected (N,4,4), (N,7) or (N,8)"
    )


def _write_ply_from_tensor_map(tensor_map_path: Path, out_ply: Path) -> Optional[int]:
    """
    Try to convert MAC-VO tensor_map.npz into a simple XYZ PLY.

    Heuristic:
      - load npz
      - find first 2D array with shape (N, >=3)
      - use its first 3 columns as XYZ

    Returns number of points written, or None if failed.
    """
    if not tensor_map_path.exists():
        return None

    data = np.load(tensor_map_path)
    pts = None

    for key in data.files:
        arr = data[key]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 3:
            pts = arr[:, :3]
            break

    if pts is None:
        log(f"[WARN] No suitable (N,>=3) array found in {tensor_map_path}; cannot write PLY.")
        return None

    n = pts.shape[0]
    log(f"[ok] Found {n} map points in {tensor_map_path}, writing PLY -> {out_ply}")

    with out_ply.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = pts[i]
            f.write(f"{float(x)} {float(y)} {float(z)}\n")

    return n


def _monitor_system_loop(
    stop_event: threading.Event,
    samples: List[SystemSample],
    interval_sec: float = 0.5,
) -> None:
    """
    Background system monitor: samples CPU / RAM / GPU while MAC-VO is running.
    """
    while not stop_event.is_set():
        t = time.time()
        append_system_sample(samples, samples, t)
        time.sleep(interval_sec)


# ============================================================
# PHASE 1: Run MAC-VO + save TUM + (optional) map
# ============================================================

def run_slam_stereo(seq_name: str) -> RunInfo:
    """
    Run MAC-VO on KITTI-360 test sequence `seq_name` using a *per-sequence*
    MAC_VO config and save:

      - POSES_<drive>.txt (TUM format)
      - MAP_<drive>.ply  (converted from tensor_map.npz if possible)

    under ASP/output/MAC_VO/<seq_name>/.
    """
    assert seq_name in TEST_DRIVES, f"Unknown seq_name: {seq_name}"
    drive_name = TEST_DRIVES[seq_name]

    # KITTI-360 data layout (reused from ORB runner)
    base_data_dir = (DATA_2D_TEST / seq_name / drive_name).resolve()
    img_dir_left = base_data_dir / "image_00" / "data_rect"
    img_ts_path = base_data_dir / "image_00" / "timestamps.txt"

    # Output directory for this sequence
    out_dir = (OUT_MACVO_ROOT / seq_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build per-sequence MAC-VO data config
    seq_data_cfg_path, dataset_name = _make_seq_data_cfg(seq_name, base_data_dir, out_dir)
    exp_prefix = f"MACVO-Fast@{dataset_name}"

    # Pretty config logging (reusing your helper)
    log_phase1_config(
        algo_name="MAC-VO",
        mode="STEREO",
        seq_name=seq_name,
        drive_name=drive_name,
        dataroot=base_data_dir,
        saving_dir=out_dir,
        cfg_paths={
            "MACVO_ODOM": str(MACVO_ODOM_CFG),
            "MACVO_DATA": str(seq_data_cfg_path),
            "poses.txt": str(DATA_POSES / drive_name / "poses.txt"),
        },
        data_dirs={
            "image_00 data_rect": img_dir_left,
            "image_00 timestamps": img_ts_path,
        },
    )

    # Basic checks
    ensure_dirs_exist(
        {
            "image_00 data_rect": img_dir_left,
        }
    )
    assert MACVO_ROOT.exists(), f"MAC-VO root not found: {MACVO_ROOT}"
    assert MACVO_ODOM_CFG.exists(), f"Missing MAC-VO odometry config: {MACVO_ODOM_CFG}"
    assert seq_data_cfg_path.exists(), f"Missing per-sequence MAC-VO data config: {seq_data_cfg_path}"
    assert img_ts_path.exists(), f"Missing image timestamps: {img_ts_path}"

    # Count frames (left images)
    imgs_left = sorted(
        [*img_dir_left.glob("*.png"),
         *img_dir_left.glob("*.jpg"),
         *img_dir_left.glob("*.jpeg")]
    )
    assert imgs_left, f"No images found in {img_dir_left}"
    frames_total = len(imgs_left)
    log(f"[check] Found {frames_total} left image files for {seq_name}")

    # Load timestamps (absolute -> relative), same as ORB runner
    ts_abs: List[float] = []
    with img_ts_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ts_abs.append(parse_kitti360_timestamp(line))

    assert ts_abs, f"No timestamps in {img_ts_path}"
    if len(ts_abs) < frames_total:
        log(
            f"[WARN] Only {len(ts_abs)} timestamps for {frames_total} frames; "
            "truncating to min length."
        )
        frames_total = min(frames_total, len(ts_abs))
        imgs_left = imgs_left[:frames_total]

    t0_ts = ts_abs[0]
    ts_rel = [t - t0_ts for t in ts_abs[:frames_total]]

    # Run MAC-VO as a subprocess with the *per-sequence* data cfg
    cmd = [
        "python3",
        "MACVO.py",
        "--odom", str(MACVO_ODOM_CFG),
        "--data", str(seq_data_cfg_path),
    ]
    log(f"[sys] Running MAC-VO: {' '.join(cmd)} (cwd={MACVO_ROOT})")

    # Start system monitoring in parallel
    sys_samples: List[SystemSample] = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor_system_loop,
        args=(stop_event, sys_samples, 0.5),
        daemon=True,
    )

    t0_wall = time.time()
    monitor_thread.start()
    try:
        subprocess.run(cmd, cwd=str(MACVO_ROOT), check=True)
    finally:
        elapsed = time.time() - t0_wall
        stop_event.set()
        monitor_thread.join(timeout=5.0)

    avg_fps = frames_total / elapsed if elapsed > 0 else 0.0

    # Summarize system metrics from collected samples
    sys_summary: Dict[str, float] = summarize_system_metrics(sys_samples)
    log(f"[sys] System metrics summary: {sys_summary}")

    # Optionally: dump raw samples as a simple log file
    sys_log_path = out_dir / f"SYS_METRICS_{drive_name}.txt"
    try:
        with sys_log_path.open("w") as f:
            f.write("# t cpu_percent ram_percent gpu_util gpu_mem_used_mb gpu_mem_total_mb gpu_power_w\n")
            for s in sys_samples:
                f.write(
                    f"{s.t:.6f} {s.cpu_percent:.2f} {s.ram_percent:.2f} "
                    f"{(s.gpu_utilization_percent if s.gpu_utilization_percent is not None else -1):.2f} "
                    f"{(s.gpu_memory_used_mb if s.gpu_memory_used_mb is not None else -1):.2f} "
                    f"{(s.gpu_memory_total_mb if s.gpu_memory_total_mb is not None else -1):.2f} "
                    f"{(s.gpu_power_watts if s.gpu_power_watts is not None else -1):.2f}\n"
                )
        log(f"[sys] Wrote raw system metrics -> {sys_log_path}")
    except Exception as e:
        log(f"[sys][WARN] Failed to write system metrics log: {e}")

    # Find latest MAC-VO result and extract poses
    exp_dir = _find_latest_macvo_result(exp_prefix)
    log(f"[ok] Using MAC-VO result folder: {exp_dir}")

    poses_npy = exp_dir / "poses.npy"
    assert poses_npy.exists(), f"MAC-VO poses.npy not found at {poses_npy}"

    tum_rows = _poses_npy_to_tum(poses_npy, ts_rel)
    frames_traj = len(tum_rows)
    log(f"[ok] Converted {frames_traj} MAC-VO poses to TUM format")

    # Save TUM poses
    base = drive_name
    poses_path = out_dir / f"POSES_{base}.txt"
    with poses_path.open("w") as f:
        f.write("\n".join(tum_rows) + ("\n" if tum_rows else ""))
    log(f"[ok] estimated poses (TUM) -> {poses_path}")

    # Map: try to convert tensor_map.npz -> PLY
    num_map_points = None

    # If MAC-VO ever writes PLY directly, prefer that
    ply_candidates = sorted(exp_dir.glob("*.ply"))
    if ply_candidates:
        src_ply = ply_candidates[0]
        dst_ply = out_dir / f"MAP_{base}.ply"
        shutil.copy2(src_ply, dst_ply)
        log(f"[ok] copied MAC-VO map -> {dst_ply}")
    else:
        tensor_map_path = exp_dir / "tensor_map.npz"
        dst_ply = out_dir / f"MAP_{base}.ply"
        n_pts = _write_ply_from_tensor_map(tensor_map_path, dst_ply)
        if n_pts is not None:
            num_map_points = n_pts
        else:
            log("[WARN] No PLY map produced from tensor_map.npz; MAP_* will not be written.")

    # Summary (centralized) – now with sys_summary filled
    log_slam_summary(
        seq_name=seq_name,
        drive_name=drive_name,
        frames_total=frames_total,
        tum_rows_count=frames_traj,
        num_pose_changes=None,
        elapsed=elapsed,
        avg_fps=avg_fps,
        sys_summary=sys_summary,
        first_ok_frame=None,
        last_ok_frame=None,
        title="SUMMARY (MAC-VO STEREO, KITTI-360)",
    )

    return RunInfo(
        seq_name=seq_name,
        drive_name=drive_name,
        out_dir=out_dir,
        frames_dataset=frames_total,
        frames_traj=frames_traj,
        wall_time=elapsed,
        avg_fps=avg_fps,
        num_map_points=num_map_points,
        sys_summary=sys_summary,
    )


# ============================================================
# PHASE 2: ATE/RPE + dashboard (same style as ORB runner)
# ============================================================

def post_process_sequence(info: RunInfo) -> RunSummary:
    """
    Compute ATE/RPE vs KITTI-360 GT (poses.txt-based, image_00 subset) and write:
      - ATE_<drive>.txt + ATE_ERRORS_<drive>.txt
      - RPE_0p5s_* and RPE_2p0s_* (summary + detailed)
      - DASHBOARD_<drive>.txt
    """
    seq_name = info.seq_name
    drive_name = info.drive_name
    out_dir = info.out_dir

    log("\n-------------------------------------------------------")
    log(f"[PHASE2] Post-processing MAC-VO (poses.txt GT): {seq_name} / {drive_name}")
    log("-------------------------------------------------------")

    poses_path = out_dir / f"POSES_{drive_name}.txt"
    assert poses_path.exists(), f"Predicted TUM poses not found: {poses_path}"

    ate: Optional[ATEResult] = None
    rpe_short: Optional[RPEResult] = None
    rpe_long: Optional[RPEResult] = None

    try:
        gt_traj: Trajectory = load_kitti360_poses3x4_as_traj_for_2d(seq_name, drive_name)
        pred_traj: Trajectory = load_tum_trajectory(poses_path)

        ate = compute_ate(
            traj_ref=gt_traj,
            traj_query=pred_traj,
            max_dt=0.05,
            with_scale=True,
        )

        ate_path = out_dir / f"ATE_{drive_name}.txt"
        ate_err_path = out_dir / f"ATE_ERRORS_{drive_name}.txt"
        write_ate_report(ate_path, ate, extra_info=None)
        write_ate_detailed(ate_err_path, gt_traj, pred_traj, ate)
        log(f"[ATE] Wrote ATE report        -> {ate_path}")
        log(f"[ATE] Wrote detailed errors  -> {ate_err_path}")

        rpe_short = compute_rpe(
            traj_ref=gt_traj,
            traj_query=pred_traj,
            delta_t=0.5,
            max_dt=0.05,
        )
        rpe_long = compute_rpe(
            traj_ref=gt_traj,
            traj_query=pred_traj,
            delta_t=2.0,
            max_dt=0.05,
        )

        rpe_short_path = out_dir / f"RPE_0p5s_{drive_name}.txt"
        rpe_short_err_path = out_dir / f"RPE_0p5s_ERRORS_{drive_name}.txt"
        rpe_long_path = out_dir / f"RPE_2p0s_{drive_name}.txt"
        rpe_long_err_path = out_dir / f"RPE_2p0s_ERRORS_{drive_name}.txt"

        write_rpe_report(rpe_short_path, rpe_short)
        write_rpe_detailed(rpe_short_err_path, rpe_short)
        write_rpe_report(rpe_long_path, rpe_long)
        write_rpe_detailed(rpe_long_err_path, rpe_long)

        log(f"[RPE] Wrote short RPE  -> {rpe_short_path}, {rpe_short_err_path}")
        log(f"[RPE] Wrote long RPE   -> {rpe_long_path}, {rpe_long_err_path}")

    except Exception as e:
        log(f"[PHASE2][ERR] Failed to compute ATE/RPE for {seq_name}/{drive_name}: {e}")

    dashboard_path = out_dir / f"DASHBOARD_{drive_name}.txt"
    summary = RunSummary(
        seq_name=seq_name,
        drive_name=drive_name,
        frames_dataset=info.frames_dataset,
        frames_traj=info.frames_traj,
        wall_time=info.wall_time,
        avg_fps=info.avg_fps,
        num_map_points=info.num_map_points,
        ate=ate,
        rpe_short=rpe_short,
        rpe_long=rpe_long,
        sys_summary=info.sys_summary,
    )
    write_dashboard(dashboard_path, summary, stereo=True)

    return summary


# ============================================================
# Main
# ============================================================

def main() -> None:
    ensure_monitoring_deps()
    # "full" = ["test_0", "test_1", "test_2", "test_3"]
    # "light" = ["test_0"]
    SCENARIO = "light"

    assert SCENARIO in TEST_SCENARIOS, f"Unknown SCENARIO: {SCENARIO}"
    seq_list = TEST_SCENARIOS[SCENARIO]

    # Phase 1: run MAC-VO
    phase1_infos: List[RunInfo] = []
    for seq_name in seq_list:
        log(
            "\n=======================================================\n"
            f"[PHASE1] MAC-VO (STEREO) on {seq_name} / {TEST_DRIVES[seq_name]}\n"
            "======================================================="
        )
        info = run_slam_stereo(seq_name)
        phase1_infos.append(info)

    # Phase 2: ATE/RPE + dashboard
    all_summaries: List[RunSummary] = []
    for info in phase1_infos:
        summary = post_process_sequence(info)
        all_summaries.append(summary)

    log_overall_summary("MAC-VO", "STEREO-poses", all_summaries)


if __name__ == "__main__":
    main()
