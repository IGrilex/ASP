#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import time
import signal
import shutil
from subprocess import TimeoutExpired
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from paths import (
    DATA_AIRSLAM_ROOT,
    OUT_AIRSLAM_ROOT,
    AIRSLAM_CONFIG_PATH,
    AIRSLAM_CAMERA_CONFIG_PATH,
    TEST_DRIVES,
    TEST_SCENARIOS,
)

# Logging + file writing helpers (centralized)
from file_creator import (
    log,
    log_phase1_config,
    ensure_dirs_exist,
    log_overall_summary,
    write_system_metrics_txt,
    write_ate_report,
    write_ate_detailed,
    write_rpe_report,
    write_rpe_detailed,
    write_dashboard,
    load_ply_xyz,  # re-exported from utils
)

from metrics import (
    append_system_sample,
    summarize_system_metrics,
    SystemSample,
    ATEResult,
    RPEResult,
    load_tum_trajectory,
    compute_ate,
    compute_rpe,
)

# General dataset helpers from utils (no more count_dataset_frames)
from utils import (
    build_frameidx_to_timestamp_for_seq,
    load_kitti360_poses3x4_as_traj_for_seq,
)

# ============================================================
# Data classes for run / summary
# ============================================================

@dataclass
class RunInfo:
    seq_name: str
    drive_name: str
    seq_out_dir: Path
    wall_time: float
    sys_summary: Dict[str, float]


@dataclass
class RunSummary:
    seq_name: str
    drive_name: str
    frames_dataset: int   # number of cam0 images
    frames_traj: int      # number of poses in trajectory
    wall_time: float
    avg_fps: float
    num_map_points: Optional[int]
    ate: Optional[ATEResult]
    rpe_short: Optional[RPEResult]  # e.g., delta_t = 0.5s
    rpe_long: Optional[RPEResult]   # e.g., delta_t = 2.0s
    sys_summary: Dict[str, float]


# ============================================================
# AirSLAM output helpers (traj + map)
# ============================================================

def find_airslam_traj_file(seq_out_dir: Path) -> Optional[Path]:
    """Return AirSLAM trajectory file if present."""
    candidates = [
        seq_out_dir / "trajectory_v0.txt",
        seq_out_dir / "trajectory.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def find_airslam_map_file(seq_out_dir: Path) -> Optional[Path]:
    """Return AirSLAM map binary file if present."""
    candidates = [
        seq_out_dir / "AirSLAM_mapv0.bin",
        seq_out_dir / "map.bin",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def parse_airslam_trajectory_to_tum(
    seq_name: str,
    drive_name: str,
    airslam_traj_path: Path,
    tum_out_path: Path,
) -> int:
    """
    Convert AirSLAM trajectory format to TUM format using frame-index-based timestamps.

    AirSLAM line format:
        frame_idx  x  y  z  qx  qy  qz  qw
    """
    if not airslam_traj_path.exists():
        raise FileNotFoundError(f"AirSLAM trajectory not found: {airslam_traj_path}")

    frameidx_to_ts = build_frameidx_to_timestamp_for_seq(seq_name)

    lines_parsed: List[tuple[int, List[float]]] = []
    with airslam_traj_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue

            try:
                frame_idx = int(round(float(parts[0])))
            except ValueError:
                log(f"[WARN] Cannot parse frame_idx from line in {airslam_traj_path.name}: {line}")
                continue

            if frame_idx not in frameidx_to_ts:
                log(f"[WARN] No timestamp for frame_idx={frame_idx} in {seq_name}, skipping.")
                continue

            xyzqw = list(map(float, parts[1:8]))  # x y z qx qy qz qw
            lines_parsed.append((frame_idx, xyzqw))

    if not lines_parsed:
        raise RuntimeError(
            f"No valid lines in {airslam_traj_path} after mapping frame_idx->timestamp"
        )

    # Sort by frame index
    lines_parsed.sort(key=lambda x: x[0])

    # Absolute timestamps for those frames
    ts_abs_list: List[float] = [frameidx_to_ts[fi] for fi, _ in lines_parsed]
    t0 = ts_abs_list[0]

    tum_out_path.parent.mkdir(parents=True, exist_ok=True)
    with tum_out_path.open("w") as f:
        for (_, xyzqw), ts_abs in zip(lines_parsed, ts_abs_list):
            t_rel = ts_abs - t0
            x, y, z, qx, qy, qz, qw = xyzqw
            f.write(
                f"{t_rel:.9f} "
                f"{x:.6f} {y:.6f} {z:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )

    log(
        f"[POSES] {seq_name}: {airslam_traj_path.name} -> {tum_out_path.name}, "
        f"poses={len(lines_parsed)} (frame_idx-based timing)"
    )

    return len(lines_parsed)


def handle_map_files(
    seq_out_dir: Path,
    drive_name: str,
) -> Optional[int]:
    """
    Handle AirSLAM map outputs:
      - copy AirSLAM_mapv0.bin/map.bin -> MAP_<drive>.bin
      - optionally count points in MAP_<drive>.ply if it exists
    """
    map_src = find_airslam_map_file(seq_out_dir)
    map_bin_named = seq_out_dir / f"MAP_{drive_name}.bin"
    map_ply_path = seq_out_dir / f"MAP_{drive_name}.ply"

    if map_src is not None:
        try:
            shutil.copy(map_src, map_bin_named)
            log(f"[MAP] Copied {map_src.name} -> {map_bin_named.name}")
        except Exception as e:
            log(f"[WARN] Failed to copy {map_src} -> {map_bin_named}: {e}")
    else:
        log(f"[MAP] No AirSLAM map binary found in {seq_out_dir}")

    if map_ply_path.exists():
        try:
            pts = load_ply_xyz(map_ply_path)
            num_points = int(pts.shape[0])
            log(f"[MAP] Found PLY map with {num_points} points: {map_ply_path}")
            return num_points
        except Exception as e:
            log(f"[WARN] Failed to load PLY map {map_ply_path}: {e}")
            return None
    else:
        log(
            f"[MAP] No PLY map found (MAP_{drive_name}.ply). "
            f"Map is currently only in binary form."
        )
        return None


def compute_and_save_ate_for_seq(
    seq_name: str,
    drive_name: str,
    tum_pred_path: Path,
    out_dir: Path,
) -> Optional[ATEResult]:
    """
    Compute ATE vs KITTI-360 GT for this sequence and write reports:
      - ATE_<drive>.txt (summary)
      - ATE_ERRORS_<drive>.txt (per-frame errors)
    """
    try:
        gt_traj = load_kitti360_poses3x4_as_traj_for_seq(seq_name, drive_name)
        pred_traj = load_tum_trajectory(tum_pred_path)
        ate = compute_ate(
            traj_ref=gt_traj,
            traj_query=pred_traj,
            max_dt=0.05,
            with_scale=True,  # stereo: expect scale ~1, but allow Sim(3)
        )
    except Exception as e:
        log(f"[ATE] Failed to compute ATE for {seq_name} / {drive_name}: {e}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    ate_path = out_dir / f"ATE_{drive_name}.txt"
    ate_detailed_path = out_dir / f"ATE_ERRORS_{drive_name}.txt"

    write_ate_report(ate_path, ate, extra_info=None)
    write_ate_detailed(ate_detailed_path, gt_traj, pred_traj, ate)

    log(f"[ATE] Wrote ATE report        -> {ate_path}")
    log(f"[ATE] Wrote detailed errors  -> {ate_detailed_path}")
    return ate


# ============================================================
# PHASE 1: run AirSLAM + monitoring
# ============================================================

def run_airslam_sequence_monitor_only(seq_name: str) -> RunInfo:
    """
    Run AirSLAM stereo on a single KITTI-360 test sequence and
    record CPU/RAM/GPU metrics + wall time.
    """
    assert seq_name in TEST_DRIVES, f"Unknown seq_name: {seq_name}"
    drive_name = TEST_DRIVES[seq_name]

    dataroot = DATA_AIRSLAM_ROOT / seq_name
    # Save outputs directly under the AirSLAM output root (no "stereo" subdir)
    seq_out_dir = OUT_AIRSLAM_ROOT / seq_name
    seq_out_dir.mkdir(parents=True, exist_ok=True)

    machine_usage_path = seq_out_dir / f"MACHINE_USAGE_{drive_name}.txt"

    cam0_dir = dataroot / "cam0" / "data"
    cam1_dir = dataroot / "cam1" / "data"

    # Pretty config logging (reusable for other algos)
    log_phase1_config(
        algo_name="AirSLAM",
        mode="STEREO",
        seq_name=seq_name,
        drive_name=drive_name,
        dataroot=dataroot,
        saving_dir=seq_out_dir,
        cfg_paths={
            "CONFIG_PATH": str(AIRSLAM_CONFIG_PATH),
            "CAMERA_CONFIG_PATH": str(AIRSLAM_CAMERA_CONFIG_PATH),
        },
        data_dirs={
            "cam0 dir": cam0_dir,
            "cam1 dir": cam1_dir,
        },
    )

    # Generic directory checks
    ensure_dirs_exist(
        {
            "dataroot": dataroot,
            "cam0 data dir": cam0_dir,
            "cam1 data dir": cam1_dir,
        }
    )

    n_cam0 = len(sorted(cam0_dir.glob("*.png")))
    n_cam1 = len(sorted(cam1_dir.glob("*.png")))
    log(f"[PHASE1] cam0 frames: {n_cam0}, cam1 frames: {n_cam1}")

    cmd = [
        "roslaunch", "air_slam", "vo_euroc.launch",
        f"config_path:={AIRSLAM_CONFIG_PATH}",
        f"camera_config_path:={AIRSLAM_CAMERA_CONFIG_PATH}",
        f"dataroot:={str(dataroot)}",
        f"saving_dir:={str(seq_out_dir)}",
        "visualization:=false",
    ]

    log("[PHASE1] Launching AirSLAM (stereo):")
    log("          " + " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    samples: List[SystemSample] = []
    t_start = time.time()
    saw_map_done = False

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")  # echo AirSLAM logs

            t = time.time()
            append_system_sample(samples, samples, t)

            if "Map saveing done" in line:
                log("[PHASE1] Detected 'Map saveing done'. Stopping roslaunch...")
                saw_map_done = True
                break

        if saw_map_done:
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass

        try:
            proc.wait(timeout=20.0)
        except TimeoutExpired:
            log("[PHASE1] roslaunch did not exit in time, killing...")
            proc.kill()

    except KeyboardInterrupt:
        log("[PHASE1] KeyboardInterrupt: sending SIGINT to roslaunch...")
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass
        proc.wait()

    t_end = time.time()
    wall_time = t_end - t_start

    log(f"[PHASE1] Wall time for {seq_name} / {drive_name}: {wall_time:.3f}s")

    log(f"[PHASE1] Writing system metrics to {machine_usage_path}")
    write_system_metrics_txt(machine_usage_path, samples)
    sys_summary = summarize_system_metrics(samples)

    return RunInfo(
        seq_name=seq_name,
        drive_name=drive_name,
        seq_out_dir=seq_out_dir,
        wall_time=wall_time,
        sys_summary=sys_summary,
    )


# ============================================================
# PHASE 2: post-process sequences (traj + metrics + dashboard)
# ============================================================

def post_process_sequence(info: RunInfo) -> RunSummary:
    """
    Post-process one sequence:
      - count dataset frames (cam0 subset)
      - convert AirSLAM traj -> POSES_<drive>.txt (TUM with correct timestamps)
      - compute ATE_<drive>.txt (+ detailed)
      - compute short/long RPE (+ detailed)
      - handle MAP_*.bin & optional MAP_*.ply
      - write DASHBOARD_<drive>.txt
    """
    seq_name = info.seq_name
    drive_name = info.drive_name
    seq_out_dir = info.seq_out_dir
    wall_time = info.wall_time

    log("\n-------------------------------------------------------")
    log(f"[PHASE2] Post-processing {seq_name} / {drive_name}")
    log("-------------------------------------------------------")

    # 1) dataset frames (cam0) – similar idea to ORB-SLAM3 using the subset used for GT
    frameidx_to_ts = build_frameidx_to_timestamp_for_seq(seq_name)
    frames_dataset = len(frameidx_to_ts)

    # 2) trajectory -> POSES_<drive>.txt (TUM)
    airslam_traj_path = find_airslam_traj_file(seq_out_dir)
    poses_path = seq_out_dir / f"POSES_{drive_name}.txt"

    frames_traj = 0
    if airslam_traj_path is not None:
        frames_traj = parse_airslam_trajectory_to_tum(
            seq_name=seq_name,
            drive_name=drive_name,
            airslam_traj_path=airslam_traj_path,
            tum_out_path=poses_path,
        )
    else:
        log(f"[PHASE2][POSES] No AirSLAM trajectory file found in {seq_out_dir}")
        frames_traj = 0

    avg_fps = (frames_dataset / wall_time) if wall_time > 0 and frames_dataset > 0 else 0.0

    # 3) Map files
    num_map_points = handle_map_files(seq_out_dir, drive_name)

    # 4) ATE / RPE vs GT
    ate: Optional[ATEResult] = None
    rpe_short: Optional[RPEResult] = None
    rpe_long: Optional[RPEResult] = None

    if frames_traj > 0:
        # ATE
        ate = compute_and_save_ate_for_seq(
            seq_name=seq_name,
            drive_name=drive_name,
            tum_pred_path=poses_path,
            out_dir=seq_out_dir,
        )

        # RPE (short + long)
        try:
            gt_traj = load_kitti360_poses3x4_as_traj_for_seq(seq_name, drive_name)
            pred_traj = load_tum_trajectory(poses_path)

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

            rpe_short_path = seq_out_dir / f"RPE_0p5s_{drive_name}.txt"
            rpe_short_err_path = seq_out_dir / f"RPE_0p5s_ERRORS_{drive_name}.txt"
            rpe_long_path = seq_out_dir / f"RPE_2p0s_{drive_name}.txt"
            rpe_long_err_path = seq_out_dir / f"RPE_2p0s_ERRORS_{drive_name}.txt"

            write_rpe_report(rpe_short_path, rpe_short)
            write_rpe_detailed(rpe_short_err_path, rpe_short)
            write_rpe_report(rpe_long_path, rpe_long)
            write_rpe_detailed(rpe_long_err_path, rpe_long)

            log(f"[RPE] Wrote short RPE  -> {rpe_short_path}, {rpe_short_err_path}")
            log(f"[RPE] Wrote long RPE   -> {rpe_long_path}, {rpe_long_err_path}")

        except Exception as e:
            log(f"[RPE] Failed to compute RPE for {seq_name} / {drive_name}: {e}")
    else:
        log("[PHASE2][ATE/RPE] Skipping ATE/RPE: no trajectory poses.")

    # 5) Dashboard
    dashboard_path = seq_out_dir / f"DASHBOARD_{drive_name}.txt"
    summary = RunSummary(
        seq_name=seq_name,
        drive_name=drive_name,
        frames_dataset=frames_dataset,
        frames_traj=frames_traj,
        wall_time=wall_time,
        avg_fps=avg_fps,
        num_map_points=num_map_points,
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
    # Choose which subset to run (use TEST_SCENARIOS from paths.py)
    SCENARIO = "full"  # or "light"

    assert SCENARIO in TEST_SCENARIOS, f"Unknown SCENARIO: {SCENARIO}"
    seq_list = TEST_SCENARIOS[SCENARIO]

    # PHASE 1: run AirSLAM + monitoring
    phase1_infos: List[RunInfo] = []
    for seq_name in seq_list:
        info = run_airslam_sequence_monitor_only(seq_name)
        phase1_infos.append(info)

    # PHASE 2: post-process
    all_summaries: List[RunSummary] = []
    for info in phase1_infos:
        summary = post_process_sequence(info)
        all_summaries.append(summary)

    # Generic reusable overall summary for any algorithm
    log_overall_summary("AirSLAM", "STEREO", all_summaries)


if __name__ == "__main__":
    main()
