#!/usr/bin/env python3
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import orbslam3 as slam

from paths import (
    DATA_2D_TEST,
    DATA_POSES,
    VOCAB,
    CFG_MONO,
    OUT_ORB3_MONO,
    TEST_DRIVES,
    TEST_SCENARIOS,
)

# File/print helpers from file_creator
from file_creator import (
    log,
    log_phase1_config,
    log_overall_summary,
    log_slam_summary,
    get_ply_vertex_count,
)

# Remaining helpers from utils
from utils import (
    tum_from_T,
    parse_kitti360_timestamp,
    rot_to_quat,
    ensure_dirs_exist,
    load_image_gray,
    load_kitti360_cam0_to_world_as_traj,
)
from runners.common import run_slam_loop, save_map_points_ply

from metrics import (
    SystemSample,
    ATEResult,
    RPEResult,
    Trajectory,
    append_system_sample,
    summarize_system_metrics,
    load_tum_trajectory,
    compute_ate,
    compute_rpe,
)
from file_creator import (
    write_system_metrics_txt,
    write_ate_report,
    write_ate_detailed,
    write_rpe_report,
    write_rpe_detailed,
    write_dashboard,
)


# ============================================================
# Dataclasses for Phase 1 / Phase 2
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
# PHASE 1: Run ORB-SLAM3 mono + system monitoring (2D_TEST)
# ============================================================

def run_slam_mono(seq_name: str) -> RunInfo:
    """
    Run ORB-SLAM3 MONO on KITTI-360 test sequence `seq_name`
    using DATA_2D_TEST images, and record:
      - TUM trajectory (st==OK)
      - MACHINE_USAGE_<drive>.txt
      - MAP_<drive>.ply (if supported by this ORB-SLAM3 build)
    """
    assert seq_name in TEST_DRIVES, f"Unknown seq_name: {seq_name}"
    drive_name = TEST_DRIVES[seq_name]

    base_data_dir = (DATA_2D_TEST / seq_name / drive_name).resolve()
    img_dir = base_data_dir / "image_00" / "data_rect"
    img_ts_path = base_data_dir / "image_00" / "timestamps.txt"

    out_dir = (OUT_ORB3_MONO / seq_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pretty config logging (mirror stereo format, but mono-specific paths)
    log_phase1_config(
        algo_name="ORB-SLAM3",
        mode="MONO",
        seq_name=seq_name,
        drive_name=drive_name,
        dataroot=base_data_dir,
        saving_dir=out_dir,
        cfg_paths={
            "VOCAB": str(VOCAB),
            "CFG_MONO": str(CFG_MONO),
            "cam0_to_world": str(DATA_POSES / drive_name / "cam0_to_world.txt"),
        },
        data_dirs={
            "image_00 data_rect": img_dir,
            "image_00 timestamps": img_ts_path,
        },
    )

    # Checks (same as before)
    ensure_dirs_exist({"image_00 data_rect": img_dir})
    assert VOCAB.exists(), f"Missing vocabulary: {VOCAB}"
    assert CFG_MONO.exists(), f"Missing mono config: {CFG_MONO}"
    assert img_ts_path.exists(), f"Missing image timestamps: {img_ts_path}"

    imgs = sorted([*img_dir.glob("*.png"), *img_dir.glob("*.jpg"), *img_dir.glob("*.jpeg")])
    assert imgs, f"No images found in {img_dir}"
    frames_total = len(imgs)
    log(f"[check] Found {frames_total} image files")

    # Load timestamps (absolute -> relative)
    ts_abs: List[float] = []
    with img_ts_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ts_abs.append(parse_kitti360_timestamp(line))

    assert ts_abs, f"No timestamps in {img_ts_path}"
    if len(ts_abs) < frames_total:
        log(f"[WARN] Only {len(ts_abs)} timestamps for {frames_total} frames; truncating to min length.")
        frames_total = min(frames_total, len(ts_abs))
        imgs = imgs[:frames_total]

    t0_ts = ts_abs[0]
    ts_rel = [t - t0_ts for t in ts_abs]

    # YAML vs image size (same check as stereo)
    fs = cv2.FileStorage(str(CFG_MONO), cv2.FILE_STORAGE_READ)
    assert fs.isOpened(), f"OpenCV failed to open {CFG_MONO}"
    cam_w = int(fs.getNode("Camera.width").real())
    cam_h = int(fs.getNode("Camera.height").real())
    fs.release()

    im0 = load_image_gray(imgs[0])
    h0, w0 = im0.shape[:2]
    log(f"[check] YAML size   : {cam_w}x{cam_h}")
    log(f"[check] First frame : {w0}x{h0}")
    if cam_w != w0 or cam_h != h0:
        log("[WARN] Image size does NOT match YAML, ORB-SLAM3 may behave badly")

    # ORB-SLAM3 system (mono)
    log("[sys] Creating ORB-SLAM3 system (MONO)...")
    slam_sys = slam.system(str(VOCAB), str(CFG_MONO), slam.Sensor.MONOCULAR)

    if hasattr(slam_sys, "set_use_viewer"):
        slam_sys.set_use_viewer(False)

    log("[sys] Initializing ORB-SLAM3...")
    slam_sys.initialize()

    have_state = hasattr(slam_sys, "get_tracking_state")
    have_pose = hasattr(slam_sys, "get_current_pose")
    can_save_ply = hasattr(slam_sys, "save_map_points_ply")
    log(f"[cap] tracking_state={have_state} pose={have_pose} ply_saver={can_save_ply}")

    # Main loop using common runner (build frames_info as in stereo but with None for right)
    t0_wall = time.time()
    frames_info = [(p, None, ts_rel[i]) for i, p in enumerate(imgs[:frames_total])]
    tum_rows, system_samples, num_pose_changes, first_ok_frame, last_ok_frame = run_slam_loop(
        slam_sys=slam_sys,
        frames_info=frames_info,
        cam_w=cam_w,
        cam_h=cam_h,
    )

    # Map (PLY) + count points (optional)
    num_map_points = None
    base = drive_name
    if can_save_ply:
        map_path = out_dir / f"MAP_{base}.ply"
        num_map_points = save_map_points_ply(slam_sys, map_path)
    else:
        log("[WARN] save_map_points_ply() not exposed in this build; no PLY map saved.")

    # Shutdown
    log("[sys] Shutting down ORB-SLAM3...")
    slam_sys.shutdown()

    elapsed = time.time() - t0_wall
    avg_fps = frames_total / elapsed if elapsed > 0 else 0.0

    # Poses
    poses_path = out_dir / f"POSES_{base}.txt"
    with poses_path.open("w") as f:
        f.write("\n".join(tum_rows) + ("\n" if tum_rows else ""))
    log(f"[ok] estimated poses (st==OK, TUM) -> {poses_path}")

    # System usage (write metrics file + summarize)
    machine_path = out_dir / f"MACHINE_USAGE_{base}.txt"
    write_system_metrics_txt(machine_path, system_samples)
    sys_summary = summarize_system_metrics(system_samples)

    # Summary (centralized)
    log_slam_summary(
        seq_name=seq_name,
        drive_name=drive_name,
        frames_total=frames_total,
        tum_rows_count=len(tum_rows),
        num_pose_changes=num_pose_changes,
        elapsed=elapsed,
        avg_fps=avg_fps,
        sys_summary=sys_summary,
        first_ok_frame=first_ok_frame,
        last_ok_frame=last_ok_frame,
        title="SUMMARY (MONO SLAM ONLY)",
    )

    frames_traj = len(tum_rows)
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
# PHASE 2: ATE/RPE + dashboard
# ============================================================

def post_process_sequence(info: RunInfo) -> RunSummary:
    seq_name = info.seq_name
    drive_name = info.drive_name
    out_dir = info.out_dir

    log("\n-------------------------------------------------------")
    log(f"[PHASE2] Post-processing MONO ORB-SLAM3: {seq_name} / {drive_name}")
    log("-------------------------------------------------------")

    poses_path = out_dir / f"POSES_{drive_name}.txt"
    assert poses_path.exists(), f"Predicted TUM poses not found: {poses_path}"

    ate: Optional[ATEResult] = None
    rpe_short: Optional[RPEResult] = None
    rpe_long: Optional[RPEResult] = None

    try:
        gt_traj: Trajectory = load_kitti360_cam0_to_world_as_traj(DATA_POSES / drive_name / "cam0_to_world.txt")  # uses utils helper
        pred_traj: Trajectory = load_tum_trajectory(poses_path)

        ate = compute_ate(traj_ref=gt_traj, traj_query=pred_traj, max_dt=0.05, with_scale=True)

        ate_path = out_dir / f"ATE_{drive_name}.txt"
        ate_err_path = out_dir / f"ATE_ERRORS_{drive_name}.txt"
        write_ate_report(ate_path, ate, extra_info=None)
        write_ate_detailed(ate_err_path, gt_traj, pred_traj, ate)
        log(f"[ATE] Wrote ATE report        -> {ate_path}")
        log(f"[ATE] Wrote detailed errors  -> {ate_err_path}")

        rpe_short = compute_rpe(traj_ref=gt_traj, traj_query=pred_traj, delta_t=0.5, max_dt=0.05)
        rpe_long = compute_rpe(traj_ref=gt_traj, traj_query=pred_traj, delta_t=2.0, max_dt=0.05)

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

    # Dashboard
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
    write_dashboard(dashboard_path, summary, stereo=False)

    return summary


# ============================================================
# Main
# ============================================================

def main() -> None:
    SCENARIO = "full"  # "light" or "full" 
    assert SCENARIO in TEST_SCENARIOS, f"Unknown SCENARIO: {SCENARIO}"
    seq_list = TEST_SCENARIOS[SCENARIO]

    phase1_infos: List[RunInfo] = []
    for seq_name in seq_list:
        info = run_slam_mono(seq_name)
        phase1_infos.append(info)

    all_summaries: List[RunSummary] = []
    for info in phase1_infos:
        summary = post_process_sequence(info)
        all_summaries.append(summary)

    log_overall_summary("ORB-SLAM3", "MONO", all_summaries)


if __name__ == "__main__":
    main()
