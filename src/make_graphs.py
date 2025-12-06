#!/usr/bin/env python3
"""
make_graphs.py

Generate GT vs predicted trajectory plots (top-down X/Y) for each algorithm
and test sequence. Saves PNGs under output/graphs/<algorithm>/<seq>/.

NEW:
  - Summary bar charts per sequence using DASHBOARD_<drive>.txt:
      - wall_time_sec (inference time)
      - avg_fps
      - ATE rmse
      - RPE 0.5s translational rmse
      - RPE 2.0s translational rmse
      - CPU/GPU/RAM average usage
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

# project helpers
from paths import (
    OUTPUT_ROOT,
    OUT_ORB3_MONO,
    OUT_ORB3_STEREO,
    OUT_MAST3R,
    DATA_POSES,
    DATA_2D_TEST,
    TEST_DRIVES,
)
from file_creator import log
from utils import (
    rot_to_quat,
    parse_kitti360_timestamp,
)

import metrics  # provides Trajectory, load_tum_trajectory, associate_by_time, align_xyz_umeyama


# Map algorithm labels -> candidate output root on disk
# (was a static dict; now we keep known mappings and also auto-discover any subfolders
#  under OUTPUT_ROOT so new outputs like "MAC_VO" are picked up automatically)
KNOWN_ALG_DIRS = {
    "ORB3_mono": OUT_ORB3_MONO,
    "ORB3_stereo": OUT_ORB3_STEREO,
    "MAST3R": OUT_MAST3R,
    "AirSLAM": OUTPUT_ROOT / "AirSLAM",
}


def build_alg_dirs() -> Dict[str, Path]:
    """
    Build ALG_DIRS by merging KNOWN_ALG_DIRS with any directories found directly
    under OUTPUT_ROOT. Known mappings take precedence, but discovered folders
    (e.g. OUTPUT_ROOT / 'MAC_VO') are added automatically.
    """
    algs: Dict[str, Path] = {}

    # Add known mappings first (keep them even if they don't exist; other code checks)
    for name, path in KNOWN_ALG_DIRS.items():
        algs[name] = path

    # Discover additional algorithm folders under OUTPUT_ROOT and add them
    if OUTPUT_ROOT.exists() and OUTPUT_ROOT.is_dir():
        for p in sorted(OUTPUT_ROOT.iterdir()):
            if not p.is_dir():
                continue
            name = p.name
            # If this name is already in known mappings, prefer the known mapping but log the discovery
            if name in algs:
                if algs[name] != p:
                    log(f"[DISCOVER] Found additional output dir for {name}: {p} (configured path={algs[name]})")
                continue
            algs[name] = p
            log(f"[DISCOVER] Auto-registered algorithm '{name}' -> {p}")
    else:
        log(f"[DISCOVER] OUTPUT_ROOT does not exist or is not a directory: {OUTPUT_ROOT}")

    return algs


# Final ALG_DIRS used throughout the script
ALG_DIRS = build_alg_dirs()


# ============================================================
# Dashboard parsing (for summary graphs)
# ============================================================

@dataclass
class DashboardStats:
    algo_name: str
    seq_name: str
    drive_name: str
    wall_time_sec: Optional[float] = None
    avg_fps: Optional[float] = None
    ate_rmse: Optional[float] = None
    rpe_0p5_trans_rmse: Optional[float] = None
    rpe_2p0_trans_rmse: Optional[float] = None
    cpu_avg: Optional[float] = None
    gpu_avg: Optional[float] = None
    ram_avg: Optional[float] = None


def parse_dashboard(algo_name: str, seq_name: str, drive_name: str, path: Path) -> Optional[DashboardStats]:
    """
    Parse a DASHBOARD_<drive>.txt in the expected format.
    We only extract what we need for plotting: wall_time, avg_fps, ATE rmse,
    RPE 0.5s/2.0s translational rmse, CPU/GPU/RAM averages.
    """
    if not path.exists():
        log(f"[DASH] Missing dashboard for {algo_name}/{seq_name}: {path}")
        return None

    stats = DashboardStats(algo_name=algo_name, seq_name=seq_name, drive_name=drive_name)

    mode = None
    try:
        with path.open("r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("==="):
                    continue

                # Global summary fields
                if s.startswith("wall_time_sec"):
                    stats.wall_time_sec = float(s.split(":")[1])
                    continue
                if s.startswith("avg_fps"):
                    stats.avg_fps = float(s.split(":")[1])
                    continue

                # Section switches
                if s.startswith("ATE (meters)"):
                    mode = "ATE"
                    continue
                if s.startswith("RPE (delta_t=0.50s) translational"):
                    mode = "RPE_0p5_trans"
                    continue
                if s.startswith("RPE (delta_t=2.00s) translational"):
                    mode = "RPE_2p0_trans"
                    continue
                if s.startswith("System usage (percent)"):
                    mode = "SYS"
                    continue

                # Values inside sections
                if mode == "ATE" and s.startswith("rmse"):
                    stats.ate_rmse = float(s.split(":")[1])
                    continue
                if mode == "RPE_0p5_trans" and s.startswith("rmse"):
                    stats.rpe_0p5_trans_rmse = float(s.split(":")[1])
                    continue
                if mode == "RPE_2p0_trans" and s.startswith("rmse"):
                    stats.rpe_2p0_trans_rmse = float(s.split(":")[1])
                    continue
                if mode == "SYS":
                    if s.startswith("cpu_avg"):
                        stats.cpu_avg = float(s.split(":")[1])
                        continue
                    if s.startswith("gpu_avg"):
                        stats.gpu_avg = float(s.split(":")[1])
                        continue
                    if s.startswith("ram_avg"):
                        stats.ram_avg = float(s.split(":")[1])
                        continue
    except Exception as e:
        log(f"[DASH][ERR] Failed to parse {path}: {e}")
        return None

    return stats


def collect_dashboard_stats_for_seq(seq_name: str) -> Dict[str, DashboardStats]:
    """
    For a given seq_name (e.g. test_0), load dashboard stats for all algorithms
    that have a DASHBOARD_<drive>.txt file.
    """
    drive_name = TEST_DRIVES.get(seq_name, None)
    if drive_name is None:
        log(f"[DASH] Unknown seq_name for dashboards: {seq_name}")
        return {}

    stats_by_algo: Dict[str, DashboardStats] = {}
    for algo_name, root in ALG_DIRS.items():
        seq_dir = root / seq_name
        dash_path = seq_dir / f"DASHBOARD_{drive_name}.txt"
        if not dash_path.exists():
            continue
        stats = parse_dashboard(algo_name, seq_name, drive_name, dash_path)
        if stats is not None:
            stats_by_algo[algo_name] = stats

    return stats_by_algo


# Simple generic bar plotting helper
def _plot_bar_for_metric(
    stats_by_algo: Dict[str, DashboardStats],
    field: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    algos = []
    vals = []
    for algo, st in stats_by_algo.items():
        v = getattr(st, field, None)
        if v is None:
            continue
        algos.append(algo)
        vals.append(v)

    if not algos:
        log(f"[PLOT] No data for metric {field} at {out_path}")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(algos))
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
        log(f"[OK] Saved summary plot -> {out_path}")
    except Exception as e:
        log(f"[ERR] Failed to save summary plot {out_path}: {e}")
    finally:
        plt.close(fig)


def _plot_bar_for_metric_logy(
    stats_by_algo: Dict[str, DashboardStats],
    field: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """
    Same as _plot_bar_for_metric, but with logarithmic y-axis.
    Useful for wall_time_sec when algorithms differ by orders of magnitude.
    """
    algos = []
    vals = []
    for algo, st in stats_by_algo.items():
        v = getattr(st, field, None)
        if v is None or v <= 0:
            # log-scale can't handle non-positive values
            continue
        algos.append(algo)
        vals.append(v)

    if not algos:
        log(f"[PLOT] No positive data for metric {field} at {out_path}")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(algos))
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
        log(f"[OK] Saved summary log plot -> {out_path}")
    except Exception as e:
        log(f"[ERR] Failed to save summary log plot {out_path}: {e}")
    finally:
        plt.close(fig)


def make_summary_plots_for_seq(seq_name: str) -> None:
    """
    Make per-sequence summary graphs for:
      - wall_time_sec (linear + optional log)
      - avg_fps
      - ate_rmse
      - rpe_0p5_trans_rmse
      - rpe_2p0_trans_rmse
      - cpu_avg
      - gpu_avg
      - ram_avg
    Saved under: OUTPUT_ROOT / "Graphs" / "Summary" / <seq_name>/
    """
    stats_by_algo = collect_dashboard_stats_for_seq(seq_name)
    if not stats_by_algo:
        log(f"[PLOT] No dashboard stats found for seq {seq_name}, skipping summary plots.")
        return

    drive_name = TEST_DRIVES[seq_name]
    base_dir = OUTPUT_ROOT / "Graphs" / "Summary" / seq_name

    _plot_bar_for_metric(
        stats_by_algo,
        field="wall_time_sec",
        title=f"Inference Time (wall time) - {seq_name} ({drive_name})",
        ylabel="Seconds",
        out_path=base_dir / f"WALL_TIME_{drive_name}.png",
    )

    # Optional log-scale wall time for large dynamic range
    _plot_bar_for_metric_logy(
        stats_by_algo,
        field="wall_time_sec",
        title=f"Inference Time (wall time, log scale) - {seq_name} ({drive_name})",
        ylabel="Seconds (log scale)",
        out_path=base_dir / f"WALL_TIME_LOG_{drive_name}.png",
    )

    _plot_bar_for_metric(
        stats_by_algo,
        field="avg_fps",
        title=f"Average FPS - {seq_name} ({drive_name})",
        ylabel="Frames per second",
        out_path=base_dir / f"FPS_{drive_name}.png",
    )

    _plot_bar_for_metric(
        stats_by_algo,
        field="ate_rmse",
        title=f"ATE RMSE - {seq_name} ({drive_name})",
        ylabel="ATE RMSE (m)",
        out_path=base_dir / f"ATE_RMSE_{drive_name}.png",
    )

    _plot_bar_for_metric(
        stats_by_algo,
        field="rpe_0p5_trans_rmse",
        title=f"RPE 0.5s Translational RMSE - {seq_name} ({drive_name})",
        ylabel="RPE 0.5s RMSE (m)",
        out_path=base_dir / f"RPE_0p5_TRANS_{drive_name}.png",
    )

    _plot_bar_for_metric(
        stats_by_algo,
        field="rpe_2p0_trans_rmse",
        title=f"RPE 2.0s Translational RMSE - {seq_name} ({drive_name})",
        ylabel="RPE 2.0s RMSE (m)",
        out_path=base_dir / f"RPE_2p0_TRANS_{drive_name}.png",
    )

    _plot_bar_for_metric(
        stats_by_algo,
        field="cpu_avg",
        title=f"CPU Usage (avg) - {seq_name} ({drive_name})",
        ylabel="CPU avg (%)",
        out_path=base_dir / f"CPU_AVG_{drive_name}.png",
    )

    _plot_bar_for_metric(
        stats_by_algo,
        field="gpu_avg",
        title=f"GPU Usage (avg) - {seq_name} ({drive_name})",
        ylabel="GPU avg (%)",
        out_path=base_dir / f"GPU_AVG_{drive_name}.png",
    )

    _plot_bar_for_metric(
        stats_by_algo,
        field="ram_avg",
        title=f"RAM Usage (avg) - {seq_name} ({drive_name})",
        ylabel="RAM avg (%)",
        out_path=base_dir / f"RAM_AVG_{drive_name}.png",
    )


# ============================================================
# Helpers: test frames & timestamps
# ============================================================

def get_test_image_paths(seq_name: str, drive_name: str) -> List[Path]:
    """
    Get the image_00/data_rect paths for this test sequence.
    """
    img_dir = DATA_2D_TEST / seq_name / drive_name / "image_00" / "data_rect"
    if not img_dir.exists():
        raise FileNotFoundError(f"Image dir not found for test sequence: {img_dir}")

    img_paths = sorted(
        list(img_dir.glob("*.png"))
        + list(img_dir.glob("*.jpg"))
        + list(img_dir.glob("*.jpeg"))
    )
    if not img_paths:
        raise RuntimeError(f"No images found in {img_dir}")

    return img_paths


def get_test_frame_indices(seq_name: str, drive_name: str) -> List[int]:
    """
    From image filenames under data_2d_test_slam, return integer frame indices
    (e.g. '0000001482.png' -> 1482).
    """
    img_paths = get_test_image_paths(seq_name, drive_name)
    frame_indices: List[int] = []
    for p in img_paths:
        try:
            frame_indices.append(int(p.stem))
        except ValueError:
            continue

    if not frame_indices:
        raise RuntimeError(f"Could not parse any frame indices from {seq_name}/{drive_name}")

    frame_indices = sorted(frame_indices)
    log(
        f"[GT-CROP] {seq_name}/{drive_name}: "
        f"{len(frame_indices)} frames from {frame_indices[0]} to {frame_indices[-1]}"
    )
    return frame_indices


def get_frameidx_to_time_rel(seq_name: str, drive_name: str) -> Dict[int, float]:
    """
    Build a mapping: frame_idx -> relative time (seconds),
    using image_00/timestamps.txt, mimicking run_slam_stereo.

    Assumes timestamps.txt lines correspond to sorted image filenames.
    """
    base_data_dir = DATA_2D_TEST / seq_name / drive_name
    ts_path = base_data_dir / "image_00" / "timestamps.txt"
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing image timestamps: {ts_path}")

    # Read timestamps (absolute) and convert to float seconds
    ts_abs: List[float] = []
    with ts_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ts_abs.append(parse_kitti360_timestamp(line))

    if not ts_abs:
        raise RuntimeError(f"No timestamps in {ts_path}")

    t0 = ts_abs[0]
    ts_rel = [t - t0 for t in ts_abs]

    # Match timestamps to filenames by order, like run_slam_stereo
    img_paths = get_test_image_paths(seq_name, drive_name)
    if len(ts_rel) < len(img_paths):
        log(
            f"[WARN] Only {len(ts_rel)} timestamps for {len(img_paths)} frames; "
            f"truncating to min length."
        )
        img_paths = img_paths[:len(ts_rel)]

    frameidx_to_time: Dict[int, float] = {}
    for p, t_rel in zip(img_paths, ts_rel):
        try:
            frame_idx = int(p.stem)
        except ValueError:
            continue
        frameidx_to_time[frame_idx] = t_rel

    if not frameidx_to_time:
        raise RuntimeError(
            f"Could not build frame_idx -> time mapping for {seq_name}/{drive_name}"
        )

    log(
        f"[TIME] {seq_name}/{drive_name}: built {len(frameidx_to_time)} "
        f"frame->time mappings"
    )

    return frameidx_to_time


# ============================================================
# GT loader: crop poses.txt using test frames + real timestamps
# ============================================================

def load_cropped_gt_for_seq(seq_name: str, drive_name: str) -> metrics.Trajectory:
    """
    Load GT from data_poses/<drive_name>/poses.txt, but ONLY for the frame
    indices that appear in the test subset. Assign timestamps using
    image_00/timestamps.txt so they live in the same time frame as the SLAM
    predictions.

    poses.txt format:
        frame_idx r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
    """
    used_frame_indices = get_test_frame_indices(seq_name, drive_name)
    used_set = set(used_frame_indices)

    frameidx_to_time = get_frameidx_to_time_rel(seq_name, drive_name)

    poses_path = DATA_POSES / drive_name / "poses.txt"
    assert poses_path.exists(), f"GT poses file not found: {poses_path}"

    frame_ids: List[int] = []
    t_list: List[float] = []
    xyz_list: List[np.ndarray] = []
    quat_list: List[np.ndarray] = []

    with poses_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 13:
                continue

            try:
                frame_idx = int(parts[0])
                mat_vals = list(map(float, parts[1:13]))
            except ValueError:
                continue

            if frame_idx not in used_set:
                continue
            if frame_idx not in frameidx_to_time:
                # frame outside the available timestamps (should be rare)
                continue

            T = np.array(mat_vals, dtype=float).reshape(3, 4)
            R = T[:, :3]
            t_vec = T[:, 3]
            q = rot_to_quat(R)  # [qx, qy, qz, qw]

            frame_ids.append(frame_idx)
            t_list.append(frameidx_to_time[frame_idx])
            xyz_list.append(t_vec)
            quat_list.append(q)

    if not frame_ids:
        raise RuntimeError(
            f"No matching GT poses (with timestamps) found in {poses_path} "
            f"for test subset frames of {seq_name}/{drive_name}"
        )

    # Sort by frame index (and reorder everything)
    frame_ids_arr = np.array(frame_ids, dtype=int)
    order = np.argsort(frame_ids_arr)
    frame_ids_arr = frame_ids_arr[order]
    t_arr = np.array(t_list, dtype=float)[order]
    xyz = np.vstack(xyz_list)[order].astype(float)
    quat = np.vstack(quat_list)[order].astype(float)

    log(
        f"[GT-CROP] {seq_name}/{drive_name}: using {len(frame_ids_arr)} GT poses "
        f"(frames {frame_ids_arr[0]}..{frame_ids_arr[-1]})"
    )

    return metrics.Trajectory(t=t_arr, xyz=xyz, quat=quat)


# ============================================================
# Predicted trajectory loader
# ============================================================

def find_pose_file(seq_out_dir: Path) -> Optional[Path]:
    """Find a POSES_*.txt (TUM) file in seq_out_dir (first match returned)."""
    if not seq_out_dir.exists():
        return None

    # Preferred: POSES_*.txt
    for p in seq_out_dir.glob("POSES_*.txt"):
        if p.is_file():
            return p

    # Fallbacks: trajectory*.txt or pose.txt (for MAST3R)
    for name in ("trajectory_v0.txt", "trajectory.txt", "pose.txt"):
        cand = seq_out_dir / name
        if cand.exists():
            return cand

    return None


def load_predicted_traj(pose_path: Path) -> Optional[metrics.Trajectory]:
    """Load predicted trajectory in standard TUM format."""
    try:
        return metrics.load_tum_trajectory(pose_path)
    except Exception as e:
        log(f"[ERR] Failed to load predicted traj {pose_path}: {e}")
        return None


# ============================================================
# Alignment: time-based (like compute_ate)
# ============================================================

def align_pred_to_gt_time_based(
    gt_traj: metrics.Trajectory,
    pred_traj: metrics.Trajectory,
    with_scale: bool = True,
    max_dt: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[str]]:
    """
    Align predicted trajectory to GT using the SAME logic as metrics.compute_ate:

      - Use associate_by_time(gt.t, pred.t, max_dt) to get matched pairs.
      - Compute Sim(3) on those matched XYZs via align_xyz_umeyama.
      - Apply that transform to ALL predicted poses (for plotting).

    IMPORTANT:
      - The Sim(3) is estimated ONLY from the subset of GT poses that
        have a corresponding prediction (i.e., matched timestamps).
      - The full GT trajectory (for the whole test subset) is still
        returned for plotting the "complete map".
      - We also return the matched GT indices so we can color that subpath
        differently (e.g., green) in the plot.
    """
    # Full GT coordinates (for plotting the complete test path)
    gt_xyz_full = gt_traj.xyz.copy()

    # Use only matched timestamps for alignment
    idx_ref, idx_q = metrics.associate_by_time(gt_traj.t, pred_traj.t, max_dt=max_dt)
    if idx_ref.size < 2:
        log("[ALIGN] Not enough matched timestamps for alignment; plotting raw paths.")
        return gt_xyz_full, pred_traj.xyz, None, None

    # Subset of GT + predictions used for estimating the transform
    xyz_ref_match = gt_traj.xyz[idx_ref]
    xyz_q_match = pred_traj.xyz[idx_q]

    try:
        R, t, s = metrics.align_xyz_umeyama(xyz_ref_match, xyz_q_match, with_scale=with_scale)
    except Exception as e:
        log(f"[ALIGN] align_xyz_umeyama failed: {e}; plotting raw paths.")
        return gt_xyz_full, pred_traj.xyz, None, None

    # Apply to ALL predicted poses (even those not in the matched subset)
    pred_xyz_all_aligned = (s * (R @ pred_traj.xyz.T)).T + t

    info = f"N={len(idx_ref)}, scale={s:.3f}"
    log(f"[ALIGN] Time-based Sim(3) alignment (matched subset only): {info}")

    # Return full GT path, aligned predictions, and the GT indices that had matches
    return gt_xyz_full, pred_xyz_all_aligned, idx_ref, info


# ============================================================
# Plotting (single algorithm)
# ============================================================

def plot_topdown_xy(
    gt_xyz: np.ndarray,
    pred_xyz: np.ndarray,
    gt_matched_idx: Optional[np.ndarray],
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1) Full GT path (cropped test drive) in blue
    if gt_xyz.size > 0:
        ax.plot(
            gt_xyz[:, 0],
            gt_xyz[:, 1],
            "-o",
            markersize=2,
            linewidth=1.0,
            label="GT (test subset)",
            color="tab:blue",
        )

    # 2) Highlight GT segment where we have predictions in green
    if gt_matched_idx is not None and gt_matched_idx.size > 0 and gt_xyz.size > 0:
        matched_xyz = gt_xyz[gt_matched_idx]
        ax.plot(
            matched_xyz[:, 0],
            matched_xyz[:, 1],
            "-o",
            markersize=3,
            linewidth=2.0,
            label="GT (with prediction)",
            color="green",
        )

    # 3) Aligned prediction in orange
    if pred_xyz.size > 0:
        ax.plot(
            pred_xyz[:, 0],
            pred_xyz[:, 1],
            "-o",
            markersize=2,
            linewidth=1.0,
            label="Pred (aligned)",
            color="tab:orange",
        )

    # Start/end markers
    if gt_xyz.size > 0:
        ax.scatter([gt_xyz[0, 0]], [gt_xyz[0, 1]], c="green", s=40, label="GT start")
        ax.scatter([gt_xyz[-1, 0]], [gt_xyz[-1, 1]], c="red", s=40, label="GT end")
    if pred_xyz.size > 0:
        ax.scatter(
            [pred_xyz[0, 0]],
            [pred_xyz[0, 1]],
            c="lime",
            s=30,
            marker="x",
            label="Pred start",
        )
        ax.scatter(
            [pred_xyz[-1, 0]],
            [pred_xyz[-1, 1]],
            c="maroon",
            s=30,
            marker="x",
            label="Pred end",
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    # Autoscale with margin
    xs: List[float] = []
    ys: List[float] = []
    if gt_xyz.size > 0:
        xs.extend(gt_xyz[:, 0].tolist())
        ys.extend(gt_xyz[:, 1].tolist())
    if pred_xyz.size > 0:
        xs.extend(pred_xyz[:, 0].tolist())
        ys.extend(pred_xyz[:, 1].tolist())

    if xs and ys:
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        dx = maxx - minx
        dy = maxy - miny
        pad = max(dx, dy) * 0.05 if max(dx, dy) > 0 else 1.0
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)

    return fig


# ============================================================
# Main processing (per-algorithm)
# ============================================================

def process_algorithm(algo_name: str, root_dir: Path) -> None:
    """Process all sequence subfolders under root_dir and produce plots."""
    if not root_dir.exists():
        log(f"[SKIP] Algorithm {algo_name} root not found: {root_dir}")
        return

    for seq_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        seq_name = seq_dir.name  # e.g. "test_0"

        drive_name = TEST_DRIVES.get(seq_name, "_unknown_drive_")
        if drive_name == "_unknown_drive_":
            log(f"[WARN] seq_name {seq_name} not in TEST_DRIVES; skipping.")
            continue

        pose_file = find_pose_file(seq_dir)
        if pose_file is None:
            log(f"[SKIP] No poses file found in {seq_dir} (algo={algo_name})")
            continue

        log(f"[PROC] {algo_name} / {seq_name} -> drive={drive_name}, pred={pose_file.name}")

        # Load cropped GT with real timestamps (full test subset)
        try:
            gt_traj = load_cropped_gt_for_seq(seq_name, drive_name)
        except Exception as e:
            log(f"[WARN] Failed to load cropped GT for {seq_name}/{drive_name}: {e}")
            continue

        # Load prediction
        pred_traj = load_predicted_traj(pose_file)
        if pred_traj is None:
            log(f"[WARN] Could not load predicted trajectory for {pose_file}, skipping.")
            continue

        # Align by time (ATE-style): estimate using only matched subset,
        # but still return full GT path for plotting.
        gt_xyz_plot, pred_xyz_plot, gt_matched_idx, align_info = align_pred_to_gt_time_based(
            gt_traj=gt_traj,
            pred_traj=pred_traj,
            with_scale=True,
            max_dt=0.05,
        )

        # Output dir
        out_dir = OUTPUT_ROOT / "Graphs" / algo_name / seq_name
        out_dir.mkdir(parents=True, exist_ok=True)

        title = f"{algo_name} - {seq_name} ({drive_name})"
        if align_info is not None:
            title += f"\nTime-based alignment ({align_info})"

        fig = plot_topdown_xy(gt_xyz_plot, pred_xyz_plot, gt_matched_idx, title)

        out_png = out_dir / f"PATH_{drive_name}.png"
        try:
            fig.savefig(str(out_png), dpi=200, bbox_inches="tight")
            log(f"[OK] Saved plot -> {out_png}")
        except Exception as e:
            log(f"[ERR] Failed to save plot {out_png}: {e}")
        finally:
            plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    log("[START] Generating GT vs Predicted path plots (cropped GT, time-based alignment)...")

    # 1) Per-algorithm path plots
    for algo_name, root in ALG_DIRS.items():
        process_algorithm(algo_name, root)

    # 2) Summary bar charts per sequence (inference time, FPS, ATE, RPE, CPU/GPU/RAM)
    for seq_name in sorted(TEST_DRIVES.keys()):
        make_summary_plots_for_seq(seq_name)

    log("[DONE] Graph generation complete. Check output/Graphs/")


if __name__ == "__main__":
    main()
