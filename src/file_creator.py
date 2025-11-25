"""
file_creator.py

Centralized helpers for logging / printing and file-writing operations.
Runners import logging + write_* from here. Some helpers are thin wrappers
around utils to avoid duplicating parsing code.

This module intentionally avoids importing 'metrics' at module import time
to prevent circular imports (metrics imports file_creator). Any use of
metrics-only helpers/types is done via local imports inside functions.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Lightweight wrappers around utils (safe at import-time)
from utils import (
    log as _log,
    log_phase1_config as log_phase1_config,
    ensure_dirs_exist as ensure_dirs_exist,
    log_overall_summary as log_overall_summary,
    log_slam_summary as log_slam_summary,
    load_ply_xyz as load_ply_xyz,
    get_ply_vertex_count as get_ply_vertex_count,
    parse_kitti360_timestamp as parse_kitti360_timestamp,
)


def log(msg: str) -> None:
    """Print a log message (flush)."""
    return _log(msg)


# -------------------------
# File-writing functions
# -------------------------

def write_system_metrics_txt(path: Path, samples: List[Any]) -> None:
    """Save per-sample system metrics to a txt file."""
    lines = []
    header = (
        "# t cpu_percent ram_percent gpu_util_percent "
        "gpu_mem_used_mb gpu_mem_total_mb gpu_power_watts gpu_error\n"
    )
    lines.append(header)
    for s in samples:
        # accept duck-typed samples (e.g., metrics.SystemSample)
        gpu_util = getattr(s, "gpu_utilization_percent", None)
        gpu_mem_used = getattr(s, "gpu_memory_used_mb", None)
        gpu_mem_total = getattr(s, "gpu_memory_total_mb", None)
        gpu_power = getattr(s, "gpu_power_watts", None)
        gpu_err = getattr(s, "gpu_error", None)

        lines.append(
            f"{getattr(s, 't', 0.0):.6f} {getattr(s, 'cpu_percent', 0.0):.1f} {getattr(s, 'ram_percent', 0.0):.1f} "
            f"{gpu_util if gpu_util is not None else -1:.1f} "
            f"{gpu_mem_used if gpu_mem_used is not None else -1:.1f} "
            f"{gpu_mem_total if gpu_mem_total is not None else -1:.1f} "
            f"{gpu_power if gpu_power is not None else -1:.1f} "
            f"{gpu_err if gpu_err is not None else ''}\n"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.writelines(lines)


def write_ate_report(path: Path, ate: Any, extra_info: Optional[Dict[str, Any]] = None) -> None:
    """Save ATE summary to txt."""
    lines: List[str] = []
    if extra_info:
        lines.append("# Extra info:\n")
        for k, v in extra_info.items():
            lines.append(f"# {k}: {v}\n")
        lines.append("\n")

    lines.append("ATE metrics (meters):\n")
    lines.append(f"RMSE   {ate.rmse:.6f}\n")
    lines.append(f"Mean   {ate.mean:.6f}\n")
    lines.append(f"Median {ate.median:.6f}\n")
    lines.append(f"Max    {ate.max:.6f}\n")
    lines.append(f"Frames {ate.num_frames}\n")
    lines.append(f"Scale  {ate.scale:.6f}\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.writelines(lines)


def write_ate_detailed(path: Path, traj_ref: Any, traj_query: Any, ate: Any) -> None:
    """
    Save per-frame ATE:
    t_ref, t_query, err, x_ref,y_ref,z_ref, x_est_aligned,y_est_aligned,z_est_aligned
    """
    if ate.errors is None or ate.idx_ref is None or ate.idx_query is None:
        raise ValueError("ATE result does not contain detailed indices/errors.")

    # Import metrics lazily to avoid circular import at module import-time
    import metrics as _metrics  # type: ignore

    idx_ref = ate.idx_ref
    idx_q = ate.idx_query

    xyz_ref = traj_ref.xyz[idx_ref]
    xyz_q = traj_query.xyz[idx_q]

    R, t, s = _metrics.align_xyz_umeyama(
        xyz_ref,
        xyz_q,
        with_scale=(ate.scale != 1.0),
    )
    xyz_q_aligned = (s * (R @ xyz_q.T)).T + t

    lines: List[str] = []
    header = (
        "# t_ref t_query err "
        "x_ref y_ref z_ref x_est_aligned y_est_aligned z_est_aligned\n"
    )
    lines.append(header)
    for k in range(len(idx_ref)):
        i_ref = int(idx_ref[k])
        i_q = int(idx_q[k])
        t_ref = float(traj_ref.t[i_ref])
        t_q = float(traj_query.t[i_q])
        err = float(ate.errors[k])
        xr, yr, zr = xyz_ref[k]
        xe, ye, ze = xyz_q_aligned[k]
        lines.append(
            f"{t_ref:.9f} {t_q:.9f} {err:.6f} "
            f"{xr:.6f} {yr:.6f} {zr:.6f} "
            f"{xe:.6f} {ye:.6f} {ze:.6f}\n"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.writelines(lines)


def write_rpe_report(path: Path, rpe: Any) -> None:
    """Save RPE summary to txt."""
    lines: List[str] = []
    lines.append(f"RPE metrics (delta_t = {rpe.delta_t:.3f} sec):\n")
    lines.append("Translational (meters):\n")
    lines.append(f"  RMSE   {rpe.trans_rmse:.6f}\n")
    lines.append(f"  Mean   {rpe.trans_mean:.6f}\n")
    lines.append(f"  Median {rpe.trans_median:.6f}\n")
    lines.append("Rotational (deg):\n")
    lines.append(f"  RMSE   {rpe.rot_rmse_deg:.6f}\n")
    lines.append(f"  Mean   {rpe.rot_mean_deg:.6f}\n")
    lines.append(f"  Median {rpe.rot_median_deg:.6f}\n")
    lines.append(f"Pairs  {rpe.num_pairs}\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.writelines(lines)


def write_rpe_detailed(path: Path, rpe: Any) -> None:
    """Save per-pair RPE: t0, t1, trans_err, rot_err_deg."""
    if (
        rpe.trans_errors is None
        or rpe.rot_errors_deg is None
        or rpe.t0 is None
        or rpe.t1 is None
    ):
        raise ValueError("RPE result does not contain detailed arrays.")

    lines: List[str] = []
    lines.append("# t0 t1 trans_err rot_err_deg\n")
    for t0, t1, et, er in zip(rpe.t0, rpe.t1, rpe.trans_errors, rpe.rot_errors_deg):
        lines.append(f"{t0:.9f} {t1:.9f} {et:.6f} {er:.6f}\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.writelines(lines)


def write_dashboard(dashboard_path: Path, summary, stereo: bool = True) -> None:
    """Write high-level DASHBOARD_<drive>.txt (centralized)."""
    seq_idx = summary.seq_name.split("_")[-1]  # "0", "1", ...
    title = ("=== KITTI-360 DASHBOARD (stereo) ===\n" if stereo else "=== KITTI-360 DASHBOARD ===\n")

    lines: List[str] = []
    lines.append(title)
    lines.append(f"timestamp           : {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"sequence            : {summary.seq_name}\n")
    lines.append(f"test_index          : {seq_idx}\n")
    lines.append(f"drive_name          : {summary.drive_name}\n")
    lines.append("\n")
    lines.append(f"frames_dataset      : {summary.frames_dataset}\n")
    lines.append(f"frames_traj         : {summary.frames_traj}\n")
    lines.append(f"wall_time_sec       : {summary.wall_time:.3f}\n")
    lines.append(f"avg_fps             : {summary.avg_fps:.3f}\n")

    if summary.num_map_points is not None:
        lines.append(f"num_map_points      : {summary.num_map_points}\n")
    else:
        lines.append("num_map_points      : N/A (no MAP_*.ply available)\n")

    # ATE block
    if summary.ate is not None:
        ate = summary.ate
        lines.append("\nATE (meters):\n")
        lines.append(f"  rmse              : {ate.rmse:.6f}\n")
        lines.append(f"  mean              : {ate.mean:.6f}\n")
        lines.append(f"  median            : {ate.median:.6f}\n")
        lines.append(f"  max               : {ate.max:.6f}\n")
        lines.append(f"  frames_ate        : {ate.num_frames}\n")
        lines.append(f"  scale             : {ate.scale:.6f}\n")
    else:
        lines.append("\nATE                 : N/A (ATE computation failed)\n")

    # RPE helper
    def _append_rpe_block(r):
        lines.append(f"\nRPE (delta_t={r.delta_t:.2f}s) translational (m):\n")
        lines.append(f"  rmse              : {r.trans_rmse:.6f}\n")
        lines.append(f"  mean              : {r.trans_mean:.6f}\n")
        lines.append(f"  median            : {r.trans_median:.6f}\n")
        lines.append(f"RPE (delta_t={r.delta_t:.2f}s) rotational (deg):\n")
        lines.append(f"  rmse              : {r.rot_rmse_deg:.6f}\n")
        lines.append(f"  mean              : {r.rot_mean_deg:.6f}\n")
        lines.append(f"  median            : {r.rot_median_deg:.6f}\n")
        lines.append(f"  pairs             : {r.num_pairs}\n")

    if summary.rpe_short is not None:
        _append_rpe_block(summary.rpe_short)
    else:
        lines.append("\nRPE (short)         : N/A\n")

    if summary.rpe_long is not None:
        _append_rpe_block(summary.rpe_long)
    else:
        lines.append("\nRPE (long)          : N/A\n")

    # System usage
    if hasattr(summary, "sys_summary") and summary.sys_summary:
        lines.append("\nSystem usage (percent):\n")
        cpu_avg = summary.sys_summary.get("cpu_avg", -1.0)
        cpu_max = summary.sys_summary.get("cpu_max", -1.0)
        ram_avg = summary.sys_summary.get("ram_avg", -1.0)
        ram_max = summary.sys_summary.get("ram_max", -1.0)
        gpu_avg = summary.sys_summary.get("gpu_avg", None)
        gpu_max = summary.sys_summary.get("gpu_max", None)

        lines.append(f"  cpu_avg           : {cpu_avg:.1f}\n")
        lines.append(f"  cpu_max           : {cpu_max:.1f}\n")
        lines.append(f"  ram_avg           : {ram_avg:.1f}\n")
        lines.append(f"  ram_max           : {ram_max:.1f}\n")

        if gpu_avg is not None and gpu_max is not None:
            lines.append(f"  gpu_avg           : {gpu_avg:.1f}\n")
            lines.append(f"  gpu_max           : {gpu_max:.1f}\n")
        else:
            lines.append("  gpu_avg           : N/A\n")
            lines.append("  gpu_max           : N/A\n")

    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    with dashboard_path.open("w") as f:
        f.writelines(lines)

    _log(f"[DASHBOARD] Wrote dashboard -> {dashboard_path}")
