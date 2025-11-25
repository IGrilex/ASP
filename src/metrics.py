#!/usr/bin/env python3
"""
metrics.py

- System monitoring (CPU / RAM / GPU)
- Trajectory metrics (ATE, RPE) with TUM-style trajectories
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

# Keep purely computational helpers in utils
from utils import voxelize_points, rot_to_quat
from file_creator import load_ply_xyz, get_ply_vertex_count, log  # small helpers re-exported for convenience

# ----------------------------------------------------------------------
# GPU initialization (NVIDIA via NVML)
# ----------------------------------------------------------------------

try:
    import pynvml

    HAS_GPU_MONITORING = True
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        print(
            f"[WARN] NVML initialization failed. Error: {e}",
            file=sys.stderr,
            flush=True,
        )
        HAS_GPU_MONITORING = False
except ImportError:
    HAS_GPU_MONITORING = False
    if __name__ == "__main__":
        print("--- GPU Monitoring Warning ---", file=sys.stderr, flush=True)
        print(
            "[WARN] 'pynvml' not installed. Install with 'pip install pynvml' "
            "for GPU metrics (NVIDIA only).",
            file=sys.stderr,
            flush=True,
        )
        print("----------------------------", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class SystemSample:
    t: float
    cpu_percent: float
    ram_percent: float
    gpu_utilization_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_power_watts: Optional[float] = None
    gpu_error: Optional[str] = None


@dataclass
class ATEResult:
    rmse: float
    mean: float
    median: float
    max: float
    num_frames: int
    scale: float
    errors: Optional[np.ndarray] = None      # (N,)
    idx_ref: Optional[np.ndarray] = None     # (N,)
    idx_query: Optional[np.ndarray] = None   # (N,)


@dataclass
class RPEResult:
    trans_rmse: float
    trans_mean: float
    trans_median: float
    rot_rmse_deg: float
    rot_mean_deg: float
    rot_median_deg: float
    num_pairs: int
    delta_t: float
    trans_errors: Optional[np.ndarray] = None   # (N,)
    rot_errors_deg: Optional[np.ndarray] = None # (N,)
    t0: Optional[np.ndarray] = None             # (N,)
    t1: Optional[np.ndarray] = None             # (N,)


@dataclass
class Trajectory:
    t: np.ndarray    # (N,)
    xyz: np.ndarray  # (N, 3)
    quat: np.ndarray # (N, 4) qx,qy,qz,qw


# ----------------------------------------------------------------------
# System monitoring
# ----------------------------------------------------------------------

def get_cpu_and_ram_usage(t: Optional[float] = None) -> Dict[str, Any]:
    """Instantaneous system-wide CPU and RAM utilization (percentage)."""
    if t is None:
        t = time.time()

    cpu_percent = psutil.cpu_percent(interval=None)
    ram_info = psutil.virtual_memory()
    ram_percent = ram_info.percent

    return {
        "t": t,
        "cpu_percent": cpu_percent,
        "ram_percent": ram_percent,
    }


def get_gpu_usage(t: Optional[float] = None, gpu_index: int = 0) -> Dict[str, Any]:
    """Instantaneous GPU utilization and memory usage for a specific GPU."""
    if t is None:
        t = time.time()

    if not HAS_GPU_MONITORING:
        return {
            "t": t,
            "error": "GPU monitoring unavailable (pynvml not initialized/installed).",
        }

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = float(utilization.gpu)

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem_used_mb = mem_info.used / (1024 * 1024)
        gpu_mem_total_mb = mem_info.total / (1024 * 1024)

        metrics: Dict[str, Any] = {
            "t": t,
            "gpu_index": gpu_index,
            "gpu_utilization_percent": gpu_util,
            "gpu_memory_used_mb": gpu_mem_used_mb,
            "gpu_memory_total_mb": gpu_mem_total_mb,
        }

        try:
            power_draw_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            metrics["gpu_power_watts"] = power_draw_mw / 1000.0
        except pynvml.NVMLError_NotSupported:
            metrics["gpu_power_watts"] = None
        except Exception:
            pass

        return metrics

    except pynvml.NVMLError as e:
        return {"t": t, "error": f"NVML error for GPU {gpu_index}: {str(e)}"}
    except Exception as e:
        return {"t": t, "error": f"Unexpected GPU metrics error: {e}"}


def append_system_sample(
    cpu_log: List[SystemSample],
    gpu_log: List[SystemSample],
    t: float,
) -> None:
    """
    Append a SystemSample at time t.

    cpu_log and gpu_log can be the same list. If they are, append only once.
    """
    cpu = get_cpu_and_ram_usage(t)
    gpu = get_gpu_usage(t)

    sample = SystemSample(
        t=cpu["t"],
        cpu_percent=cpu["cpu_percent"],
        ram_percent=cpu["ram_percent"],
        gpu_utilization_percent=gpu.get("gpu_utilization_percent"),
        gpu_memory_used_mb=gpu.get("gpu_memory_used_mb"),
        gpu_memory_total_mb=gpu.get("gpu_memory_total_mb"),
        gpu_power_watts=(
            gpu.get("gpu_power_watts")
            if isinstance(gpu.get("gpu_power_watts"), float)
            else None
        ),
        gpu_error=gpu.get("error"),
    )

    if cpu_log is gpu_log:
        cpu_log.append(sample)
    else:
        cpu_log.append(sample)
        gpu_log.append(sample)


def summarize_system_metrics(samples: List[SystemSample]) -> Dict[str, float]:
    """
    Average / max CPU, RAM, and GPU utilization.

    Returns a dict with keys: cpu_avg, cpu_max, ram_avg, ram_max, (optional) gpu_avg, gpu_max
    """
    if not samples:
        return {}

    cpu_vals = np.array([s.cpu_percent for s in samples], dtype=float)
    ram_vals = np.array([s.ram_percent for s in samples], dtype=float)
    gpu_vals = np.array(
        [s.gpu_utilization_percent for s in samples if s.gpu_utilization_percent is not None],
        dtype=float,
    )

    summary: Dict[str, float] = {
        "cpu_avg": float(cpu_vals.mean()),
        "cpu_max": float(cpu_vals.max()),
        "ram_avg": float(ram_vals.mean()),
        "ram_max": float(ram_vals.max()),
    }

    if gpu_vals.size > 0:
        summary["gpu_avg"] = float(gpu_vals.mean())
        summary["gpu_max"] = float(gpu_vals.max())

    return summary


# ----------------------------------------------------------------------
# Pose evaluation (ATE) using TUM-style trajectories
# ----------------------------------------------------------------------

def load_tum_trajectory(path: Path) -> Trajectory:
    """Load a TUM-format trajectory: t x y z qx qy qz qw."""
    data = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            t, x, y, z, qx, qy, qz, qw = map(float, parts[:8])
            data.append((t, x, y, z, qx, qy, qz, qw))

    if not data:
        raise RuntimeError(f"No valid TUM data in {path}")

    arr = np.array(data, dtype=float)
    return Trajectory(
        t=arr[:, 0],
        xyz=arr[:, 1:4],
        quat=arr[:, 4:8],
    )


def associate_by_time(
    t_ref: np.ndarray,
    t_query: np.ndarray,
    max_dt: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Associate two timestamp arrays by nearest neighbor.
    Returns indices into ref and query.
    """
    idx_ref: List[int] = []
    idx_query: List[int] = []

    j = 0
    for i in range(len(t_ref)):
        while j + 1 < len(t_query) and t_query[j + 1] <= t_ref[i]:
            j += 1

        candidates = [j]
        if j + 1 < len(t_query):
            candidates.append(j + 1)

        best_k = None
        best_dt = None
        for k in candidates:
            dt = abs(t_query[k] - t_ref[i])
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_k = k

        if best_k is not None and best_dt is not None and best_dt <= max_dt:
            idx_ref.append(i)
            idx_query.append(best_k)

    return np.array(idx_ref, dtype=int), np.array(idx_query, dtype=int)


def align_xyz_umeyama(
    xyz_ref: np.ndarray,
    xyz_query: np.ndarray,
    with_scale: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Align query onto ref using Umeyama’s method:
      x_ref ≈ s * R * x_query + t
    """
    assert xyz_ref.shape == xyz_query.shape
    n = xyz_ref.shape[0]

    mu_ref = xyz_ref.mean(axis=0)
    mu_query = xyz_query.mean(axis=0)

    X = xyz_query - mu_query
    Y = xyz_ref - mu_ref

    Sigma = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    if with_scale:
        var_query = (X ** 2).sum() / n
        s = (D @ S.diagonal()) / var_query
    else:
        s = 1.0

    t = mu_ref - s * R @ mu_query
    return R, t, s


def compute_ate(
    traj_ref: Trajectory,
    traj_query: Trajectory,
    max_dt: float = 0.05,
    with_scale: bool = False,
) -> ATEResult:
    """Absolute Trajectory Error (RMSE, mean, median, max) + detailed info."""
    idx_ref, idx_q = associate_by_time(traj_ref.t, traj_query.t, max_dt=max_dt)
    if len(idx_ref) == 0:
        raise RuntimeError("No matching timestamps between GT and prediction.")

    xyz_ref = traj_ref.xyz[idx_ref]
    xyz_q = traj_query.xyz[idx_q]

    R, t, s = align_xyz_umeyama(xyz_ref, xyz_q, with_scale=with_scale)
    xyz_q_aligned = (s * (R @ xyz_q.T)).T + t

    errors = np.linalg.norm(xyz_ref - xyz_q_aligned, axis=1)

    return ATEResult(
        rmse=float(np.sqrt((errors ** 2).mean())),
        mean=float(errors.mean()),
        median=float(np.median(errors)),
        max=float(errors.max()),
        num_frames=int(len(errors)),
        scale=float(s),
        errors=errors,
        idx_ref=idx_ref,
        idx_query=idx_q,
    )


def compute_ate_from_files(
    gt_path: Path,
    pred_path: Path,
    max_dt: float = 0.05,
    with_scale: bool = False,
) -> ATEResult:
    gt = load_tum_trajectory(gt_path)
    pred = load_tum_trajectory(pred_path)
    return compute_ate(gt, pred, max_dt=max_dt, with_scale=with_scale)


# ----------------------------------------------------------------------
# RPE (Relative Pose Error)
# ----------------------------------------------------------------------

def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Quaternion [qx,qy,qz,qw] -> 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=float,
    )


def rotation_matrix_to_angle_deg(R: np.ndarray) -> float:
    """Return rotation angle of R in degrees."""
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(tr)
    return float(np.degrees(angle))


def relative_se3(
    t: np.ndarray,
    xyz: np.ndarray,
    quat: np.ndarray,
    i: int,
    j: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Relative pose T_i^{-1} T_j as (R_rel, t_rel)."""
    pi = xyz[i]
    pj = xyz[j]

    qi = quat[i]
    qj = quat[j]

    Ri = quat_to_R(qi)
    Rj = quat_to_R(qj)

    R_rel = Ri.T @ Rj
    t_rel = Ri.T @ (pj - pi)
    return R_rel, t_rel


def compute_rpe(
    traj_ref: Trajectory,
    traj_query: Trajectory,
    delta_t: float = 1.0,
    max_dt: float = 0.05,
) -> RPEResult:
    """
    Relative Pose Error over a time interval delta_t (sec).

    Translational error = || t_ref_rel - t_query_rel ||
    Rotational error = angle( R_ref_rel * R_query_rel^T )
    """
    idx_ref, idx_q = associate_by_time(traj_ref.t, traj_query.t, max_dt=max_dt)
    if len(idx_ref) < 2:
        raise RuntimeError("Not enough matched timestamps for RPE.")

    t_ref_assoc = traj_ref.t[idx_ref]

    trans_errs: List[float] = []
    rot_errs_deg: List[float] = []
    t0_list: List[float] = []
    t1_list: List[float] = []

    for k in range(len(t_ref_assoc)):
        t0 = t_ref_assoc[k]
        t_target = t0 + delta_t

        j = np.searchsorted(t_ref_assoc, t_target)
        candidates: List[int] = []
        if j < len(t_ref_assoc):
            candidates.append(j)
        if j > 0:
            candidates.append(j - 1)

        if not candidates:
            continue

        best = None
        best_dt = None
        for c in candidates:
            dt = abs(t_ref_assoc[c] - t_target)
            if best is None or dt < best_dt:
                best = c
                best_dt = dt

        if best is None or best == k:
            continue

        if best_dt > max(delta_t * 0.5, max_dt):
            continue

        i_ref0 = idx_ref[k]
        i_ref1 = idx_ref[best]
        i_q0 = idx_q[k]
        i_q1 = idx_q[best]

        R_ref, t_ref_rel = relative_se3(traj_ref.t, traj_ref.xyz, traj_ref.quat, i_ref0, i_ref1)
        R_q, t_q_rel = relative_se3(traj_query.t, traj_query.xyz, traj_query.quat, i_q0, i_q1)

        t_err = t_ref_rel - t_q_rel
        trans_err = float(np.linalg.norm(t_err))

        R_err = R_ref @ R_q.T
        rot_err_deg = rotation_matrix_to_angle_deg(R_err)

        trans_errs.append(trans_err)
        rot_errs_deg.append(rot_err_deg)
        t0_list.append(traj_ref.t[i_ref0])
        t1_list.append(traj_ref.t[i_ref1])

    if not trans_errs:
        raise RuntimeError("No valid pairs for RPE computation.")

    trans_arr = np.array(trans_errs, dtype=float)
    rot_arr = np.array(rot_errs_deg, dtype=float)
    t0_arr = np.array(t0_list, dtype=float)
    t1_arr = np.array(t1_list, dtype=float)

    return RPEResult(
        trans_rmse=float(np.sqrt((trans_arr ** 2).mean())),
        trans_mean=float(trans_arr.mean()),
        trans_median=float(np.median(trans_arr)),
        rot_rmse_deg=float(np.sqrt((rot_arr ** 2).mean())),
        rot_mean_deg=float(rot_arr.mean()),
        rot_median_deg=float(np.median(rot_arr)),
        num_pairs=int(len(trans_arr)),
        delta_t=float(delta_t),
        trans_errors=trans_arr,
        rot_errors_deg=rot_arr,
        t0=t0_arr,
        t1=t1_arr,
    )

# ----------------------------------------------------------------------
# Simple CLI monitor (for quick sanity checks)
# ----------------------------------------------------------------------

def monitor_system(duration_seconds: int = 5, interval_seconds: float = 1.0) -> List[SystemSample]:
    samples: List[SystemSample] = []
    start = time.time()
    while time.time() - start < duration_seconds:
        t = time.time()
        append_system_sample(samples, samples, t)
        print(
            f"t={t:.3f} cpu={samples[-1].cpu_percent:.1f}% "
            f"ram={samples[-1].ram_percent:.1f}%",
            flush=True,
        )
        time.sleep(interval_seconds)
    return samples


if __name__ == "__main__":
    s = monitor_system(5, 1.0)
    summary = summarize_system_metrics(s)
    print(summary)
    if HAS_GPU_MONITORING:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
