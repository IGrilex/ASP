#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
import time
import runpy
import subprocess
from pathlib import Path
from typing import List, Optional

from paths import (
    DATA_2D_TEST,
    DATA_POSES,
    OUT_MAST3R,
    TEST_DRIVES,
    TEST_SCENARIOS,
)

# Add metric/reporting imports (used by run_mono_ORBSlam3)
from dataclasses import dataclass
from file_creator import (
    write_ate_report,
    write_ate_detailed,
    write_rpe_report,
    write_rpe_detailed,
    write_dashboard,
    get_ply_vertex_count,
    write_system_metrics_txt,  # added
)
from utils import load_kitti360_cam0_to_world_as_traj
from metrics import load_tum_trajectory, compute_ate, compute_rpe
import psutil  # added for sampling

# Adjust this to your MASt3R-SLAM repo root
MAST3R_ROOT = Path("/home/igrilex/ASP-SLAM/MASt3R-SLAM").resolve()
SHIM_NO_VIEWER = Path("/home/igrilex/ASP-SLAM/ASP/shim_no_viewer").resolve()

# new: MASt3R config file (common location in repo)
CFG_MAST3R = MAST3R_ROOT / "config" / "base.yaml"


def log(msg: str) -> None:
    print(msg, flush=True)


def find_newest_with_ext(roots: List[Path], started_at: float, ext: str) -> Optional[Path]:
    cands: List[Path] = []
    for r in roots:
        if not r.exists():
            continue
        for p in r.rglob(f"*{ext}"):
            try:
                if p.stat().st_mtime > started_at:
                    cands.append(p)
            except Exception:
                pass
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def run_master_for_sequence(seq_name: str) -> None:
    """
    Phase 1: run MASt3R headless on KITTI-360 input for seq_name,
    collect TUM poses and (if present) PLY map, move them to OUT_MAST3R/<seq_name>.
    Phase 2: run pose benchmark (ATE) using run_kitti360_benchmarks.
    """
    drive_name = TEST_DRIVES[seq_name]
    log(f"\n=== [PHASE 1] MASt3R-SLAM: {seq_name} / {drive_name} ===")

    # Use the drive folder as dataset root (same level as image_00)
    drive_root = (DATA_2D_TEST / seq_name / drive_name).resolve()
    assert drive_root.is_dir(), f"Drive root not found: {drive_root}"

    # Prefer explicit image folder required by MASt3R: image_00/data_rect (PNG images)
    image_dir = (drive_root / "image_00" / "data_rect").resolve()
    if not image_dir.exists():
        # fallback to image_00 if data_rect not present, else use drive_root
        alt = drive_root / "image_00"
        if alt.exists():
            image_dir = alt.resolve()
        else:
            image_dir = drive_root

    # dataset_root kept for compatibility naming elsewhere; we pass image_dir to MASt3R
    dataset_root = image_dir
    out_dir = (OUT_MAST3R / seq_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # sanity checks: ensure images exist and are readable
    img_candidates = list(dataset_root.rglob("*.png")) + list(dataset_root.rglob("*.jpg")) + list(dataset_root.rglob("*.jpeg"))
    if not img_candidates:
        raise AssertionError(f"No image files found under dataset path passed to MASt3R: {dataset_root}")
    # try opening the first image to detect format problems (optional dependency)
    try:
        from PIL import Image
        with Image.open(str(img_candidates[0])) as im:
            im.verify()
    except Exception:
        # not fatal here, but warn so you can inspect the image files manually
        log(f"[warn] Unable to open/verify sample image: {img_candidates[0]} (PIL may be missing or file is corrupted)")

    cam0_to_world = (DATA_POSES / drive_name / "cam0_to_world.txt").resolve()
    assert cam0_to_world.exists(), f"cam0_to_world.txt missing: {cam0_to_world}"

    log(f"[cfg] MASt3R repo : {MAST3R_ROOT}")
    log(f"[cfg] DATA root   : {dataset_root} (images passed to MASt3R)")
    log(f"[cfg] OUT dir     : {out_dir}")
    log(f"[cfg] CONFIG      : {CFG_MAST3R}")

    # Prepare environment for headless run (viewer shim)
    sys.path.insert(0, str(SHIM_NO_VIEWER))
    sys.path.insert(1, str(MAST3R_ROOT))
    cwd_prev = os.getcwd()
    os.chdir(str(MAST3R_ROOT))

    # Use supported CLI flags: --no-viz instead of --headless, provide --save-as and --config
    base = drive_name
    sys.argv = [
        "main.py",
        "--dataset", str(dataset_root),
        "--config", str(CFG_MAST3R),
        "--save-as", base,
        "--no-viz",
    ]
    log(f"[run] main.py args: {' '.join(sys.argv[1:])}")

    # Run MASt3R as a separate process (safer for multiprocessing)
    cmd = [
        sys.executable,
        "main.py",
        "--dataset", str(dataset_root),
        "--config", str(CFG_MAST3R),
        "--save-as", base,
        "--no-viz",
    ]
    # start subprocess and sample machine usage while it runs
    started_at = time.time()
    system_samples = []
    sample_interval = 0.5
    def _get_gpu_percent():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                text=True,
            )
            return float(out.strip().splitlines()[0])
        except Exception:
            return None

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(MAST3R_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # poll loop to sample usage while process runs
        try:
            while proc.poll() is None:
                t_rel = time.time() - started_at
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                gpu = _get_gpu_percent()
                system_samples.append({"t": t_rel, "cpu": cpu, "ram": ram, "gpu": gpu})
                time.sleep(sample_interval)
            # collect remaining output
            stdout_text, _ = proc.communicate(timeout=1)
        except Exception:
            # ensure process termination on sampler error
            try:
                proc.kill()
            except Exception:
                pass
            stdout_text, _ = proc.communicate()
        exit_code = proc.returncode
        # persist stdout for debugging
        try:
            (out_dir / "MASt3R_run_stdout.txt").write_text(stdout_text or "")
        except Exception:
            pass
        log(f"[MASt3R stdout written to] {out_dir / 'MASt3R_run_stdout.txt'}")
    except Exception as e:
        exit_code = 1
        log(f"[ERR] Failed to launch MASt3R subprocess: {e}")

    elapsed = time.time() - started_at
    log(f"[done] MASt3R finished exit={exit_code} elapsed={elapsed:.2f}s")

    # restore cwd (if changed earlier)
    try:
        os.chdir(cwd_prev)
    except Exception:
        pass

    # --- New: copy outputs from MASt3R logs (if present) so we can evaluate poses/maps ---
    logs_dir = MAST3R_ROOT / "logs" / base
    if logs_dir.exists():
        import shutil
        # copy/normalize trajectory txt (prefer data_rect.txt)
        data_txt = logs_dir / "data_rect.txt"
        if data_txt.exists():
            dst = out_dir / f"POSES_{base}.txt"
            try:
                lines = data_txt.read_text().splitlines()
                normalized: list[str] = []
                t0: Optional[float] = None

                # Expect lines: t x y z qx qy qz qw (8 floats)
                for ln in lines:
                    parts = ln.strip().split()
                    if len(parts) == 8:
                        try:
                            vals = [float(x) for x in parts]
                        except Exception:
                            continue
                        if t0 is None:
                            t0 = vals[0]
                        # Fix MASt3R time: shift to 0 and stretch by 3x
                        vals[0] = (vals[0] - t0) * 3.0
                        normalized.append(" ".join(f"{v:.12g}" for v in vals))

                if normalized:
                    dst.write_text("\n".join(normalized) + "\n")
                    log(f"[ok] normalized+copied log traj (3x time scale) -> {dst}")
                else:
                    # fallback to raw copy if normalization produced nothing
                    shutil.copy2(data_txt, dst)
                    log(f"[warn] normalization produced no lines; raw copy -> {dst}")
            except Exception as e:
                log(f"[warn] failed to copy/normalize {data_txt}: {e}")
        else:
            # fallback: any .txt in logs_dir
            txts = sorted(logs_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if txts:
                try:
                    dst = out_dir / f"POSES_{base}.txt"
                    lines = txts[0].read_text().splitlines()
                    normalized: list[str] = []
                    t0: Optional[float] = None

                    for ln in lines:
                        parts = ln.strip().split()
                        if len(parts) >= 8:
                            try:
                                vals = [float(x) for x in parts[:8]]
                            except Exception:
                                continue
                            if t0 is None:
                                t0 = vals[0]
                            # Fix MASt3R time: shift to 0 and stretch by 3x
                            vals[0] = (vals[0] - t0) * 3.0
                            normalized.append(" ".join(f"{v:.12g}" for v in vals))

                    if normalized:
                        dst.write_text("\n".join(normalized) + "\n")
                        log(f"[ok] normalized+copied newest log txt (3x time scale) -> {dst}")
                    else:
                        shutil.copy2(txts[0], dst)
                        log(f"[ok] copied newest log txt -> {dst}")
                except Exception as e:
                    log(f"[warn] failed to copy {txts[0]}: {e}")

        # copy ply map (prefer data_rect.ply)
        data_ply = logs_dir / "data_rect.ply"
        if data_ply.exists():
            try:
                dst = out_dir / f"MAP_{base}.ply"
                shutil.copy2(data_ply, dst)
                log(f"[ok] copied log ply -> {dst}")
            except Exception as e:
                log(f"[warn] failed to copy {data_ply}: {e}")
        else:
            plys = sorted(logs_dir.glob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
            if plys:
                try:
                    dst = out_dir / f"MAP_{base}.ply"
                    shutil.copy2(plys[0], dst)
                    log(f"[ok] copied newest log ply -> {dst}")
                except Exception as e:
                    log(f"[warn] failed to copy {plys[0]}: {e}")

        # copy keyframes folder (common path: logs/<drive>/keyframes/data_rect)
        kf_src = logs_dir / "keyframes"
        if kf_src.exists():
            kf_dst = out_dir / "keyframes"
            try:
                # copytree with overwrite behavior (Python 3.8+)
                if kf_dst.exists():
                    shutil.rmtree(kf_dst)
                shutil.copytree(kf_src, kf_dst)
                log(f"[ok] copied keyframes -> {kf_dst}")
            except Exception as e:
                log(f"[warn] failed to copy keyframes: {e}")
    else:
        log(f"[info] No logs dir found at {logs_dir} (skipping log copy)")
    # --- end logs copy block ---

    # Prefer any files we've already copied into out_dir (logs copy above).
    poses_copy = out_dir / f"POSES_{base}.txt"
    map_copy = out_dir / f"MAP_{base}.ply"
    tum_src = poses_copy if poses_copy.exists() else None
    ply_src = map_copy if map_copy.exists() else None

    # If not present, search typical output locations for new TUM trajectory (.txt) and PLY map (.ply)
    if not tum_src or not ply_src:
        search_roots = [
            MAST3R_ROOT / "outputs",
            MAST3R_ROOT / "results",
            MAST3R_ROOT / "logs" / base,
            dataset_root,
            dataset_root / "results",
        ]
        if not tum_src:
            tum_src = find_newest_with_ext(search_roots, started_at, ".txt")
        if not ply_src:
            ply_src = find_newest_with_ext(search_roots, started_at, ".ply")

    # Move/rename found outputs into standardized OUT_MAST3R dir
    if tum_src:
        poses_dst = out_dir / f"POSES_{base}.txt"
        try:
            # if source is already the destination, skip
            if tum_src.resolve() != poses_dst.resolve():
                tum_src.rename(poses_dst)
                log(f"[ok] moved poses -> {poses_dst}")
            else:
                log(f"[ok] poses already at -> {poses_dst}")
        except Exception:
            import shutil
            shutil.copy2(tum_src, poses_dst)
            log(f"[ok] copied poses -> {poses_dst}")
    else:
        log("[warn] No new TUM trajectory (.txt) found from MASt3R run.")

    if ply_src:
        map_dst = out_dir / f"MAP_{base}.ply"
        try:
            if ply_src.resolve() != map_dst.resolve():
                ply_src.rename(map_dst)
                log(f"[ok] moved map -> {map_dst}")
            else:
                log(f"[ok] map already at -> {map_dst}")
        except Exception:
            import shutil
            shutil.copy2(ply_src, map_dst)
            log(f"[ok] copied map -> {map_dst}")
    else:
        log("[info] No .ply map found from MASt3R run (ensure map saving is enabled in MASt3R config).")

    log(f"[info] MASt3R run completed for {seq_name}/{drive_name}. Skipping benchmark/evaluation as requested.")

    # Write machine usage file + compute summary (like run_mono_ORBSlam3 does)
    try:
        machine_path = out_dir / f"MACHINE_USAGE_{base}.txt"
        # custom CSV-like dump (timestamp(s) relative to process start, cpu, ram, gpu)
        lines = ["time_sec,cpu_percent,ram_percent,gpu_percent"]
        for s in system_samples:
            gpu_val = "" if s.get("gpu") is None else f"{s.get('gpu'):.1f}"
            lines.append(f"{s['t']:.3f},{s['cpu']:.1f},{s['ram']:.1f},{gpu_val}")
        machine_path.write_text("\n".join(lines) + "\n")
        # try to call canonical writer if available (best-effort)
        try:
            write_system_metrics_txt(machine_path, system_samples)
        except Exception:
            pass

        # compute summary dict similar to other runners
        cpu_vals = [float(s["cpu"]) for s in system_samples] if system_samples else []
        ram_vals = [float(s["ram"]) for s in system_samples] if system_samples else []
        gpu_vals = [float(s["gpu"]) for s in system_samples if s.get("gpu") is not None]
        def _avg_max(vals):
            if not vals:
                return None, None
            return float(sum(vals) / len(vals)), float(max(vals))
        cpu_avg, cpu_max = _avg_max(cpu_vals)
        ram_avg, ram_max = _avg_max(ram_vals)
        gpu_avg, gpu_max = _avg_max(gpu_vals)
        sys_summary = {}
        if cpu_avg is not None:
            sys_summary["cpu_avg"] = round(cpu_avg, 1)
            sys_summary["cpu_max"] = round(cpu_max, 1)
        if ram_avg is not None:
            sys_summary["ram_avg"] = round(ram_avg, 1)
            sys_summary["ram_max"] = round(ram_max, 1)
        if gpu_avg is not None:
            sys_summary["gpu_avg"] = round(gpu_avg, 1)
            sys_summary["gpu_max"] = round(gpu_max, 1)
        log(f"[ok] Machine usage written -> {machine_path}")
    except Exception as e:
        log(f"[warn] Failed to write machine usage: {e}")
        sys_summary = {}

    # -------------------------
    # Phase 2 - compute ATE / RPE + dashboard
    # -------------------------
    @dataclass
    class RunSummary:
        seq_name: str
        drive_name: str
        frames_dataset: int
        frames_traj: int
        wall_time: float
        avg_fps: float
        num_map_points: Optional[int]
        ate: Optional[object]
        rpe_short: Optional[object]
        rpe_long: Optional[object]
        sys_summary: dict

    try:
        poses_path = out_dir / f"POSES_{base}.txt"
        if not poses_path.exists():
            log(f"[PHASE2][warn] Predicted TUM poses not found ({poses_path}), skipping analysis.")
        else:
            try:
                gt_traj = load_kitti360_cam0_to_world_as_traj(DATA_POSES / drive_name / "cam0_to_world.txt")
                pred_traj = load_tum_trajectory(poses_path)

                ate = compute_ate(traj_ref=gt_traj, traj_query=pred_traj, max_dt=0.05, with_scale=True)
                ate_path = out_dir / f"ATE_{base}.txt"
                ate_err_path = out_dir / f"ATE_ERRORS_{base}.txt"
                write_ate_report(ate_path, ate, extra_info=None)
                write_ate_detailed(ate_err_path, gt_traj, pred_traj, ate)
                log(f"[PHASE2][ATE] Wrote -> {ate_path}, {ate_err_path}")

                rpe_short = compute_rpe(traj_ref=gt_traj, traj_query=pred_traj, delta_t=0.5, max_dt=0.05)
                rpe_long = compute_rpe(traj_ref=gt_traj, traj_query=pred_traj, delta_t=2.0, max_dt=0.05)

                rpe_short_path = out_dir / f"RPE_0p5s_{base}.txt"
                rpe_short_err_path = out_dir / f"RPE_0p5s_ERRORS_{base}.txt"
                rpe_long_path = out_dir / f"RPE_2p0s_{base}.txt"
                rpe_long_err_path = out_dir / f"RPE_2p0s_ERRORS_{base}.txt"

                write_rpe_report(rpe_short_path, rpe_short)
                write_rpe_detailed(rpe_short_err_path, rpe_short)
                write_rpe_report(rpe_long_path, rpe_long)
                write_rpe_detailed(rpe_long_err_path, rpe_long)
                log(f"[PHASE2][RPE] Wrote -> {rpe_short_path}, {rpe_long_path}")

                # basic frame counts and map vertex count (if any)
                frames_dataset = len(img_candidates)
                frames_traj = sum(1 for _ in poses_path.open("r") if _.strip())
                map_path = out_dir / f"MAP_{base}.ply"
                num_map_points = get_ply_vertex_count(map_path) if map_path.exists() else None

                # dashboard / summary
                elapsed_sec = elapsed if 'elapsed' in locals() else 0.0
                avg_fps = frames_dataset / elapsed_sec if elapsed_sec > 0 else 0.0
                summary = RunSummary(
                    seq_name=seq_name,
                    drive_name=drive_name,
                    frames_dataset=frames_dataset,
                    frames_traj=frames_traj,
                    wall_time=elapsed_sec,
                    avg_fps=avg_fps,
                    num_map_points=num_map_points,
                    ate=ate,
                    rpe_short=rpe_short,
                    rpe_long=rpe_long,
                    sys_summary=sys_summary,
                )
                dashboard_path = out_dir / f"DASHBOARD_{base}.txt"
                write_dashboard(dashboard_path, summary, stereo=False)
                log(f"[PHASE2] Dashboard written -> {dashboard_path}")
            except Exception as e:
                log(f"[PHASE2][ERR] Failed to compute ATE/RPE for {seq_name}/{drive_name}: {e}")
    except Exception as e:
        log(f"[PHASE2][ERR] Unexpected error in post-processing: {e}")


def main():
    # Run MASt3R on light scenario only
    SCENARIO = "light"
    seq_list = TEST_SCENARIOS[SCENARIO]

    for seq_name in seq_list:
        try:
            run_master_for_sequence(seq_name)
        except Exception as e:
            log(f"[ERR] Sequence {seq_name} failed: {e}")


if __name__ == "__main__":
    main()
