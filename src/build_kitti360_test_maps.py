#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import shutil

# centralized log
from file_creator import log

DATA_2D_TEST_SLAM = Path("/ASP/data/data_2d_test_slam")
AIRSLAM_K360_ROOT = Path("/ASP/data/airslam_kitti360")
TEST_DRIVES = {
    "test_0": "2013_05_28_drive_0008_sync",
    "test_1": "2013_05_28_drive_0008_sync",
    "test_2": "2013_05_28_drive_0004_sync",
    "test_3": "2013_05_28_drive_0002_sync",
}


def ensure_empty_dir(dir_path: Path) -> None:
    """
    Make sure dir_path exists and remove all *.png files in it.
    We don't delete the directory, just clean the PNGs so no old
    timestamp-named images remain.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    removed = 0
    for p in dir_path.glob("*.png"):
        try:
            p.unlink()
            removed += 1
        except Exception as e:
            log(f"[WARN] Failed to remove old file {p}: {e}")
    if removed > 0:
        log(f"[clean] Removed {removed} old PNGs in {dir_path}")


def copy_camera_stream(
    src_root: Path,
    cam_folder: str,
    dst_seq_root: Path,
) -> None:
    """
    Copy one camera stream (image_00 or image_01) into AirSLAM format.

    - src_root:  .../test_X/<drive_name>
    - cam_folder: "image_00" or "image_01"
    - dst_seq_root: /ASP/data/airslam_kitti360/test_X

    Result:
      /ASP/data/airslam_kitti360/test_X/cam0 or cam1:
        - data/*.png (original KITTI-360 frame-index filenames)
        - timestamps.txt (copied from image_0x/timestamps.txt)
    """
    cam_idx = 0 if cam_folder == "image_00" else 1
    dst_cam_dir = dst_seq_root / f"cam{cam_idx}"
    dst_cam_data = dst_cam_dir / "data"
    dst_cam_ts = dst_cam_dir / "timestamps.txt"

    src_cam_dir = src_root / cam_folder
    src_data_rect = src_cam_dir / "data_rect"
    src_ts = src_cam_dir / "timestamps.txt"

    assert src_data_rect.is_dir(), f"Missing source data_rect: {src_data_rect}"
    assert src_ts.exists(), f"Missing source timestamps.txt: {src_ts}"

    log(f"[cfg] {cam_folder} src data_rect: {src_data_rect}")
    log(f"[cfg] {cam_folder} src timestamps: {src_ts}")
    log(f"[cfg] cam{cam_idx} dst data      : {dst_cam_data}")
    log(f"[cfg] cam{cam_idx} dst timestamps: {dst_cam_ts}")

    # Clean old PNGs in destination
    ensure_empty_dir(dst_cam_data)

    src_pngs = sorted(src_data_rect.glob("*.png"))
    if not src_pngs:
        raise RuntimeError(f"No PNG images in {src_data_rect}")

    # Copy all images, preserving filenames (e.g. 0000001482.png)
    for i, src in enumerate(src_pngs):
        dst = dst_cam_data / src.name
        shutil.copy2(src, dst)
        if (i + 1) % 500 == 0 or (i + 1) == len(src_pngs):
            log(f"[prog] cam{cam_idx}: copied {i+1}/{len(src_pngs)} images")

    # Copy timestamps.txt verbatim
    dst_cam_ts.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ts, dst_cam_ts)
    log(f"[ok] cam{cam_idx}: copied timestamps -> {dst_cam_ts}")
    log(f"[ok] cam{cam_idx}: total images copied = {len(src_pngs)}")


def build_airslam_dataset_for_seq(seq_name: str) -> None:
    """
    Build a clean AirSLAM KITTI-360 dataset for one test sequence.

    For seq_name (e.g. "test_0"), and drive_name from TEST_DRIVES, we expect:

      DATA_2D_TEST_SLAM / seq_name / drive_name / image_00/data_rect/*.png
      DATA_2D_TEST_SLAM / seq_name / drive_name / image_00/timestamps.txt
      DATA_2D_TEST_SLAM / seq_name / drive_name / image_01/data_rect/*.png
      DATA_2D_TEST_SLAM / seq_name / drive_name / image_01/timestamps.txt

    We create:

      AIRSLAM_K360_ROOT / seq_name / cam0/data/*.png
      AIRSLAM_K360_ROOT / seq_name / cam0/timestamps.txt
      AIRSLAM_K360_ROOT / seq_name / cam1/data/*.png
      AIRSLAM_K360_ROOT / seq_name / cam1/timestamps.txt
    """
    assert seq_name in TEST_DRIVES, f"Unknown seq_name: {seq_name}"
    drive_name = TEST_DRIVES[seq_name]

    log(f"\n=== Building AirSLAM KITTI-360 dataset for {seq_name} / {drive_name} ===")

    src_root = (
        DATA_2D_TEST_SLAM
        / seq_name
        / drive_name
    ).resolve()

    assert src_root.is_dir(), f"Missing source drive root: {src_root}"

    dst_seq_root = (AIRSLAM_K360_ROOT / seq_name).resolve()
    dst_seq_root.mkdir(parents=True, exist_ok=True)

    log(f"[cfg] src_root (2D test_slam) : {src_root}")
    log(f"[cfg] dst_seq_root (AirSLAM)  : {dst_seq_root}")

    # Copy left (image_00 -> cam0)
    copy_camera_stream(
        src_root=src_root,
        cam_folder="image_00",
        dst_seq_root=dst_seq_root,
    )

    # Copy right (image_01 -> cam1)
    copy_camera_stream(
        src_root=src_root,
        cam_folder="image_01",
        dst_seq_root=dst_seq_root,
    )

    log(f"[DONE] AirSLAM dataset ready for {seq_name}")


def main() -> None:
    # Build for all sequences in TEST_DRIVES
    for seq_name in sorted(TEST_DRIVES.keys()):
        build_airslam_dataset_for_seq(seq_name)


if __name__ == "__main__":
    main()
