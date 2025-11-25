#!/usr/bin/env python3
"""
prep_data_4_AirSlam.py

Convert KITTI-360-style test data into AirSLAM's ASL dataset format:

    data/airslam_kitti360/<seq_name>/
        cam0/data/<timestamp_ns>.png
        cam1/data/<timestamp_ns>.png

Run from project root:  python3 prep_data_4_AirSlam.py
"""

from pathlib import Path
import shutil

from paths import DATA_2D_TEST, DATA_ROOT, TEST_DRIVES
from utils import parse_kitti360_timestamp

# Use centralized logger
from file_creator import log


def collect_images(img_dir):
    exts = ["*.png", "*.jpg", "*.jpeg"]
    imgs = []
    for e in exts:
        imgs.extend(img_dir.glob(e))
    return sorted(imgs)


def prep_one_sequence(seq_name: str, out_root: Path):
    drive_name = TEST_DRIVES[seq_name]
    log(f"\n=== Preparing {seq_name} / {drive_name} for AirSLAM ===")

    base_data_dir = (DATA_2D_TEST / seq_name / drive_name).resolve()
    img_dir_left = base_data_dir / "image_00" / "data_rect"
    img_dir_right = base_data_dir / "image_01" / "data_rect"
    ts_path = base_data_dir / "image_00" / "timestamps.txt"

    assert img_dir_left.is_dir(), f"Missing left image dir: {img_dir_left}"
    assert img_dir_right.is_dir(), f"Missing right image dir: {img_dir_right}"
    assert ts_path.exists(), f"Missing timestamps file: {ts_path}"

    imgs_left = collect_images(img_dir_left)
    if not imgs_left:
        raise RuntimeError(f"No images found in {img_dir_left}")

    log(f"[check] Found {len(imgs_left)} left images")

    # --- read timestamps and convert to float seconds ---
    ts_list = []
    with ts_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ts_list.append(parse_kitti360_timestamp(line))

    if len(ts_list) < len(imgs_left):
        log(
            f"[WARN] timestamps ({len(ts_list)}) < images ({len(imgs_left)}), "
            f"truncating to match images."
        )
    ts_list = ts_list[: len(imgs_left)]

    # --- create ASL-style dirs ---
    seq_root = out_root / seq_name
    cam0_dir = seq_root / "cam0" / "data"
    cam1_dir = seq_root / "cam1" / "data"
    cam0_dir.mkdir(parents=True, exist_ok=True)
    cam1_dir.mkdir(parents=True, exist_ok=True)

    log(f"[out] cam0 -> {cam0_dir}")
    log(f"[out] cam1 -> {cam1_dir}")

    # --- copy images with EuRoC/ASL-style timestamp filenames (nanoseconds) ---
    copied = 0
    for i, (imgL, t_sec) in enumerate(zip(imgs_left, ts_list)):
        imgR = img_dir_right / imgL.name
        if not imgR.exists():
            log(f"[WARN] Missing right image for frame {i}: {imgR}, skipping this pair.")
            continue

        # Convert seconds to nanoseconds integer (EuRoC-style)
        stamp_ns = int(round(t_sec * 1e9))
        fname = f"{stamp_ns}.png"

        dstL = cam0_dir / fname
        dstR = cam1_dir / fname

        shutil.copy2(imgL, dstL)
        shutil.copy2(imgR, dstR)

        copied += 1
        if (i + 1) % 200 == 0 or (i + 1) == len(imgs_left):
            log(f"[prog] {i+1}/{len(imgs_left)} frames processed, {copied} stereo pairs copied")

    log(f"[done] {seq_name}: copied {copied} stereo pairs.")


def main():
    project_root = Path(__file__).resolve().parent
    out_root = (DATA_ROOT / "airslam_kitti360").resolve()

    log(f"[cfg] DATA_2D_TEST = {DATA_2D_TEST}")
    log(f"[cfg] Output root  = {out_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    for seq_name in sorted(TEST_DRIVES.keys()):
        prep_one_sequence(seq_name, out_root)

    log("\nAll sequences done.")
    log("You can now point AirSLAM's 'dataroot' to one of:")
    for seq_name in sorted(TEST_DRIVES.keys()):
        log(f"  {out_root / seq_name}")


if __name__ == "__main__":
    main()
