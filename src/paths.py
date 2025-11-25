#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

# ============================================================
# Project roots
# ============================================================

# Repo root (this file lives in <repo>/src/ or similar)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------
# DATA (on host / local filesystem, under the repo)
# ------------------------------------------------------------

DATA_ROOT = PROJECT_ROOT / "data"

# 2D & 3D KITTI-360 data in repo layout
DATA_2D_TEST = DATA_ROOT / "data_2d_test_slam"        # test_* 2D images
DATA_POSES = DATA_ROOT / "data_poses"                # poses.txt, cam0_to_world.txt, ...
DATA_3D_SEM = DATA_ROOT / "data_3d_semantics" / "train"
DATA_3D_TEST_SLAM = DATA_ROOT / "data_3d_test_slam"

# Prebuilt GT maps
MAPS_ROOT = DATA_ROOT / "maps"
MAPS_TEST: Dict[str, Path] = {
    "test_0": MAPS_ROOT / "test_0",
    "test_1": MAPS_ROOT / "test_1",
    "test_2": MAPS_ROOT / "test_2",
    "test_3": MAPS_ROOT / "test_3",
}

# ------------------------------------------------------------
# KITTI-360 calibration (repo data/calibration)
# ------------------------------------------------------------

CALIB_ROOT = DATA_ROOT / "calibration"
CALIB_CAM_TO_VELO = CALIB_ROOT / "calib_cam_to_velo.txt"
CALIB_CAM_TO_POSE = CALIB_ROOT / "calib_cam_to_pose.txt"
CALIB_SICK_TO_VELO = CALIB_ROOT / "calib_sick_to_velo.txt"

# ============================================================
# ORB-SLAM3 & MAST3R config / output
# ============================================================

ORB3_ROOT = PROJECT_ROOT.parent / "ORB-SLAM3-python"
VOCAB = ORB3_ROOT / "third_party" / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt"

CONFIGS_ROOT = PROJECT_ROOT / "configs"
CFG_MONO = CONFIGS_ROOT / "mono_ORB_SLAM3.yaml"
CFG_STEREO = CONFIGS_ROOT / "stereo_ORB_SLAM3.yaml"

# Experiments output under repo
OUTPUT_ROOT = PROJECT_ROOT / "output"

# ORB-SLAM3 outputs
OUT_ORB3_MONO = OUTPUT_ROOT / "ORB3" / "mono"
OUT_ORB3_STEREO = OUTPUT_ROOT / "ORB3" / "stereo"

# MAST3R outputs
OUT_MAST3R = OUTPUT_ROOT / "MAST3R"

# ============================================================
# KITTI-360 sequence mapping & scenarios (shared by all algos)
# ============================================================

TEST_DRIVES: Dict[str, str] = {
    "test_0": "2013_05_28_drive_0008_sync",
    "test_1": "2013_05_28_drive_0008_sync",
    "test_2": "2013_05_28_drive_0004_sync",
    "test_3": "2013_05_28_drive_0002_sync",
}

FULL_TEST_SEQS: List[str] = ["test_0", "test_1", "test_2", "test_3"]
LIGHT_TEST_SEQS: List[str] = ["test_0"]

# Generic scenarios – can be used by ORB, MAST3R, AirSLAM
TEST_SCENARIOS: Dict[str, List[str]] = {
    "full": FULL_TEST_SEQS,
    "light": LIGHT_TEST_SEQS,
}

KITTI_FPS: float = 10.0  # nominal KITTI-360 frame rate

# ============================================================
# AirSLAM (inside Docker, using /ASP mounts)
# ============================================================

# In the AirSLAM Docker, host data is mounted at /ASP/data
ASP_DATA_ROOT = Path("/ASP/data")
ASP_OUTPUT_ROOT = Path("/ASP/output")

# AirSLAM KITTI-360 dataset layout inside the container
DATA_AIRSLAM_ROOT = ASP_DATA_ROOT / "airslam_kitti360"
DATA_POSES_DOCKER = ASP_DATA_ROOT / "data_poses"  # same content as DATA_POSES, different root

# AirSLAM configs inside the container
AIRSLAM_CONFIG_PATH = "/workspace/src/AirSLAM/configs/visual_odometry/vo_kitti360.yaml"
AIRSLAM_CAMERA_CONFIG_PATH = "/workspace/src/AirSLAM/configs/camera/kitti360.yaml"

# AirSLAM outputs (mirror ORB layout: algorithm / {mono, stereo})
OUT_AIRSLAM_ROOT = ASP_OUTPUT_ROOT / "AirSLAM"
# Backwards-compatible name used by older scripts — point it to the same root
OUT_AIRSLAM_STEREO = OUT_AIRSLAM_ROOT