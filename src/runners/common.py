from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from utils import load_image_gray, tum_from_T
from metrics import append_system_sample  # collects SystemSample objects
from file_creator import load_ply_xyz, log


def run_slam_loop(
    slam_sys: Any,
    frames_info: List[Tuple[Path, Optional[Path], float]],
    cam_w: int,
    cam_h: int,
) -> Tuple[List[str], List[Any], int, Optional[int], Optional[int]]:
    """
    Generic runner for ORB-SLAM3 system instances.

    frames_info: list of (left_path, right_path_or_None, timestamp_rel)
      - if right_path is None, mono mode is assumed.
      - otherwise stereo mode is assumed.

    Returns:
      tum_rows: List[str]  (TUM-format lines for frames with st==OK)
      samples: List[SystemSample]
      num_pose_changes: int
      first_ok_frame: Optional[int]
      last_ok_frame: Optional[int]
    """
    tum_rows: List[str] = []
    samples: List[Any] = []
    last_T: Optional[np.ndarray] = None
    num_pose_changes = 0
    first_ok_frame: Optional[int] = None
    last_ok_frame: Optional[int] = None

    have_state = hasattr(slam_sys, "get_tracking_state")
    have_pose = hasattr(slam_sys, "get_current_pose")

    for i, (p_left, p_right, ts) in enumerate(frames_info):
        # Read images
        try:
            im_left = load_image_gray(p_left)
        except Exception as e:
            log(f"[ERR] Failed to load left image {p_left}: {e}")
            continue

        im_right = None
        if p_right is not None:
            try:
                im_right = load_image_gray(p_right)
            except Exception as e:
                log(f"[ERR] Failed to load right image {p_right}: {e}")
                im_right = None

        # Process image(s)
        try:
            if im_right is not None and hasattr(slam_sys, "process_image_stereo"):
                # stereo
                try:
                    slam_sys.process_image_stereo(im_left, im_right, float(ts))
                except TypeError:
                    # some bindings expect file paths; try both
                    try:
                        slam_sys.process_image_stereo(str(p_left), str(p_right), float(ts))
                    except Exception as e2:
                        log(f"[ERR] stereo process failed: {e2}")
                except Exception as e:
                    log(f"[ERR] stereo process failed: {e}")
            else:
                # mono
                try:
                    slam_sys.process_image_mono(im_left, float(ts))
                except TypeError:
                    try:
                        slam_sys.process_image_mono(str(p_left), float(ts))
                    except Exception as e2:
                        log(f"[ERR] mono process failed: {e2}")
                except Exception as e:
                    log(f"[ERR] mono process failed: {e}")
        except Exception:
            # ignore processing exceptions to keep runner robust
            pass

        # System metrics (use wall-time sample)
        append_system_sample(samples, samples, time.time())

        # Tracking state
        st = None
        if have_state:
            try:
                st = int(slam_sys.get_tracking_state())
            except Exception:
                st = None

        # Pose logging
        if have_pose:
            try:
                raw_pose = slam_sys.get_current_pose()
                if raw_pose is not None:
                    T = np.array(raw_pose, dtype=float)
                    if T.size == 16 and np.all(np.isfinite(T)):
                        T = T.reshape(4, 4)

                        if last_T is None:
                            last_T = T.copy()
                        elif not np.allclose(T, last_T):
                            num_pose_changes += 1
                            last_T = T.copy()

                        # Only log when tracking OK (2) if state available, else log whenever pose present
                        if (st is None) or (st == 2):
                            if first_ok_frame is None:
                                first_ok_frame = i
                            last_ok_frame = i

                            t_xyz, q_xyzw = tum_from_T(T)
                            tum_rows.append(
                                f"{float(ts):.9f} "
                                f"{t_xyz[0]:.6f} {t_xyz[1]:.6f} {t_xyz[2]:.6f} "
                                f"{q_xyzw[0]:.6f} {q_xyzw[1]:.6f} {q_xyzw[2]:.6f} {q_xyzw[3]:.6f}"
                            )
            except Exception as e:
                log(f"[ERR] get_current_pose failed at frame {i}: {e}")

        # Optional progress logging (light)
        if (i + 1) % 200 == 0 or (i + 1) == len(frames_info):
            log(f"[prog] processed {i+1}/{len(frames_info)} frames, TUM poses={len(tum_rows)}, pose_changes={num_pose_changes}")

    return tum_rows, samples, num_pose_changes, first_ok_frame, last_ok_frame


def save_map_points_ply(slam_sys: Any, ply_out: Path) -> Optional[int]:
    """
    Try to save map points to PLY via slam_sys API and return number of vertices
    if successful (uses file_creator.load_ply_xyz to count points).
    Returns None on failure.
    """
    ply_out.parent.mkdir(parents=True, exist_ok=True)
    # Try several possible call signatures
    tried = []
    try:
        # common signature: save_map_points_ply(path, bool)
        try:
            slam_sys.save_map_points_ply(str(ply_out), True)
            tried.append("path_bool")
        except TypeError:
            # try single-arg path
            slam_sys.save_map_points_ply(str(ply_out))
            tried.append("path_only")
    except Exception as e:
        log(f"[WARN] save_map_points_ply() call failed: {e}; tried={tried}")
        return None

    # Count points
    try:
        pts = load_ply_xyz(ply_out)
        num = int(pts.shape[0])
        return num
    except Exception as e:
        log(f"[WARN] Could not load/count points from {ply_out}: {e}")
        return None
