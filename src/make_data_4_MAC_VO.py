#!/usr/bin/env python3
from pathlib import Path
from paths import TEST_DRIVES, FULL_TEST_SEQS, LIGHT_TEST_SEQS

BASE = Path("/workspace/ASP/data/data_2d_test_slam")

def make_symlink(target: str, link_path: Path):
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target)

def main(scenario: str):
    if scenario == "light":
        seqs = LIGHT_TEST_SEQS
        print("Running LIGHT scenario: only test_0")
    elif scenario == "full":
        seqs = FULL_TEST_SEQS
        print("Running FULL scenario: test_0–test_3")
    else:
        print(f"[ERROR] Unknown scenario: {scenario}")
        return

    for seq in seqs:
        drive = TEST_DRIVES[seq]
        seq_path = BASE / seq / drive

        print(f"Processing: {seq_path}")

        if not seq_path.exists():
            print(f"[ERROR] Path NOT_FOUND: {seq_path}")
            continue

        make_symlink("image_00/data_rect", seq_path / "left")
        make_symlink("image_01/data_rect", seq_path / "right")

        print(f"[OK] Created symlinks in {seq_path}")

if __name__ == "__main__":
    scenario = "light"  # change to "full" if needed
    main(scenario)
