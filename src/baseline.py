#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List

from paths import OUTPUT_ROOT
from metrics import (
    SystemSample,
    monitor_system,
    summarize_system_metrics,
)
from file_creator import write_system_metrics_txt, log


def main(duration_seconds: int = 120, interval_seconds: float = 1.0) -> None:
    """
    Measure baseline OS / system usage with no SLAM running.

    - Samples CPU/RAM/GPU every `interval_seconds` for `duration_seconds`
    - Writes per-sample log to output/machine.txt
    - Prints simple summary (avg/max CPU/RAM/GPU)
    """
    log(
        f"[BASELINE] Measuring system baseline for {duration_seconds}s "
        f"(interval={interval_seconds}s)"
    )

    samples: List[SystemSample] = monitor_system(
        duration_seconds=duration_seconds,
        interval_seconds=interval_seconds,
    )

    summary = summarize_system_metrics(samples)
    log(f"[BASELINE] Summary: {summary}")

    out_dir: Path = OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "machine.txt"

    write_system_metrics_txt(out_path, samples)
    log(f"[BASELINE] Wrote baseline system metrics -> {out_path}")


if __name__ == "__main__":
    main()
