import argparse
import json
import time

import numpy as np

from config.settings import NUM_BITS, STREAM_FALL_PROBABILITY, STREAM_INTERVAL_SECONDS
from simulation.run_complete_pipeline import run_pipeline
from visualization.live_radar_plot import LiveRadarPlotter


def main():
    parser = argparse.ArgumentParser(
        description="Run a live radar sensing graph synchronized with terminal output."
    )
    parser.add_argument("--num-bits", type=int, default=NUM_BITS)
    parser.add_argument("--frames", type=int, default=0, help="0 means run until Ctrl+C.")
    parser.add_argument("--interval", type=float, default=STREAM_INTERVAL_SECONDS)
    parser.add_argument("--fall-probability", type=float, default=STREAM_FALL_PROBABILITY)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--force-fall-every",
        type=int,
        default=0,
        help="Inject a fall every N frames. 0 uses fall probability only.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Print synchronized terminal output without opening the live graph.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    plotter = None if args.no_show else LiveRadarPlotter()
    if plotter is not None:
        plotter.show()

    frame_id = 0
    try:
        while args.frames == 0 or frame_id < args.frames:
            inject_fall = _should_inject_fall(
                frame_id,
                rng,
                args.fall_probability,
                args.force_fall_every,
            )
            result = run_pipeline(
                num_bits=args.num_bits,
                inject_fall=inject_fall,
                show_plots=False,
                rng_seed=args.seed + frame_id,
            )
            frame = _build_live_frame(frame_id, inject_fall, result)
            print(json.dumps(frame["terminal"]), flush=True)

            if plotter is not None:
                plotter.update(frame)
                plotter.pause(args.interval)
            else:
                time.sleep(args.interval)

            frame_id += 1
    except KeyboardInterrupt:
        print(json.dumps({"status": "stopped", "last_frame_id": frame_id - 1}), flush=True)
    finally:
        if plotter is not None:
            plotter.close()


def _should_inject_fall(frame_id, rng, fall_probability, force_fall_every):
    if force_fall_every > 0 and frame_id > 0 and frame_id % force_fall_every == 0:
        return True

    return bool(rng.random() < fall_probability)


def _build_live_frame(frame_id, inject_fall, result):
    radar = result["radar"]
    fall = result["fall"]
    track_snr = np.asarray(radar["track_snr_db"])

    terminal = {
        "frame_id": int(frame_id),
        "inject_fall": bool(inject_fall),
        "fall_detected": bool(fall["fall_detected"]),
        "severity": fall["severity"],
        "confidence": round(float(fall["confidence"]), 3),
        "basis": fall["basis"],
        "max_snr_db": round(float(radar["max_snr_db"]), 3),
        "track_range_m": round(float(radar["track_range_m"]), 3),
        "track_stable": bool(radar["track_stable"]),
        "track_lost": bool(radar["track_lost"]),
        "track_baseline_snr_db": round(float(radar["track_baseline_snr_db"]), 3),
        "track_loss_threshold_db": round(float(radar["track_loss_threshold_db"]), 3),
        "min_track_snr_db": round(float(track_snr.min()), 3) if len(track_snr) else 0.0,
        "track_loss_frame_count": int(radar["track_loss_frame_count"]),
    }

    return {
        "frame_id": int(frame_id),
        "inject_fall": bool(inject_fall),
        "fall": fall,
        "radar": radar,
        "terminal": terminal,
    }


if __name__ == "__main__":
    main()
