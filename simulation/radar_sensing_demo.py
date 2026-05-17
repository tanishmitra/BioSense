import argparse
import json

from simulation.run_complete_pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run one ISAC frame and plot the radar sensing evidence."
    )
    parser.add_argument("--num-bits", type=int, default=1024)
    parser.add_argument("--no-fall", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--save",
        default="outputs/radar_fall_evidence.png",
        help="Path where the radar evidence graph should be saved.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the graph without opening a matplotlib window.",
    )
    args = parser.parse_args()

    result = run_pipeline(
        num_bits=args.num_bits,
        inject_fall=not args.no_fall,
        show_plots=not args.no_show,
        rng_seed=args.seed,
        save_radar_plot_path=args.save,
    )

    print(json.dumps({
        "fall": result["fall"],
        "radar": {
            "event_detected": result["radar"]["event_detected"],
            "event_frame_count": result["radar"]["event_frame_count"],
            "max_snr_db": round(result["radar"]["max_snr_db"], 3),
            "range_migration_m": round(result["radar"]["range_migration_m"], 3),
            "estimated_velocity_mps": round(result["radar"]["estimated_velocity_mps"], 6),
            "threshold_snr_db": result["radar"]["threshold_snr_db"],
            "track_stable": result["radar"]["track_stable"],
            "track_lost": result["radar"]["track_lost"],
            "track_range_m": round(result["radar"]["track_range_m"], 3),
            "track_baseline_snr_db": round(result["radar"]["track_baseline_snr_db"], 3),
            "track_loss_threshold_db": round(result["radar"]["track_loss_threshold_db"], 3),
            "track_loss_frame_count": result["radar"]["track_loss_frame_count"],
        },
        "plot_saved_to": args.save,
    }, indent=2))


if __name__ == "__main__":
    main()
