from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config.settings import RADAR_SAMPLE_DELAY_PER_METER


class RadarPlotter:

    def plot(self, radar_result, fall_result=None, save_path=None, show=True):
        range_profiles = np.asarray(radar_result["range_profiles"])
        snr_db = np.asarray(radar_result["snr_db"])
        event_frames = np.asarray(radar_result["event_frames"], dtype=int)
        track_loss_frames = np.asarray(radar_result["track_loss_frames"], dtype=int)
        threshold = float(radar_result["threshold_snr_db"])

        fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
        fig.suptitle(self._title(fall_result), fontsize=14, fontweight="bold")

        self._plot_range_heatmap(axes[0], range_profiles, event_frames, track_loss_frames)
        self._plot_snr(axes[1], snr_db, threshold, event_frames)
        self._plot_track_loss(axes[2], radar_result, track_loss_frames)

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=160)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def _plot_range_heatmap(self, ax, range_profiles, event_frames, track_loss_frames):
        if range_profiles.size == 0:
            ax.text(0.5, 0.5, "No radar range profiles", ha="center", va="center")
            ax.set_axis_off()
            return

        power_db = 10 * np.log10(range_profiles.T + 1e-12)
        max_range_m = (range_profiles.shape[1] - 1) / max(RADAR_SAMPLE_DELAY_PER_METER, 1e-12)

        image = ax.imshow(
            power_db,
            aspect="auto",
            origin="lower",
            extent=[0, range_profiles.shape[0] - 1, 0, max_range_m],
            cmap="magma",
        )
        for frame_idx in event_frames:
            ax.axvline(frame_idx, color="cyan", alpha=0.45, linewidth=1)
        for frame_idx in track_loss_frames:
            ax.axvline(frame_idx, color="lime", alpha=0.65, linewidth=1.5)

        ax.set_title("Matched-filter radar range profile")
        ax.set_xlabel("Radar processing window")
        ax.set_ylabel("Range (m)")
        plt.colorbar(image, ax=ax, label="Return power (dB)")

    def _plot_snr(self, ax, snr_db, threshold, event_frames):
        frames = np.arange(len(snr_db))
        ax.plot(frames, snr_db, color="#1f77b4", linewidth=2, label="Peak SNR")
        ax.axhline(
            threshold,
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label="Detection threshold",
        )
        if len(event_frames):
            ax.scatter(
                event_frames,
                snr_db[event_frames],
                color="#17becf",
                edgecolor="black",
                zorder=3,
                label="SNR threshold frames",
            )

        ax.set_title("Radar SNR trigger evidence")
        ax.set_xlabel("Radar processing window")
        ax.set_ylabel("SNR (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    def _plot_track_loss(self, ax, radar_result, track_loss_frames):
        track_snr_db = np.asarray(radar_result["track_snr_db"])
        frames = np.arange(len(track_snr_db))
        threshold = float(radar_result["track_loss_threshold_db"])
        baseline = float(radar_result["track_baseline_snr_db"])

        ax.plot(
            frames,
            track_snr_db,
            color="#2ca02c",
            linewidth=2,
            label="Tracked patient SNR",
        )
        ax.axhline(
            baseline,
            color="#1f77b4",
            linestyle=":",
            linewidth=1.5,
            label="Stable-track baseline",
        )
        ax.axhline(
            threshold,
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label="Track-loss threshold",
        )
        if len(track_loss_frames):
            ax.scatter(
                track_loss_frames,
                track_snr_db[track_loss_frames],
                color="#ff7f0e",
                edgecolor="black",
                zorder=3,
                label="Track-loss frames",
            )

        migration = radar_result["range_migration_m"]
        track_range = radar_result["track_range_m"]
        ax.set_title(
            f"Patient track loss: range={track_range:.2f} m, migration={migration:.2f} m"
        )
        ax.set_xlabel("Radar processing window")
        ax.set_ylabel("Tracked SNR (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    def _title(self, fall_result):
        if not fall_result:
            return "Radar Sensing"

        status = "FALL ALERT TRIGGERED" if fall_result["fall_detected"] else "NO FALL ALERT"
        return (
            f"{status} | severity={fall_result['severity']} | "
            f"confidence={fall_result['confidence']:.2f}"
        )
