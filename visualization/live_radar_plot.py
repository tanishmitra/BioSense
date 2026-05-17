import matplotlib.pyplot as plt
import numpy as np


class LiveRadarPlotter:

    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
        self.fig.canvas.manager.set_window_title("Live ISAC Radar Sensing")

    def update(self, frame):
        radar = frame["radar"]
        fall = frame["fall"]
        frame_id = frame["frame_id"]

        for ax in self.axes:
            ax.clear()

        self._plot_snr(self.axes[0], radar)
        self._plot_track_loss(self.axes[1], radar)

        status = "FALL ALERT" if fall["fall_detected"] else "NO FALL"
        self.fig.suptitle(
            (
                f"Frame {frame_id} | {status} | severity={fall['severity']} | "
                f"confidence={fall['confidence']:.2f} | basis={fall['basis']}"
            ),
            fontsize=13,
            fontweight="bold",
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def show(self):
        plt.ion()
        plt.show(block=False)

    def pause(self, seconds):
        plt.pause(seconds)

    def close(self):
        plt.ioff()
        plt.close(self.fig)

    def _plot_snr(self, ax, radar):
        snr_db = np.asarray(radar["snr_db"])
        event_frames = np.asarray(radar["event_frames"], dtype=int)
        threshold = float(radar["threshold_snr_db"])
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

        ax.set_title(
            f"Radar SNR evidence: max={radar['max_snr_db']:.2f} dB"
        )
        ax.set_xlabel("Radar processing window inside current frame")
        ax.set_ylabel("SNR (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    def _plot_track_loss(self, ax, radar):
        track_snr_db = np.asarray(radar["track_snr_db"])
        loss_frames = np.asarray(radar["track_loss_frames"], dtype=int)
        frames = np.arange(len(track_snr_db))
        baseline = float(radar["track_baseline_snr_db"])
        threshold = float(radar["track_loss_threshold_db"])

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
        if len(loss_frames):
            ax.scatter(
                loss_frames,
                track_snr_db[loss_frames],
                color="#ff7f0e",
                edgecolor="black",
                zorder=3,
                label="Track-loss frames",
            )

        ax.set_title(
            (
                f"Patient track at {radar['track_range_m']:.2f} m | "
                f"stable={radar['track_stable']} | lost={radar['track_lost']} | "
                f"loss frames={radar['track_loss_frame_count']}"
            )
        )
        ax.set_xlabel("Radar processing window inside current frame")
        ax.set_ylabel("Tracked SNR (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
