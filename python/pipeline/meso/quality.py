from . import schema 
import datajoint as dj
from pipeline.meso.metadata import ScanInfo, CURRENT_VERSION
import json 
from pipeline.utils import performance, signal, quality
import numpy as np 
from pipeline import experiment
import matplotlib.pyplot as plt
from pipeline import notify
from datajoint.jobs import key_hash
import ray


@schema
class Quality(dj.Computed):
    definition = """ # different quality metrics for a scan (before corrections)

    -> ScanInfo
    """

    @property
    def key_source(self):
        return ScanInfo() & {"pipe_version": CURRENT_VERSION}

    class MeanIntensity(dj.Part):
        definition = """ # mean intensity values across time

        -> Quality
        -> shared.Field
        -> shared.Channel
        ---
        intensities                 : longblob
        """

    class SummaryFrames(dj.Part):
        definition = """ # 16-part summary of the scan (mean of 16 blocks)

        -> Quality
        -> shared.Field
        -> shared.Channel
        ---
        summary                     : longblob      # h x w x 16
        """

    class Contrast(dj.Part):
        definition = """ # difference between 99 and 1 percentile across time

        -> Quality
        -> shared.Field
        -> shared.Channel
        ---
        contrasts                   : longblob
        """

    class QuantalSize(dj.Part):
        definition = """ # quantal size in images

        -> Quality
        -> shared.Field
        -> shared.Channel
        ---
        min_intensity               : int           # min value in movie
        max_intensity               : int           # max value in movie
        quantal_size                : float         # variance slope, corresponds to quantal size
        zero_level                  : int           # level corresponding to zero (computed from variance dependence)
        quantal_frame               : longblob      # average frame expressed in quanta
        """

    class EpileptiformEvents(dj.Part):
        definition = """ # compute frequency of epileptiform events

        -> Quality
        -> shared.Field
        -> shared.Channel
        ---
        frequency       : float         # (events / sec) frequency of epileptiform events
        abn_indices     : longblob      # indices of epileptiform events (0-based)
        peak_indices    : longblob      # indices of all local maxima peaks (0-based)
        prominences     : longblob      # peak prominence for all peaks
        widths          : longblob      # (secs) width at half prominence for all peaks
        """

    def make(self, key):
        # Read the scan
        ## TODO: read in scan
        # Insert in Quality
        self.insert1(key)

        for field_id in range(scan.num_fields):
            print("Computing quality metrics for field", field_id + 1)
            for channel in range(scan.num_channels):
                # Map: Compute quality metrics in parallel
                chunker = performance.ScanChunks(scan,field_id=field_id,channel=channel)
                chunks = chunker.chunks
                quality_metrics = [performance.parralel_quality_metrics.remote(i) for i in chunks]
                results = ray.get(quality_metrics)
            

                # Reduce
                mean_intensities = np.zeros(scan.num_frames)
                contrasts = np.zeros(scan.num_frames)
                for frames, chunk_mis, chunk_contrasts, _ in results:
                    mean_intensities[frames] = chunk_mis
                    contrasts[frames] = chunk_contrasts
                sorted_results = sorted(results, key=lambda res: res[0])
                mean_groups = np.array_split(
                    [r[3] for r in sorted_results], 16
                )  # 16 groups
                frames = np.stack(
                    [np.mean(g, axis=0) for g in mean_groups if g.any()], axis=-1
                )

                # Compute quantal size
                middle_frame = int(np.floor(scan.num_frames / 2))
                mini_scan = scan[
                    field_id,
                    :,
                    :,
                    channel,
                    max(middle_frame - 2000, 0) : middle_frame + 2000,
                ]
                mini_scan = mini_scan.astype(np.float32)
                results = quality.compute_quantal_size(mini_scan)
                min_intensity, max_intensity, _, _, quantal_size, zero_level = results
                quantal_frame = (
                    np.mean(mini_scan, axis=-1) - zero_level
                ) / quantal_size

                # Compute abnormal event frequency
                deviations = (
                    mean_intensities - mean_intensities.mean()
                ) / mean_intensities.mean()
                peaks, prominences, widths = quality.find_peaks(deviations)
                widths = np.array([w / scan.fps for w in widths])  # in seconds
                abnormal = peaks[
                    [p > 0.2 and w < 0.4 for p, w in zip(prominences, widths)]
                ]
                abnormal_freq = len(abnormal) / (scan.num_frames / scan.fps)

                # Insert
                field_key = {**key, "field": field_id + 1, "channel": channel + 1}
                self.MeanIntensity().insert1(
                    {**field_key, "intensities": mean_intensities}
                )
                self.Contrast().insert1({**field_key, "contrasts": contrasts})
                self.SummaryFrames().insert1({**field_key, "summary": frames})
                self.QuantalSize().insert1(
                    {
                        **field_key,
                        "min_intensity": min_intensity,
                        "max_intensity": max_intensity,
                        "quantal_size": quantal_size,
                        "zero_level": zero_level,
                        "quantal_frame": quantal_frame,
                    }
                )
                abnormal = np.array(abnormal)
                peaks = np.array(peaks)
                prominences = np.array(prominences)
            
                self.EpileptiformEvents.insert1(
                    {
                        **field_key,
                        "frequency": abnormal_freq,
                        "abn_indices": abnormal,
                        "peak_indices": peaks,
                        "prominences": prominences,
                        "widths": widths,
                    }
                )

                self.notify(field_key, frames, mean_intensities, contrasts)

    @notify.ignore_exceptions
    def notify(self, key, summary_frames, mean_intensities, contrasts):
        # Send summary frames
        import imageio

        video_filename = "/tmp/" + key_hash(key) + ".gif"
        percentile_99th = np.percentile(summary_frames, 99.5)
        summary_frames = np.clip(summary_frames, None, percentile_99th)
        summary_frames = signal.float2uint8(summary_frames).transpose([2, 0, 1])
        imageio.mimsave(video_filename, summary_frames, duration=0.4)

        msg = (
            "summary frames for {animal_id}-{session}-{scan_idx} field {field} "
            "channel {channel}"
        ).format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=video_filename, file_title=msg)

        # Send intensity and contrasts
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        axes[0].set_title("Mean intensity", size="small")
        axes[0].plot(mean_intensities)
        axes[0].set_ylabel("Pixel intensities")
        axes[1].set_title("Contrast (99 - 1 percentile)", size="small")
        axes[1].plot(contrasts)
        axes[1].set_xlabel("Frames")
        axes[1].set_ylabel("Pixel intensities")
        img_filename = "/tmp/" + key_hash(key) + ".png"
        fig.savefig(img_filename, bbox_inches="tight")
        plt.close(fig)

        msg = (
            "quality traces for {animal_id}-{session}-{scan_idx} field {field} "
            "channel {channel}"
        ).format(**key)
        slack_user.notify(file=img_filename, file_title=msg)