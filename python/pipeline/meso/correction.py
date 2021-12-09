from . import schema
from .metadata import ScanInfo, CorrectionChannel
import json
import numpy as np
import matplotlib.pyplot as plt
from ..utils import galvo_corrections, performance
from .. import notify, experiment
import scanreader
import datajoint as dj
import ray


@schema
class RasterCorrection(dj.Computed):
    definition = """ # raster correction for bidirectional resonant scans

    -> ScanInfo                         # animal_id, session, scan_idx, version
    -> CorrectionChannel                # animal_id, session, scan_idx, field
    ---
    raster_template     : longblob      # average frame from the middle of the movie
    raster_phase        : float         # difference between expected and recorded scan angle
    """

    @property
    def key_source(self):
        return ScanInfo * CorrectionChannel & {"pipe_version": CURRENT_VERSION}

    def make(self, key):
        from scipy.signal import tukey

        # Read the scan
        scan_filename = (experiment.Scan & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)

        # Select correction channel
        channel = (CorrectionChannel() & key).fetch1("channel") - 1
        field_id = key["field"] - 1

        # Load some frames from the middle of the scan
        middle_frame = int(np.floor(scan.num_frames / 2))
        frames = slice(max(middle_frame - 1000, 0), middle_frame + 1000)
        mini_scan = scan[field_id, :, :, channel, frames]

        # Create results tuple
        tuple_ = key.copy()

        # Create template (average frame tapered to avoid edge artifacts)
        taper = np.sqrt(
            np.outer(
                tukey(scan.field_heights[field_id], 0.4),
                tukey(scan.field_widths[field_id], 0.4),
            )
        )
        anscombed = 2 * np.sqrt(
            mini_scan - mini_scan.min() + 3 / 8
        )  # anscombe transform
        template = np.mean(anscombed, axis=-1) * taper
        tuple_["raster_template"] = template

        # Compute raster correction parameters
        if scan.is_bidirectional:
            tuple_["raster_phase"] = galvo_corrections.compute_raster_phase(
                template, scan.temporal_fill_fraction
            )
        else:
            tuple_["raster_phase"] = 0

        # Insert
        self.insert1(tuple_)

    def get_correct_raster(self):
        """ Returns a function to perform raster correction on the scan. """
        raster_phase = self.fetch1("raster_phase")
        fill_fraction = (ScanInfo() & self).fetch1("fill_fraction")
        if abs(raster_phase) < 1e-7:
            correct_raster = lambda scan: scan.astype(np.float32, copy=False)
        else:
            correct_raster = lambda scan: galvo_corrections.correct_raster(
                scan, raster_phase, fill_fraction
            )
        return correct_raster


@schema
class MotionCorrection(dj.Computed):
    definition = """ # motion correction for galvo scans

    -> RasterCorrection
    ---
    motion_template                 : longblob      # image used as alignment template
    y_shifts                        : longblob      # (pixels) y motion correction shifts
    x_shifts                        : longblob      # (pixels) x motion correction shifts
    y_std                           : float         # (pixels) standard deviation of y shifts
    x_std                           : float         # (pixels) standard deviation of x shifts
    outlier_frames                  : longblob      # mask with true for frames with outlier shifts (already corrected)
    align_time=CURRENT_TIMESTAMP    : timestamp     # automatic
    """

    @property
    def key_source(self):
        return RasterCorrection() & {"pipe_version": CURRENT_VERSION}

    def make(self, key):
        """Computes the motion shifts per frame needed to correct the scan."""
        from scipy import ndimage

        # Read the scan
        scan_filename = (experiment.Scan & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)

        # Get some params
        px_height, px_width = (ScanInfo.Field() & key).fetch1("px_height", "px_width")
        channel = (CorrectionChannel() & key).fetch1("channel") - 1
        field_id = key["field"] - 1

        # Load some frames from middle of scan to compute template
        skip_rows = int(
            round(px_height * 0.10)
        )  # we discard some rows/cols to avoid edge artifacts
        skip_cols = int(round(px_width * 0.10))
        middle_frame = int(np.floor(scan.num_frames / 2))
        mini_scan = scan[
            field_id,
            skip_rows:-skip_rows,
            skip_cols:-skip_cols,
            channel,
            max(middle_frame - 1000, 0) : middle_frame + 1000,
        ]
        mini_scan = mini_scan.astype(np.float32, copy=False)

        # Correct mini scan
        correct_raster = (RasterCorrection() & key).get_correct_raster()
        mini_scan = correct_raster(mini_scan)

        # Create template
        mini_scan = 2 * np.sqrt(mini_scan - mini_scan.min() + 3 / 8)  # *
        template = np.mean(mini_scan, axis=-1)
        template = ndimage.gaussian_filter(template, 0.7)  # **
        # * Anscombe tranform to normalize noise, increase contrast and decrease outliers' leverage
        # ** Small amount of gaussian smoothing to get rid of high frequency noise

        # Map: compute motion shifts in parallel
         
        
         # function to map
        raster_phase = (RasterCorrection() & key).fetch1("raster_phase")
        fill_fraction = (ScanInfo() & key).fetch1("fill_fraction")
        kwargs = {
            "raster_phase": raster_phase,
            "fill_fraction": fill_fraction,
            "template": template,
        }
        
        chunks = performance.ScanChunker(scan,field_id=field_id,channel=channel,kwargs=kwargs)

        shifts = [performance.parralel_motion_shifts.remote(i,raster_phase,fill_fraction) for i in chunks]
        results = ray.get(shifts)


        # Reduce
        y_shifts = np.zeros(scan.num_frames)
        x_shifts = np.zeros(scan.num_frames)
        for frames, chunk_y_shifts, chunk_x_shifts in results:
            y_shifts[frames] = chunk_y_shifts
            x_shifts[frames] = chunk_x_shifts

        # Detect outliers
        max_y_shift, max_x_shift = 20 / (ScanInfo.Field() & key).microns_per_pixel
        y_shifts, x_shifts, outliers = galvo_corrections.fix_outliers(
            y_shifts, x_shifts, max_y_shift, max_x_shift
        )

        # Center shifts around zero
        y_shifts -= np.median(y_shifts)
        x_shifts -= np.median(x_shifts)

        # Create results tuple
        tuple_ = key.copy()
        tuple_["motion_template"] = template
        tuple_["y_shifts"] = y_shifts
        tuple_["x_shifts"] = x_shifts
        tuple_["outlier_frames"] = outliers
        tuple_["y_std"] = np.std(y_shifts)
        tuple_["x_std"] = np.std(x_shifts)

        # Insert
        self.insert1(tuple_)

        # Notify after all fields have been processed
        scan_key = {
            "animal_id": key["animal_id"],
            "session": key["session"],
            "scan_idx": key["scan_idx"],
            "pipe_version": key["pipe_version"],
        }
        if len(MotionCorrection - CorrectionChannel & scan_key) > 0:
            self.notify(scan_key, scan.num_frames, scan.num_fields)

    @notify.ignore_exceptions
    def notify(self, key, num_frames, num_fields):
        fps = (ScanInfo() & key).fetch1("fps")
        seconds = np.arange(num_frames) / fps

        fig, axes = plt.subplots(
            num_fields, 1, figsize=(15, 4 * num_fields), sharey=True
        )
        axes = [axes] if num_fields == 1 else axes  # make list if single axis object
        for i in range(num_fields):
            y_shifts, x_shifts = (self & key & {"field": i + 1}).fetch1(
                "y_shifts", "x_shifts"
            )
            axes[i].set_title("Shifts for field {}".format(i + 1))
            axes[i].plot(seconds, y_shifts, label="y shifts")
            axes[i].plot(seconds, x_shifts, label="x shifts")
            axes[i].set_ylabel("Pixels")
            axes[i].set_xlabel("Seconds")
            axes[i].legend()
        fig.tight_layout()
        img_filename = "/tmp/" + key_hash(key) + ".png"
        fig.savefig(img_filename, bbox_inches="tight")
        plt.close(fig)

        msg = "motion shifts for {animal_id}-{session}-{scan_idx}".format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=img_filename, file_title=msg)

    def save_video(
        self,
        filename="galvo_corrections.mp4",
        channel=1,
        start_index=0,
        seconds=30,
        dpi=250,
    ):
        """Creates an animation video showing the original vs corrected scan.

        :param string filename: Output filename (path + filename)
        :param int channel: What channel from the scan to use. Starts at 1
        :param int start_index: Where in the scan to start the video.
        :param int seconds: How long in seconds should the animation run.
        :param int dpi: Dots per inch, controls the quality of the video.

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        # Get fps and total_num_frames
        fps = (ScanInfo() & self).fetch1("fps")
        num_video_frames = int(round(fps * seconds))
        stop_index = start_index + num_video_frames

        # Load the scan
        #scan = scanreader.read_scan()
        scan_filename = (experiment.Scan & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)
        scan_ = scan[
            self.fetch1("field") - 1, :, :, channel - 1, start_index:stop_index
        ]
        original_scan = scan_.copy()

        # Correct the scan
        correct_raster = (RasterCorrection() & self).get_correct_raster()
        correct_motion = self.get_correct_motion()
        corrected_scan = correct_motion(
            correct_raster(scan_), slice(start_index, stop_index)
        )

        # Create animation
        import matplotlib.animation as animation

        ## Set the figure
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

        axes[0].set_title("Original")
        im1 = axes[0].imshow(
            original_scan[:, :, 0], vmin=original_scan.min(), vmax=original_scan.max()
        )  # just a placeholder
        fig.colorbar(im1, ax=axes[0])
        axes[0].axis("off")

        axes[1].set_title("Corrected")
        im2 = axes[1].imshow(
            corrected_scan[:, :, 0],
            vmin=corrected_scan.min(),
            vmax=corrected_scan.max(),
        )  # just a placeholder
        fig.colorbar(im2, ax=axes[1])
        axes[1].axis("off")

        ## Make the animation
        def update_img(i):
            im1.set_data(original_scan[:, :, i])
            im2.set_data(corrected_scan[:, :, i])

        video = animation.FuncAnimation(
            fig, update_img, corrected_scan.shape[2], interval=1000 / fps
        )

        # Save animation
        if not filename.endswith(".mp4"):
            filename += ".mp4"
        print("Saving video at:", filename)
        print(
            "If this takes too long, stop it and call again with dpi <",
            dpi,
            "(default)",
        )
        video.save(filename, dpi=dpi)

        return fig

    def get_correct_motion(self):
        """ Returns a function to perform motion correction on scans. """
        x_shifts, y_shifts = self.fetch1("x_shifts", "y_shifts")

        return lambda scan, indices=slice(None): galvo_corrections.correct_motion(
            scan, x_shifts[indices], y_shifts[indices]
        )