""" Schemas for resonant scanners."""
import datajoint as dj
from datajoint.jobs import key_hash
import matplotlib.pyplot as plt
import numpy as np
import scanreader
import gc

from . import experiment, notify, shared
from .utils import galvo_corrections, signal, quality, mask_classification
from .exceptions import PipelineException


schema = dj.schema('pipeline_reso', locals())
CURRENT_VERSION = 1


@schema
class Version(dj.Lookup):
    definition = """ # versions for the reso pipeline

    reso_version                    : smallint
    ---
    description = ''                : varchar(256)      # any notes on this version
    date = CURRENT_TIMESTAMP        : timestamp         # automatic
    """
    contents = [
        {'reso_version': 0, 'description': 'test'},
        {'reso_version': 1, 'description': 'first release'}
    ]


@schema
class ScanInfo(dj.Imported):
    definition = """ # master table with general data about the scans

    -> experiment.Scan
    -> Version                                  # reso version
    ---
    nslices                 : tinyint           # number of slices
    nchannels               : tinyint           # number of channels
    nframes                 : int               # number of recorded frames
    nframes_requested       : int               # number of requested frames (from header)
    px_height               : smallint          # lines per frame
    px_width                : smallint          # pixels per line
    um_height               : float             # height in microns
    um_width                : float             # width in microns
    x                       : float             # (um) center of scan in the motor coordinate system
    y                       : float             # (um) center of scan in the motor coordinate system
    fps                     : float             # (Hz) frames per second
    zoom                    : decimal(5,2)      # zoom factor
    bidirectional           : boolean           # true = bidirectional scanning
    usecs_per_line          : float             # microseconds per scan line
    fill_fraction           : float             # raster scan temporal fill fraction (see scanimage)
    """

    @property
    def key_source(self):
        rigs = [{'rig': '2P2'}, {'rig': '2P3'}, {'rig': '2P5'}]
        reso_sessions = (experiment.Session() & rigs)
        reso_scans = (experiment.Scan() - experiment.ScanIgnored()) & reso_sessions
        return reso_scans * (Version() & {'reso_version': CURRENT_VERSION})

    class Slice(dj.Part):
        definition = """ # slice-specific scan information

        -> ScanInfo
        -> shared.Slice
        ---
        z           : float             # (um) absolute depth with respect to the surface of the cortex
        """

    class QuantalSize(dj.Part):
        definition = """ # quantal size in images

        -> ScanInfo
        -> shared.Slice
        -> shared.Channel
        ---
        min_intensity               : int           # min value in movie
        max_intensity               : int           # max value in movie
        intensities                 : longblob      # intensities for fitting variances
        variances                   : longblob      # variances for each intensity
        quantal_size                : float         # variance slope, corresponds to quantal size
        zero_level                  : int           # level corresponding to zero (computed from variance dependence)
        quantal_frame               : longblob      # average frame expressed in quanta
        median_quantum_rate         : float         # median value in frame
        percentile95_quantum_rate   : float         # 95th percentile in frame
        """

        def _make_tuples(self, key, scan, slice_id, channel):
            # Create results tuple
            tuple_ = key.copy()
            tuple_['slice'] = slice_id + 1
            tuple_['channel'] = channel + 1

            # Compute quantal size
            middle_frame = int(np.floor(scan.num_frames / 2))
            frames = slice(max(middle_frame - 2000, 0), middle_frame + 2000)
            mini_scan = scan[slice_id, :, :, channel, frames]
            results = quality.compute_quantal_size(mini_scan)

            # Add results to tuple
            tuple_['min_intensity'] = results[0]
            tuple_['max_intensity'] = results[1]
            tuple_['intensities'] = results[2]
            tuple_['variances'] = results[3]
            tuple_['quantal_size'] = results[4]
            tuple_['zero_level'] = results[5]

            # Compute average frame rescaled with the quantal size
            mean_frame = np.mean(mini_scan, axis=-1)
            average_frame = (mean_frame - tuple_['zero_level']) / tuple_['quantal_size']
            tuple_['quantal_frame'] = average_frame
            tuple_['median_quantum_rate'] = np.median(average_frame)
            tuple_['percentile95_quantum_rate'] = np.percentile(average_frame, 95)

            # Insert
            self.insert1(tuple_)

    def _make_tuples(self, key):
        """ Read some scan parameters, compute FOV in microns and quantal size."""
        from decimal import Decimal

        # Read the scan
        print('Reading header...')
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)

        # Get attributes
        tuple_ = key.copy()  # in case key is reused somewhere else
        tuple_['nslices'] = scan.num_fields
        tuple_['nchannels'] = scan.num_channels
        tuple_['nframes'] = scan.num_frames
        tuple_['nframes_requested'] = scan.num_requested_frames
        tuple_['px_height'] = scan.image_height
        tuple_['px_width'] = scan.image_width
        tuple_['x'] = scan.motor_position_at_zero[0]
        tuple_['y'] = scan.motor_position_at_zero[1]
        tuple_['fps'] = scan.fps
        tuple_['zoom'] = Decimal(str(scan.zoom))
        tuple_['bidirectional'] = scan.is_bidirectional
        tuple_['usecs_per_line'] = scan.seconds_per_line * 1e6
        tuple_['fill_fraction'] = scan.temporal_fill_fraction

        # Estimate height and width in microns using measured FOVs for similar setups
        fov_rel = (experiment.FOV() * experiment.Session() * experiment.Scan() & key
                   & 'session_date>=fov_ts')
        zooms = fov_rel.fetch('mag').astype(np.float32)  # zooms measured in same setup
        closest_zoom = zooms[np.argmin(np.abs(np.log(zooms / scan.zoom)))]

        dims = (fov_rel & 'ABS(mag - {}) < 1e-4'.format(closest_zoom)).fetch1('height', 'width')
        um_height, um_width = [float(um) * (closest_zoom / scan.zoom) for um in dims]
        tuple_['um_height'] = um_height * scan._y_angle_scale_factor
        tuple_['um_width'] = um_width * scan._x_angle_scale_factor

        # Insert in ScanInfo
        self.insert1(tuple_)

        # Insert slice information
        z_zero = (experiment.Scan() & key).fetch1('depth')  # true depth at ScanImage's 0
        for slice_id, z_slice in enumerate(scan.field_depths):
            ScanInfo.Slice().insert1({**key, 'slice': slice_id + 1, 'z': z_zero + z_slice})

        # Compute quantal size for all slice/channel combinations
        for slice_id in range(scan.num_fields):
            print('Computing quantal size for slice', slice_id + 1)
            for channel in range(scan.num_channels):
                ScanInfo.QuantalSize()._make_tuples(key, scan, slice_id, channel)

        self.notify(key)

    def notify(self, key):
        msg = 'ScanInfo for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg)

    @property
    def microns_per_pixel(self):
        """ Returns an array with microns per pixel in height and width. """
        um_height, px_height, um_width, px_width = self.fetch1('um_height', 'px_height',
                                                               'um_width', 'px_width')
        return np.array([um_height / px_height, um_width / px_width])


@schema
class CorrectionChannel(dj.Manual):
    definition = """ # channel to use for raster and motion correction

    -> experiment.Scan
    -> shared.Slice
    ---
    -> shared.Channel
    """


@schema
class RasterCorrection(dj.Computed):
    definition = """ # raster correction for bidirectional resonant scans

    -> ScanInfo                         # animal_id, session, scan_idx, version
    -> CorrectionChannel                # animal_id, session, scan_idx, slice
    ---
    template            : longblob      # average frame from the middle of the movie
    raster_phase        : float         # difference between expected and recorded scan angle
    """

    @property
    def key_source(self):
        # Run make_tuples once per scan iff correction channel has been set for all slices
        scans = (ScanInfo() & CorrectionChannel()) - (ScanInfo.Slice() - CorrectionChannel())
        return scans & {'reso_version': CURRENT_VERSION}

    def _make_tuples(self, key):
        # Read the scan
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)

        for slice_id in range(scan.num_fields):
            print('Computing raster correction for slice', slice_id + 1)

            # Select channel
            correction_channel = (CorrectionChannel() & key & {'slice': slice_id + 1})
            channel = correction_channel.fetch1('channel') - 1

            # Create results tuple
            tuple_ = key.copy()
            tuple_['slice'] = slice_id + 1

            # Create the template (an average frame from the middle of the scan)
            middle_frame = int(np.floor(scan.num_frames / 2))
            frames = slice(max(middle_frame - 1000, 0), middle_frame + 1000)
            mini_scan = scan[slice_id, :, :, channel, frames]
            template = np.mean(mini_scan, axis=-1)
            tuple_['template'] = template

            # Compute raster correction parameters
            if scan.is_bidirectional:
                tuple_['raster_phase'] = galvo_corrections.compute_raster_phase(template,
                                                                                scan.temporal_fill_fraction)
            else:
                tuple_['raster_phase'] = 0

            # Insert
            self.insert1(tuple_)

        self.notify(key)

    def notify(self, key):
        msg = 'RasterCorrection for `{}` has been populated.'.format(key)
        msg += '\nRaster phases: {}'.format((self & key).fetch('raster_phase'))
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg)

    def get_correct_raster(self):
        """ Returns a function to perform raster correction on the scan. """
        raster_phase = self.fetch1('raster_phase')
        fill_fraction = (ScanInfo() & self).fetch1('fill_fraction')
        if raster_phase == 0:
            return lambda scan: scan.astype(np.float32, copy=False)
        else:
            return lambda scan: galvo_corrections.correct_raster(scan, raster_phase,
                                                                 fill_fraction)


@schema
class MotionCorrection(dj.Computed):
    definition = """ # motion correction for galvo scans

    -> RasterCorrection
    ---
    template                        : longblob      # image used as alignment template
    y_shifts                        : longblob      # (pixels) y motion correction shifts
    x_shifts                        : longblob      # (pixels) x motion correction shifts
    y_std                           : float         # (um) standard deviation of y shifts
    x_std                           : float         # (um) standard deviation of x shifts
    y_outlier_frames                : longblob      # mask with true for frames with high y shifts (already corrected)
    x_outlier_frames                : longblob      # mask with true for frames with high x shifts (already corrected)
    align_time=CURRENT_TIMESTAMP    : timestamp     # automatic
    """

    @property
    def key_source(self):
        # Run make_tuples once per scan iff RasterCorrection is done
        return ScanInfo() & RasterCorrection() & {'reso_version': CURRENT_VERSION}

    def _make_tuples(self, key):
        """Computes the motion shifts per frame needed to correct the scan."""
        from scipy import ndimage

        # Read the scan
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)

        # Get some params
        um_height, px_height, um_width, px_width = \
            (ScanInfo() & key).fetch1('um_height', 'px_height', 'um_width', 'px_width')

        for slice_id in range(scan.num_fields):
            print('Correcting motion in slice', slice_id + 1)

            # Select channel
            correction_channel = (CorrectionChannel() & key & {'slice': slice_id + 1})
            channel = correction_channel.fetch1('channel') - 1

            # Create results tuple
            tuple_ = key.copy()
            tuple_['slice'] = slice_id + 1

            # Load scan (we discard some rows/cols to avoid edge artifacts)
            skip_rows = int(round(px_height * 0.10))
            skip_cols = int(round(px_width * 0.10))
            scan_ = scan[slice_id, skip_rows: -skip_rows, skip_cols: -skip_cols, channel, :]  # height x width x frames

            # Correct raster effects (needed for subpixel changes in y)
            correct_raster = (RasterCorrection() & key & {'slice': slice_id + 1}).get_correct_raster()
            scan_ = correct_raster(scan_)
            scan_ -= scan_.min()  # make nonnegative for fft

            # Create template
            middle_frame = int(np.floor(scan.num_frames / 2))
            mini_scan = scan_[:, :, max(middle_frame - 1000, 0): middle_frame + 1000]
            mini_scan = 2 * np.sqrt(mini_scan + 3 / 8)  # *
            template = np.mean(mini_scan, axis=-1)
            template = ndimage.gaussian_filter(template, 0.7)  # **
            tuple_['template'] = template
            # * Anscombe tranform to normalize noise, increase contrast and decrease outliers' leverage
            # ** Small amount of gaussian smoothing to get rid of high frequency noise

            # Compute smoothing window size
            size_in_ms = 300  # smooth over a 300 milliseconds window
            window_size = int(round(scan.fps * (size_in_ms / 1000)))  # in frames
            window_size += 1 if window_size % 2 == 0 else 0  # make odd

            # Get motion correction shifts
            results = galvo_corrections.compute_motion_shifts(scan_, template,
                                                              smoothing_window_size=window_size)
            y_shifts = results[0] - results[0].mean()  # center motions around zero
            x_shifts = results[1] - results[1].mean()
            tuple_['y_shifts'] = y_shifts
            tuple_['x_shifts'] = x_shifts
            tuple_['y_outlier_frames'] = results[2]
            tuple_['x_outlier_frames'] = results[3]
            tuple_['y_std'] = np.std(y_shifts)
            tuple_['x_std'] = np.std(x_shifts)

            # Free memory
            del scan_
            gc.collect()

            # Insert
            self.insert1(tuple_)

        self.notify(key, scan)

    def notify(self, key, scan):
        import seaborn as sns

        fps = (ScanInfo() & key).fetch1('fps')
        seconds = np.arange(scan.num_frames) / fps

        with sns.axes_style('white'):
            fig, axes = plt.subplots(scan.num_fields, 1, figsize=(15, 4 * scan.num_fields),
                                     sharey=True)
        axes = [axes] if scan.num_fields == 1 else axes # make list if single axis object
        for i in range(scan.num_fields):
            y_shifts, x_shifts = (self & key & {'slice': i + 1}).fetch1('y_shifts', 'x_shifts')
            axes[i].set_title('Shifts for slice {}'.format(i + 1))
            axes[i].plot(seconds, y_shifts, label='y shifts')
            axes[i].plot(seconds, x_shifts, label='x shifts')
            axes[i].set_ylabel('Pixels')
            axes[i].set_xlabel('Seconds')
            axes[i].legend()
        fig.tight_layout()
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)
        sns.reset_orig()

        msg = 'MotionCorrection for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg, file=img_filename,
                                                                   file_title='motion shifts')

    def save_video(self, filename='galvo_corrections.mp4', channel=1, start_index=0,
                   seconds=30, dpi=250):
        """ Creates an animation video showing the original vs corrected scan.

        :param string filename: Output filename (path + filename)
        :param int channel: What channel from the scan to use. Starts at 1
        :param int start_index: Where in the scan to start the video.
        :param int seconds: How long in seconds should the animation run.
        :param int dpi: Dots per inch, controls the quality of the video.

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        # Get fps and total_num_frames
        fps = (ScanInfo() & self).fetch1('fps')
        num_video_frames = int(round(fps * seconds))
        stop_index = start_index + num_video_frames

        # Load the scan
        scan_filename = (experiment.Scan() & self).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)
        scan_ = scan[self.fetch1('slice') - 1, :, :, channel - 1, start_index: stop_index]
        original_scan = scan_.copy()

        # Correct the scan
        correct_raster = (RasterCorrection() & self.proj()).get_correct_raster()
        correct_motion = self.get_correct_motion()
        corrected_scan = correct_motion(correct_raster(scan_), slice(start_index, stop_index))

        # Create animation
        import matplotlib.animation as animation

        ## Set the figure
        fig, axes = plt.subplots(1, 2)

        axes[0].set_title('Original')
        im1 = axes[0].imshow(original_scan[:, :, 0], vmin=original_scan.min(),
                             vmax=original_scan.max())  # just a placeholder
        fig.colorbar(im1, ax=axes[0])
        axes[0].axis('off')

        axes[1].set_title('Corrected')
        im2 = axes[1].imshow(corrected_scan[:, :, 0], vmin=corrected_scan.min(),
                             vmax=corrected_scan.max())  # just a placeholder
        fig.colorbar(im2, ax=axes[1])
        axes[1].axis('off')

        ## Make the animation
        def update_img(i):
            im1.set_data(original_scan[:, :, i])
            im2.set_data(corrected_scan[:, :, i])

        video = animation.FuncAnimation(fig, update_img, corrected_scan.shape[2],
                                        interval=1000 / fps)

        # Save animation
        if not filename.endswith('.mp4'):
            filename += '.mp4'
        print('Saving video at:', filename)
        print('If this takes too long, stop it and call again with dpi <', dpi, '(default)')
        video.save(filename, dpi=dpi)

        return fig

    def get_correct_motion(self):
        """ Returns a function to perform motion correction on scans. """
        y_shifts, x_shifts = self.fetch1('y_shifts', 'x_shifts')
        xy_motion = np.stack([x_shifts, y_shifts])

        def my_lambda_function(scan, indices=None):
            if indices is None:
                return galvo_corrections.correct_motion(scan, xy_motion)
            else:
                return galvo_corrections.correct_motion(scan, xy_motion[:, indices])

        return my_lambda_function


@schema
class SummaryImages(dj.Computed):
    definition = """ # summary images for each slice and channel after corrections

    -> MotionCorrection
    -> shared.Channel
    ---
    average             : longblob          # l6-norm (across time) of each pixel
    correlation         : longblob          # (average) temporal correlation between each pixel and its eight neighbors
    """

    @property
    def key_source(self):
        # Run make_tuples once per scan iff MotionCorrection is done
        return ScanInfo() & MotionCorrection() & {'reso_version': CURRENT_VERSION}

    def _make_tuples(self, key):
        from .utils import correlation_image as ci

        # Read the scan
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)

        for slice_id in range(scan.num_fields):
            print('Computing summary images for slice', slice_id + 1)

            # Get raster and motion correction functions
            correct_raster = (RasterCorrection() & key & {'slice': slice_id + 1}).get_correct_raster()
            correct_motion = (MotionCorrection() & key & {'slice': slice_id + 1}).get_correct_motion()

            for channel in range(scan.num_channels):
                tuple_ = key.copy()
                tuple_['slice'] = slice_id + 1
                tuple_['channel'] = channel + 1

                # Correct scan
                scan_ = scan[slice_id, :, :, channel, :]
                scan_ = correct_motion(correct_raster(scan_))
                scan_ -= scan_.min()  # make nonnegative for lp-norm

                # Compute and insert correlation image
                tuple_['correlation'] = ci.compute_correlation_image(scan_)

                # Compute and insert lp-norm of each pixel over time
                p = 6
                scan_ = np.power(scan_, p, out=scan_)  # in place
                tuple_['average'] = np.sum(scan_, axis=-1, dtype=np.float64) ** (1 / p)

                # Free memory
                del scan_
                gc.collect()

                # Insert
                self.insert1(tuple_)

            self.notify({**key, 'slice': slice_id + 1}, scan.num_channels)  # once per slice

    def notify(self, key, num_channels):
        import seaborn as sns

        with sns.axes_style('white'):
            fig, axes = plt.subplots(num_channels, 2, squeeze=False, figsize=(12, 5 * num_channels))
        for channel in range(num_channels):
            axes[channel, 0].set_ylabel('Channel {}'.format(channel + 1), size='large',
                                        rotation='horizontal', ha='right')
            for i, img_name in enumerate(['average', 'correlation']):
                axes[0, i].set_title(img_name.title(), va='top')
                axes[channel, i].set_xticklabels([])
                axes[channel, i].set_yticklabels([])
                image = (self & key & {'channel': channel + 1}).fetch1(img_name)
                axes[channel, i].matshow(image)
        fig.suptitle('Slice {}'.format(key['slice']))
        fig.tight_layout()
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)
        sns.reset_orig()

        msg = 'SummaryImages for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg, file=img_filename,
                                                                   file_title='summary images')


@schema
class SegmentationTask(dj.Manual):
    definition = """ # defines the target of segmentation and the channel to use

    -> experiment.Scan
    -> shared.Slice
    -> shared.Channel
    -> shared.SegmentationMethod
    ---
    -> experiment.Compartment
    """

    def estimate_num_components(self):
        """ Estimates the number of components per slice using simple rules of thumb.

        For somatic scans, estimate number of neurons based on:
        (100x100x100)um^3 = 1e6 um^3 -> 1e2 neurons; (1x1x1)mm^3 = 1e9 um^3 -> 1e5 neurons

        For axonal/dendritic scans, just ten times our estimate of neurons.

        :returns: Number of components
        :rtype: int
        """

        # Get slice dimensions (in micrometers)
        scan = (ScanInfo() & self & {'reso_version': CURRENT_VERSION})
        slice_height, slice_width = scan.fetch1('um_height', 'um_width')
        slice_thickness = 10  # assumption
        slice_volume = slice_width * slice_height * slice_thickness

        # Estimate number of components
        compartment = self.fetch1('compartment')
        if compartment == 'soma':
            num_components = slice_volume * 0.0001
        elif compartment == 'axon':
            num_components = slice_volume * 0.001  # ten times as many neurons
        else:
            PipelineException("Compartment type '{}' not recognized".format(compartment))

        return int(round(num_components))

@schema
class DoNotSegment(dj.Manual):
    definition = """ # slice/channels that should not be segmented (used for web interface only)

    -> experiment.Scan
    -> shared.Slice
    -> shared.Channel
    """


@schema
class Segmentation(dj.Computed):
    definition = """ # Different mask segmentations.

    -> MotionCorrection         # animal_id, session, scan_idx, version, slice
    -> SegmentationTask         # animal_id, session, scan_idx, slice, channel, segmentation_method
    ---
    segmentation_time=CURRENT_TIMESTAMP     : timestamp     # automatic
    """

    @property
    def key_source(self):
        return MotionCorrection() * SegmentationTask() & {'reso_version': CURRENT_VERSION}

    class Mask(dj.Part):
        definition = """ # mask produced by segmentation.

        -> Segmentation
        mask_id         : smallint
        ---
        pixels          : longblob      # indices into the image in column major (Fortran) order
        weights         : longblob      # weights of the mask at the indices above
        """

        def get_mask_as_image(self):
            """ Return this mask as an image (2-d numpy array)."""
            # Get params
            pixels, weights = self.fetch('pixels', 'weights')
            image_height, image_width = (ScanInfo() & self).fetch1('px_height', 'px_width')

            # Reshape mask
            mask = Segmentation.reshape_masks(pixels, weights, image_height, image_width)

            return np.squeeze(mask)

    class Manual(dj.Part):
        definition = """ # masks created manually

        -> Segmentation
        """

        def _make_tuples(self, key):
            print('Warning: Manual segmentation is not implemented in Python.')
            # Copy any masks (and MaskClassification) that were there before
            # Delete key from Segmentation (this is needed for trace and ScanSet and Activity computation to restart when things are added)
            # Show GUI with the current masks
            # User modifies it somehow to produce the new set of masks
            # Insert info in Segmentation -> Segmentation.Manual -> Segmentation.Mask -> MaskClassification -> MaskClassification.Type

    class CNMF(dj.Part):
        definition = """ # source extraction using constrained non-negative matrix factorization

        -> Segmentation
        ---
        params              : varchar(1024)     # parameters send to CNMF as JSON array
        """

        def _make_tuples(self, key):
            """ Use CNMF to extract masks and traces.

            See caiman_interface.extract_masks for explanation of parameters
            """
            from .utils import caiman_interface as cmn
            import json

            print('')
            print('*' * 85)
            print('Processing {}'.format(key))

            # Load scan
            channel = key['channel'] - 1
            slice_id = key['slice'] - 1
            scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
            scan = scanreader.read_scan(scan_filename, dtype=np.float32)
            scan_ = scan[slice_id, :, :, channel, :]

            # Correct scan
            print('Correcting scan...')
            correct_raster = (RasterCorrection() & key).get_correct_raster()
            correct_motion = (MotionCorrection() & key).get_correct_motion()
            scan_ = correct_motion(correct_raster(scan_))
            scan_ -= scan_.min()  # make nonnegative for caiman

            # Set CNMF parameters
            ## Estimate number of components per slice and soma radius in pixels
            num_components = (SegmentationTask() & key).estimate_num_components()
            soma_radius_in_pixels = 7 / (ScanInfo() & key).microns_per_pixel  # assumption: radius is 7 microns

            ## Set general parameters
            kwargs = {}
            kwargs['num_components'] = num_components
            kwargs['merge_threshold'] = 0.8

            ## Set performance/execution parameters (heuristically), decrease if memory overflows
            kwargs['num_processes'] = 12  # Set to None for all cores available
            kwargs['num_pixels_per_process'] = 10000

            ## Set params specific to somatic or axonal/dendritic scans
            target = (SegmentationTask() & key).fetch1('compartment')
            if target == 'soma':
                kwargs['init_method'] = 'greedy_roi'
                kwargs['soma_radius'] = tuple(soma_radius_in_pixels)
                kwargs['num_background_components'] = 4
                kwargs['init_on_patches'] = False
            else:  # axons/dendrites
                kwargs['init_method'] = 'sparse_nmf'
                kwargs['snmf_alpha'] = 500  # 10^2 to 10^3.5 is a good range
                kwargs['num_background_components'] = 1
                kwargs['init_on_patches'] = True

            ## Set params specific to initialization on patches
            if kwargs['init_on_patches']:
                kwargs['patch_downsampling_factor'] = 4
                kwargs['proportion_patch_overlap'] = 0.2

            # Extract traces
            print('Extracting masks and traces (cnmf)...')
            cnmf_result = cmn.extract_masks(scan_, **kwargs)
            (masks, traces, background_masks, background_traces, raw_traces) = cnmf_result

            # Insert CNMF results
            print('Inserting masks, background components and traces...')

            ## Insert in CNMF, Segmentation and Fluorescence
            Segmentation().insert1(key)
            Segmentation.CNMF().insert1({**key, 'params': json.dumps(kwargs)})
            Fluorescence().insert1(key)  # we also insert traces

            ## Insert background components
            Segmentation.CNMFBackground().insert1({**key, 'masks': background_masks,
                                                   'activity': background_traces})

            ## Insert masks and traces (masks in Matlab format)
            num_masks = masks.shape[-1]
            masks = masks.reshape(-1, num_masks, order='F').T  # [num_masks x num_pixels] in F order
            for mask_id, mask, trace in zip(range(1, num_masks + 1), masks, raw_traces):
                mask_pixels = np.where(mask)[0]
                mask_weights = mask[mask_pixels]
                mask_pixels += 1  # matlab indices start at 1
                Segmentation.Mask().insert1({**key, 'mask_id': mask_id, 'pixels': mask_pixels,
                                             'weights': mask_weights})

                Fluorescence.Trace().insert1({**key, 'mask_id': mask_id, 'trace': trace})

            Segmentation().notify(key)

        def save_video(self, filename='cnmf_results.mp4', start_index=0, seconds=30,
                       dpi=250, first_n=None):
            """ Creates an animation video showing the results of CNMF.

            :param string filename: Output filename (path + filename)
            :param int start_index: Where in the scan to start the video.
            :param int seconds: How long in seconds should the animation run.
            :param int dpi: Dots per inch, controls the quality of the video.
            :param int first_n: Draw only the first n components.

            :returns Figure. You can call show() on it.
            :rtype: matplotlib.figure.Figure
            """
            # Get fps and calculate total number of frames
            fps = (ScanInfo() & self).fetch1('fps')
            num_video_frames = int(round(fps * seconds))
            stop_index = start_index + num_video_frames

            # Load the scan
            channel = self.fetch1('channel') - 1
            slice_id = self.fetch1('slice') - 1
            scan_filename = (experiment.Scan() & self).local_filenames_as_wildcard
            scan = scanreader.read_scan(scan_filename, dtype=np.float32)
            scan_ = scan[slice_id, :, :, channel, start_index: stop_index]

            # Correct the scan
            correct_raster = (RasterCorrection() & self).get_correct_raster()
            correct_motion = (MotionCorrection() & self).get_correct_motion()
            scan_ = correct_motion(correct_raster(scan_), slice(start_index, stop_index))

            # Get scan dimensions
            image_height, image_width, _ = scan_.shape
            num_pixels = image_height * image_width

            # Get masks and traces
            masks = (Segmentation() & self).get_all_masks()
            traces = (Fluorescence() & self).get_all_traces()  # always there for CNMF
            background_masks, background_traces = (Segmentation.CNMFBackground() &
                                                   self).fetch1('masks', 'activity')

            # Select first n components
            if first_n is not None:
                masks = masks[:, :, :first_n]
                traces = traces[:first_n, :]

            # Drop frames that won't be displayed
            traces = traces[:, start_index: stop_index]
            background_traces = background_traces[:, start_index: stop_index]

            # Create movies
            extracted = np.dot(masks.reshape(num_pixels, -1), traces)
            extracted = extracted.reshape(image_height, image_width, -1)
            background = np.dot(background_masks.reshape(num_pixels, -1), background_traces)
            background = background.reshape(image_height, image_width, -1)
            residual = scan_ - extracted - background

            # Create animation
            import matplotlib.animation as animation

            ## Set the figure
            fig, axes = plt.subplots(2, 2)

            axes[0, 0].set_title('Original (Y)')
            im1 = axes[0, 0].imshow(scan_[:, :, 0], vmin=scan_.min(), vmax=scan_.max())  # just a placeholder
            fig.colorbar(im1, ax=axes[0, 0])

            axes[0, 1].set_title('Extracted (A*C)')
            im2 = axes[0, 1].imshow(extracted[:, :, 0], vmin=extracted.min(), vmax=extracted.max())
            fig.colorbar(im2, ax=axes[0, 1])

            axes[1, 0].set_title('Background (B*F)')
            im3 = axes[1, 0].imshow(background[:, :, 0], vmin=background.min(),
                                    vmax=background.max())
            fig.colorbar(im3, ax=axes[1, 0])

            axes[1, 1].set_title('Residual (Y - A*C - B*F)')
            im4 = axes[1, 1].imshow(residual[:, :, 0], vmin=residual.min(), vmax=residual.max())
            fig.colorbar(im4, ax=axes[1, 1])

            for ax in axes.ravel():
                ax.axis('off')

            ## Make the animation
            def update_img(i):
                im1.set_data(scan_[:, :, i])
                im2.set_data(extracted[:, :, i])
                im3.set_data(background[:, :, i])
                im4.set_data(residual[:, :, i])

            video = animation.FuncAnimation(fig, update_img, scan_.shape[2],
                                            interval=1000 / fps)

            # Save animation
            if not filename.endswith('.mp4'):
                filename += '.mp4'
            print('Saving video at:', filename)
            print('If this takes too long, stop it and call again with dpi <', dpi, '(default)')
            video.save(filename, dpi=dpi)

            return fig

    class CNMFBackground(dj.Part):
        definition = """ # inferred background components

        -> Segmentation.CNMF
        ---
        masks               : longblob      # array (im_height x im_width x num_background_components)
        activity            : longblob      # array (num_background_components x timesteps)
        """

    def _make_tuples(self, key):
        # Create masks
        if key['segmentation_method'] == 1:  # manual
            Segmentation.Manual()._make_tuples(key)
        elif key['segmentation_method'] == 2:  # nmf
            Segmentation.CNMF()._make_tuples(key)
        else:
            msg = 'Unrecognized segmentation method {}'.format(key['segmentation_method'])
            raise PipelineException(msg)

    def notify(self, key):
        fig = (Segmentation() & key).plot_masks()
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)

        msg = 'Segmentation for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg, file=img_filename,
                                                                   file_title='mask contours')

    @staticmethod
    def reshape_masks(mask_pixels, mask_weights, image_height, image_width):
        """ Reshape masks into an image_height x image_width x num_masks array."""
        masks = np.zeros([image_height, image_width, len(mask_pixels)])

        # Reshape each mask
        for i, (mp, mw) in enumerate(zip(mask_pixels, mask_weights)):
            mask_as_vector = np.zeros(image_height * image_width)
            mask_as_vector[np.squeeze(mp - 1).astype(int)] = np.squeeze(mw)
            masks[:, :, i] = mask_as_vector.reshape(image_height, image_width, order='F')

        return masks

    def get_all_masks(self):
        """Returns an image_height x image_width x num_masks matrix with all masks."""
        mask_rel = (Segmentation.Mask() & self)

        # Get masks
        image_height, image_width = (ScanInfo() & self).fetch1('px_height', 'px_width')
        mask_pixels, mask_weights = mask_rel.fetch('pixels', 'weights', order_by='mask_id')

        # Reshape masks
        masks = Segmentation.reshape_masks(mask_pixels, mask_weights, image_height, image_width)

        return masks

    def plot_masks(self, first_n=None):
        """ Draw contours of masks over the correlation image (if available).

        :param first_n: Number of masks to plot. None for all.

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        from .utils import caiman_interface as cmn

        # Get masks
        masks = self.get_all_masks()
        if first_n is not None:
            masks = masks[:, :, :first_n]

        # Get correlation image if defined, black background otherwise.
        image_rel = SummaryImages() & self
        if image_rel:
            background_image = image_rel.fetch1('correlation')
        else:
            background_image = np.zeros(masks.shape[:-1])

        # Draw contours
        image_height, image_width = background_image.shape
        figsize = np.array([image_width, image_height]) / min(image_height, image_width)
        fig = plt.figure(figsize=figsize * 7)
        cmn.plot_masks(masks, background_image)

        return fig


@schema
class Fluorescence(dj.Computed):
    definition = """  # fluorescence traces before spike extraction or filtering

    -> Segmentation   # animal_id, session, scan_idx, reso_version, slice, channel, segmentation_method
    """

    @property
    def key_source(self):
        return Segmentation() & {'reso_version': CURRENT_VERSION}

    class Trace(dj.Part):
        definition = """

        -> Fluorescence
        -> Segmentation.Mask
        ---
        trace                   : longblob
        """

    def _make_tuples(self, key):
        # Load scan
        print('Loading scan...')
        slice_id = key['slice'] - 1
        channel = key['channel'] - 1
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)
        scan_ = scan[slice_id, :, :, channel, :]

        # Correct the scan
        print('Correcting scan...')
        correct_raster = (RasterCorrection() & key).get_correct_raster()
        correct_motion = (MotionCorrection() & key).get_correct_motion()
        scan_ = correct_motion(correct_raster(scan_))

        # Get masks
        print('Creating fluorescence traces...')
        mask_ids, pixels, weights = (Segmentation.Mask() & key).fetch('mask_id', 'pixels', 'weights')
        masks = Segmentation.reshape_masks(pixels, weights, scan.image_height, scan.image_width)
        masks = masks.transpose([2, 0, 1])

        self.insert1(key)
        for mask_id, mask in zip(mask_ids, masks):
            trace = np.average(scan_.reshape(-1, scan.num_frames), weights=mask.ravel(),
                               axis=0)

            Fluorescence.Trace().insert1({**key, 'mask_id': mask_id, 'trace': trace})

        self.notify(key)

    def notify(self, key):
        fig = plt.figure(figsize=(15, 4))
        plt.plot((Fluorescence() & key).get_all_traces().T)
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)

        msg = 'Fluorescence.Trace for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg, file=img_filename,
                                                                   file_title='calcium traces')

    def get_all_traces(self):
        """ Returns a num_traces x num_timesteps matrix with all traces."""
        traces = (Fluorescence.Trace() & self).fetch('trace', order_by='mask_id')
        return np.array([x.squeeze() for x in traces])


@schema
class MaskClassification(dj.Computed):
    definition = """ # classification of segmented masks.

    -> Segmentation                     # animal_id, session, scan_idx, reso_version, slice, channel, segmentation_method
    -> SummaryImages                    # animal_id, session, scan_idx, reso_version, slice, channel
    -> shared.ClassificationMethod
    ---
    classif_time=CURRENT_TIMESTAMP    : timestamp     # automatic
    """

    @property
    def key_source(self):
        return (Segmentation() * SummaryImages() * shared.ClassificationMethod() &
                {'reso_version': CURRENT_VERSION})

    class Type(dj.Part):
        definition = """

        -> MaskClassification
        -> Segmentation.Mask
        ---
        -> shared.MaskType
        """

    def _make_tuples(self, key):
        # Get masks
        image_height, image_width = (ScanInfo() & key).fetch1('px_height', 'px_width')
        mask_ids, pixels, weights = (Segmentation.Mask() & key).fetch('mask_id', 'pixels', 'weights')
        masks = Segmentation.reshape_masks(pixels, weights, image_height, image_width)
        masks = masks.transpose([2, 0, 1])  # num_masks, image_height, image_width

        # Classify masks
        if key['classification_method'] == 1:  # manual
            template = (SummaryImages() & key).fetch1('correlation')
            mask_types = mask_classification.classify_manual(masks, template)
        elif key['classification_method'] == 2:  # cnn
            raise PipelineException('Convnet not yet implemented.')
            # template = (SummaryImages() & key).fetch1('correlation')
            # mask_types = mask_classification.classify_cnn(masks, template)
        else:
            msg = 'Unrecognized classification method {}'.format(key['classification_method'])
            raise PipelineException(msg)

        print('Generated types:', mask_types)

        # Insert results
        self.insert1(key)
        for mask_id, mask_type in zip(mask_ids, mask_types):
            MaskClassification.Type().insert1({**key, 'mask_id': mask_id, 'type': mask_type})


@schema
class ScanSet(dj.Computed):
    definition = """ # set of all units in the same scan

    -> Fluorescence                 # processing done per slice
    """

    @property
    def key_source(self):
        return Fluorescence() & {'reso_version': CURRENT_VERSION}

    class Unit(dj.Part):
        definition = """ # single unit in the scan

        -> ScanInfo
        -> shared.SegmentationMethod
        unit_id                 : int           # unique per scan & segmentation method
        ---
        -> ScanSet                              # for it to act as a part table of ScanSet
        -> Fluorescence.Trace
        """

    # class Match(dj.Part) # MaskSet?
    #    definition = """ # unit-mask pairs per scan
    #    -> ScanSet.Unit
    #    -> Fluorescence.Trace
    #    """

    class UnitInfo(dj.Part):
        definition = """ # unit type and coordinates in x, y, z

        -> ScanSet.Unit
        ---
        -> shared.MaskType                  # type of the unit
        um_x                : smallint      # x-coordinate of centroid in motor coordinate system
        um_y                : smallint      # y-coordinate of centroid in motor coordinate system
        um_z                : smallint      # z-coordinate of mask relative to surface of the cortex
        px_x                : smallint      # x-coordinate of centroid in the frame
        px_y                : smallint      # y-coordinate of centroid in the frame
        """

    def job_key(self, key):
        # Force reservation key to be per scan so diff slices are not run in parallel
        return {k: v for k, v in key.items() if k not in ['slice', 'channel']}

    def _make_tuples(self, key):
        from pipeline.utils import caiman_interface as cmn

        # Get masks
        image_height, image_width = (ScanInfo() & key).fetch1('px_height', 'px_width')
        mask_ids, pixels, weights = (Segmentation.Mask() & key).fetch('mask_id', 'pixels', 'weights')
        masks = Segmentation.reshape_masks(pixels, weights, image_height, image_width)

        # Compute units' coordinates
        px_center = [image_height / 2, image_width / 2]
        um_center = (ScanInfo() & key).fetch1('y', 'x')
        um_z = (ScanInfo.Slice() & key).fetch1('z')
        px_centroids = cmn.get_centroids(masks)
        um_centroids = um_center + (px_centroids - px_center) * (ScanInfo() & key).microns_per_pixel

        # Get type from MaskClassification if available, else SegmentationTask
        if MaskClassification() & key:
            ids, types = (MaskClassification.Type() & key).fetch('mask_id', 'type')
            get_type = lambda mask_id: types[ids == mask_id].item()
        else:
            mask_type = (SegmentationTask() & key).fetch1('compartment')
            get_type = lambda mask_id: mask_type

        # Get next unit_id for scan
        unit_rel = (ScanSet.Unit().proj() & key)
        unit_id = np.max(unit_rel.fetch('unit_id')) + 1 if unit_rel else 1

        # Insert in ScanSet
        self.insert1(key)

        # Insert units
        unit_ids = range(unit_id, unit_id + len(mask_ids) + 1)
        for unit_id, mask_id, (um_y, um_x), (px_y, px_x) in zip(unit_ids, mask_ids,
                                                                um_centroids, px_centroids):
            ScanSet.Unit().insert1({**key, 'unit_id': unit_id, 'mask_id': mask_id})

            unit_info = {**key, 'unit_id': unit_id, 'type': get_type(mask_id), 'um_x': um_x,
                         'um_y': um_y, 'um_z': um_z, 'px_x': px_x, 'px_y': px_y}
            ScanSet.UnitInfo().insert1(unit_info, ignore_extra_fields=True)

        self.notify(key)

    def notify(self, key):
        fig = (ScanSet() & key).plot_centroids()
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)

        msg = 'ScanSet for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg, file=img_filename,
                                                                   file_title='unit centroids')

    def plot_centroids(self, first_n=None):
        """ Draw masks centroids over the correlation image. Works on a single slice/channel

        :param first_n: Number of masks to plot. None for all

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        # Get centroids
        centroids = self.get_all_centroids(centroid_type='px')
        if first_n is not None:
            centroids = centroids[:, :first_n]  # select first n components

        # Get correlation image if defined, black background otherwise.
        image_rel = SummaryImages() & self
        if image_rel:
            background_image = image_rel.fetch1('correlation')
        else:
            image_height, image_width = (ScanInfo() & self).fetch1('px_height', 'px_width')
            background_image = np.zeros([image_height, image_width])

        # Plot centroids
        image_height, image_width = background_image.shape
        figsize = np.array([image_width, image_height]) / min(image_height, image_width)
        fig = plt.figure(figsize=figsize * 7)
        plt.imshow(background_image)
        plt.plot(centroids[:, 0], centroids[:, 1], 'ow', markersize=3)

        return fig

    def plot_centroids3d(self):
        """ Plots the centroids of all units in the motor coordinate system (in microns)

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        from mpl_toolkits.mplot3d import Axes3D

        # Get centroids
        centroids = self.get_all_centroids()

        # Plot
        # TODO: Add different colors for different types, correlation image as 2-d planes
        # masks from diff channels with diff colors.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2])
        ax.invert_zaxis()
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.set_zlabel('z (um)')

        return fig

    def get_all_centroids(self, centroid_type='um'):
        """ Returns the centroids for all units in the scan. Could also be limited by slice.

        Centroid type is either 'um' or 'px':
            'um': Array (num_units x 3) with x, y, z in motor coordinate system (microns).
            'px': Array (num_units x 2) with x, y pixel coordinates.
        """
        units_rel = ScanSet.UnitInfo() & (ScanSet.Unit() & self)
        if centroid_type == 'um':
            xs, ys, zs = units_rel.fetch('um_x', 'um_y', 'um_z', order_by='unit_id')
            centroids = np.stack([xs, ys, zs], axis=1)
        else:
            xs, ys = units_rel.fetch('px_x', 'px_y', order_by='unit_id')
            centroids = np.stack([xs, ys], axis=1)
        return centroids


@schema
class Activity(dj.Computed):
    definition = """ # activity inferred from fluorescence traces

    -> ScanSet                                        # processing done per slice
    -> shared.SpikeMethod
    ---
    activity_time=CURRENT_TIMESTAMP   : timestamp     # automatic
    """

    @property
    def key_source(self):
        return ScanSet() * shared.SpikeMethod() & {'reso_version': CURRENT_VERSION}

    class Trace(dj.Part):
        definition = """ # deconvolved calcium acitivity

        -> ScanSet.Unit
        -> shared.SpikeMethod
        ---
        -> Activity                             # for it to act as part table of Activity
        trace               : longblob
        """

    class ARCoefficients(dj.Part):
        definition = """ # fitted parameters for the autoregressive process (nmf deconvolution)

        -> Activity.Trace
        ---
        g                   : blob          # g1, g2, ... coefficients for the AR process
        """

    def _make_tuples(self, key):
        print('Creating activity traces...')

        # Get fluorescence
        fps = (ScanInfo() & key).fetch1('fps')
        unit_ids, traces = (Fluorescence.Trace() * (ScanSet.Unit() & key)).fetch('unit_id', 'trace')
        full_traces = [signal.fill_nans(np.squeeze(trace).copy()) for trace in traces]

        # Insert in Activity
        self.insert1(key)
        if key['spike_method'] == 2:  # oopsie
            import pyfnnd  # Install from https://github.com/cajal/PyFNND.git

            for unit_id, trace in zip(unit_ids, full_traces):
                spike_trace = pyfnnd.deconvolve(trace, dt=1 / fps)[0]
                Activity.Trace().insert1({**key, 'unit_id': unit_id, 'trace': spike_trace})

        elif key['spike_method'] == 3:  # stm
            import c2s  # Install from https://github.com/lucastheis/c2s

            for unit_id, trace in zip(unit_ids, full_traces):
                start = signal.notnan(trace)
                end = signal.notnan(trace, len(trace) - 1, increment=-1)
                trace_dict = {'calcium': np.atleast_2d(trace[start:end + 1]), 'fps': fps}

                data = c2s.predict(c2s.preprocess([trace_dict], fps=fps), verbosity=0)
                spike_trace = np.squeeze(data[0].pop('predictions'))

                Activity.Trace().insert1({**key, 'unit_id': unit_id, 'trace': spike_trace})

        elif key['spike_method'] == 5:  # nmf
            from pipeline.utils import caiman_interface as cmn

            for unit_id, trace in zip(unit_ids, full_traces):
                spike_trace, ar_coeffs = cmn.deconvolve(trace)
                Activity.Trace().insert1({**key, 'unit_id': unit_id, 'trace': spike_trace})
                Activity.ARCoefficients().insert1({**key, 'unit_id': unit_id, 'g': ar_coeffs},
                                                  ignore_extra_fields=True)
        else:
            msg = 'Unrecognized spike method {}'.format(key['spike_method'])
            raise PipelineException(msg)

        self.notify(key)

    def notify(self, key):
        fig = plt.figure(figsize=(15, 4))
        plt.plot((Activity() & key).get_all_spikes().T)
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)

        msg = 'Activity.Trace for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg, file=img_filename,
                                                                   file_title='spike traces')

    def plot_impulse_responses(self, num_timepoints=100):
        """ Plots the impulse response functions for all traces.

        :param int num_timepoints: The number of points after impulse to use for plotting.

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        ar_rel = Activity.ARCoefficients() & (Activity.Trace() & self)
        if ar_rel:  # if an AR model was used
            # Get some params
            fps = (ScanInfo() & self).fetch1('fps')
            ar_coeffs = ar_rel.fetch('g')

            # Define the figure
            fig = plt.figure()
            x_axis = np.arange(num_timepoints) / fps  # make it seconds

            # Over each trace
            for g in ar_coeffs:
                AR_order = len(g)

                # Calculate impulse response function
                irf = np.zeros(num_timepoints)
                irf[0] = 1  # initial spike
                for i in range(1, num_timepoints):
                    if i <= AR_order:  # start of the array needs special care
                        irf[i] = np.sum(g[:i] * irf[i - 1:: -1])
                    else:
                        irf[i] = np.sum(g * irf[i - 1: i - AR_order - 1: -1])

                # Plot
                plt.plot(x_axis, irf)
                plt.xlabel('Seconds')

            return fig

    def get_all_spikes(self):
        """ Returns a num_traces x num_timesteps matrix with all spikes."""
        spikes = (Activity.Trace() & self).fetch('trace', order_by='unit_id')
        return np.array([x.squeeze() for x in spikes])


@schema
class ScanDone(dj.Computed):
    definition = """ # scans that are fully processed (updated every time a slice is added)

    -> ScanInfo
    -> shared.SegmentationMethod
    -> shared.SpikeMethod
    """

    @property
    def key_source(self):
        return Activity() & {'reso_version': CURRENT_VERSION}

    @property
    def target(self):
        return ScanDone.Partial() # trigger make_tuples for slices in Activity that aren't in ScanDone.Partial

    class Partial(dj.Part):
        definition = """ # slices that have been processed in the current scan

        -> ScanDone
        -> Activity
        """

    def _make_tuples(self, key):
        scan_key = {k: v for k, v in key.items() if k not in ['slice', 'channel']}

        # Delete current ScanDone entry
        with dj.config(safemode=False):
            (ScanDone() & scan_key).delete()

        # Reinsert in ScanDone
        self.insert1(scan_key)

        # Insert all processed slices in Partial
        ScanDone.Partial().insert((Activity() & scan_key).proj())

        self.notify(scan_key)

    def notify(self, key):
        msg = 'ScanDone for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg)


schema.spawn_missing_classes()