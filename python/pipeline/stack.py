""" Schemas for structural stacks. """
import datajoint as dj
from datajoint.jobs import key_hash
import matplotlib.pyplot as plt
import numpy as np
import scanreader
from scipy import signal
from scipy import ndimage
from scipy import optimize
import itertools

from . import experiment, notify, shared, reso, meso
anatomy = dj.create_virtual_module('pipeline_anatomy','pipeline_anatomy')

from .utils import galvo_corrections, stitching, performance, enhancement
from .utils.signal import mirrconv, float2uint8
from .exceptions import PipelineException

""" Note on our coordinate system:
Our stack/motor coordinate system is consistent with numpy's: z in the first axis pointing
downwards, y in the second axis pointing towards you and x on the third axis pointing to 
the right.
"""
dj.config['external-stack'] = {'protocol': 'file',
                               'location': '/mnt/dj-stor01/pipeline-externals'}
dj.config['cache'] = '/tmp/dj-cache'


schema = dj.schema('pipeline_stack', locals(), create_tables=False)


@schema
class StackInfo(dj.Imported):
    definition = """ # master table with general data about the stacks

    -> experiment.Stack
    ---
    nrois           : tinyint           # number of ROIs
    nchannels       : tinyint           # number of channels
    fill_fraction   : float             # raster scan temporal fill fraction (see scanimage)
    """

    class ROI(dj.Part):
        definition = """ # 3-D volumes that compose this stack (usually tiled to form a bigger fov)

        -> StackInfo
        roi_id          : tinyint           # same as ScanImage's
        ---
        -> experiment.Stack.Filename
        field_ids           : blob              # list of field_ids (0-index) sorted from shallower to deeper
        roi_z               : float             # (um) center of ROI in the motor coordinate system (cortex is at 0)
        roi_y               : float             # (um) center of ROI in the motor coordinate system
        roi_x               : float             # (um) center of ROI in the motor coordinate system
        roi_px_depth        : smallint          # number of slices
        roi_px_height       : smallint          # lines per frame
        roi_px_width        : smallint          # pixels per line
        roi_um_depth        : float             # depth in microns
        roi_um_height       : float             # height in microns
        roi_um_width        : float             # width in microns
        nframes             : smallint          # number of recorded frames per plane
        fps                 : float             # (Hz) volumes per second
        bidirectional       : boolean           # true = bidirectional scanning
        is_slow             : boolean           # whether all frames in one depth were recorded before moving to the next
        """

        def _make_tuples(self, key, stack, id_in_file):
            # Create results tuple
            tuple_ = key.copy()

            # Get field_ids ordered from shallower to deeper field in this ROI
            surf_z = (experiment.Stack() & key).fetch1('surf_depth')  # surface depth in fastZ coordinates (meso) or motor coordinates (reso)
            if stack.is_multiROI:
                field_ids = [i for i, field_roi in enumerate(stack.field_rois) if
                             id_in_file in field_roi]
                field_depths = [stack.field_depths[i] - surf_z for i in field_ids]
            else:
                field_ids = range(stack.num_scanning_depths)
                motor_zero = surf_z - stack.motor_position_at_zero[2]
                if stack.is_slow_stack and not stack.is_slow_stack_with_fastZ:  # using motor
                    initial_fastZ = stack.initial_secondary_z or 0
                    field_depths = [motor_zero - stack.field_depths[i] + 2 * initial_fastZ
                                    for i in field_ids]
                else:  # using fastZ
                    field_depths = [motor_zero + stack.field_depths[i] for i in field_ids]
            field_depths, field_ids = zip(*sorted(zip(field_depths, field_ids)))
            tuple_['field_ids'] = field_ids

            # Get reso/meso specific coordinates
            x_zero, y_zero, _ = stack.motor_position_at_zero  # motor x, y at ScanImage's 0
            if stack.is_multiROI:
                tuple_['roi_y'] = y_zero + stack._degrees_to_microns(stack.fields[
                                                                         field_ids[0]].y)
                tuple_['roi_x'] = x_zero + stack._degrees_to_microns(stack.fields[
                                                                         field_ids[0]].x)
                tuple_['roi_px_height'] = stack.field_heights[field_ids[0]]
                tuple_['roi_px_width'] = stack.field_widths[field_ids[0]]
                tuple_['roi_um_height'] = stack.field_heights_in_microns[field_ids[0]]
                tuple_['roi_um_width'] = stack.field_widths_in_microns[field_ids[0]]
            else:
                tuple_['roi_y'] = y_zero
                tuple_['roi_x'] = x_zero
                tuple_['roi_px_height'] = stack.image_height
                tuple_['roi_px_width'] = stack.image_width

                # Estimate height and width in microns using measured FOVs for similar setups
                fov_rel = (experiment.FOV() * experiment.Session() * experiment.Stack() &
                           key & 'session_date>=fov_ts')
                zooms = fov_rel.fetch('mag').astype(np.float32)  # zooms measured in same setup
                closest_zoom = zooms[np.argmin(np.abs(np.log(zooms / stack.zoom)))]
                dims = (fov_rel & 'ABS(mag - {}) < 1e-4'.format(closest_zoom)).fetch1(
                    'height', 'width')
                um_height, um_width = [float(um) * (closest_zoom / stack.zoom) for um in
                                       dims]
                tuple_['roi_um_height'] = um_height * stack._y_angle_scale_factor
                tuple_['roi_um_width'] = um_width * stack._x_angle_scale_factor

            # Get common parameters
            z_step = field_depths[1] - field_depths[0]
            tuple_['roi_z'] = field_depths[0] + (field_depths[-1] - field_depths[0]) / 2
            tuple_['roi_px_depth'] = len(field_ids)
            tuple_['roi_um_depth'] = field_depths[-1] - field_depths[0] + z_step
            tuple_['nframes'] = stack.num_frames
            tuple_['fps'] = stack.fps
            tuple_['bidirectional'] = stack.is_bidirectional
            tuple_['is_slow'] = stack.is_slow_stack

            self.insert1(tuple_)

        @property
        def microns_per_pixel(self):
            """ Returns an array with microns per pixel in depth, height and width. """
            um_dims = self.fetch1('roi_um_depth', 'roi_um_height', 'roi_um_width')
            px_dims = self.fetch1('roi_px_depth', 'roi_px_height', 'roi_px_width')
            return np.array([um_dim / px_dim for um_dim, px_dim in zip(um_dims, px_dims)])

    def _make_tuples(self, key):
        """ Read and store stack information."""
        print('Reading header...')

        # Read files forming this stack
        filename_keys = (experiment.Stack.Filename() & key).fetch(dj.key)
        stacks = []
        for filename_key in filename_keys:
            stack_filename = (experiment.Stack.Filename() &
                              filename_key).local_filenames_as_wildcard
            stacks.append(scanreader.read_scan(stack_filename))
        num_rois_per_file = [(s.num_rois if s.is_multiROI else 1) for s in stacks]

        # Create Stack tuple
        tuple_ = key.copy()
        tuple_['nrois'] = np.sum(num_rois_per_file)
        tuple_['nchannels'] = stacks[0].num_channels
        tuple_['fill_fraction'] = stacks[0].temporal_fill_fraction

        # Insert Stack
        self.insert1(tuple_)

        # Insert ROIs
        roi_id = 1
        for filename_key, num_rois, stack in zip(filename_keys, num_rois_per_file,
                                                 stacks):
            for roi_id_in_file in range(num_rois):
                roi_key = {**key, **filename_key, 'roi_id': roi_id}
                StackInfo.ROI()._make_tuples(roi_key, stack, roi_id_in_file)
                roi_id += 1

        # Fill in CorrectionChannel if only one channel
        if stacks[0].num_channels == 1:
            CorrectionChannel().fill(key)


@schema
class Quality(dj.Computed):
    definition = """ # different quality metrics for a scan (before corrections)

    -> StackInfo
    """

    class MeanIntensity(dj.Part):
        definition = """ # mean intensity per frame and slice

        -> Quality
        -> StackInfo.ROI
        -> shared.Channel
        ---
        intensities                 : longblob      # num_slices x num_frames
        """

    class SummaryFrames(dj.Part):
        definition = """ # mean slice at 8 different depths

        -> Quality
        -> StackInfo.ROI
        -> shared.Channel
        ---
        summary                     : longblob      # h x w x 8
        """

    class Contrast(dj.Part):
        definition = """ # difference between 99 and 1 percentile per frame and slice

        -> Quality
        -> StackInfo.ROI
        -> shared.Channel
        ---
        contrasts                   : longblob      # num_slices x num_frames
        """

    def _make_tuples(self, key):
        print('Computing quality metrics for stack', key)

        # Insert in Quality
        self.insert1(key)

        for roi_tuple in (StackInfo.ROI() & key).fetch():
            # Load ROI
            roi_filename = (experiment.Stack.Filename() &
                            roi_tuple).local_filenames_as_wildcard
            roi = scanreader.read_scan(roi_filename)

            for channel in range((StackInfo() & key).fetch1('nchannels')):
                # Map: Compute quality metrics in each field
                f = performance.parallel_quality_stack  # function to map
                field_ids = roi_tuple['field_ids']
                results = performance.map_fields(f, roi, field_ids=field_ids,
                                                 channel=channel)

                # Reduce: Collect results
                mean_intensities = np.empty((roi_tuple['roi_px_depth'],
                                             roi_tuple['nframes']))
                contrasts = np.empty((roi_tuple['roi_px_depth'], roi_tuple['nframes']))
                for field_idx, field_mis, field_contrasts, _ in results:
                    mean_intensities[field_idx] = field_mis
                    contrasts[field_idx] = field_contrasts
                frames = [res[3] for res in sorted(results, key=lambda res: res[0])]
                frames = np.stack(frames[:: int(len(frames) / 8)], axis=-1)  # frames at 8 diff depths

                # Insert
                roi_key = {**key, 'roi_id': roi_tuple['roi_id'], 'channel': channel + 1}
                self.MeanIntensity().insert1({**roi_key, 'intensities': mean_intensities})
                self.Contrast().insert1({**roi_key, 'contrasts': contrasts})
                self.SummaryFrames().insert1({**roi_key, 'summary': frames})

                self.notify(roi_key, frames, mean_intensities, contrasts)

    @notify.ignore_exceptions
    def notify(self, key, summary_frames, mean_intensities, contrasts):
        # Send summary frames
        import imageio
        video_filename = '/tmp/' + key_hash(key) + '.gif'
        percentile_99th = np.percentile(summary_frames, 99.5)
        summary_frames = np.clip(summary_frames, None, percentile_99th)
        summary_frames = float2uint8(summary_frames).transpose([2, 0, 1])
        imageio.mimsave(video_filename, summary_frames, duration=0.4)

        msg = ('summary frames for {animal_id}-{session}-{stack_idx} channel '
               '{channel}').format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=video_filename, file_title=msg)

        # Send intensity and contrasts
        figsize = (min(4, contrasts.shape[1] / 10 + 1), contrasts.shape[0] / 30 + 1)  # set heuristically
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
        fig.tight_layout()
        axes[0].set_title('Mean intensity', size='small')
        axes[0].imshow(mean_intensities)
        axes[0].set_ylabel('Slices')
        axes[0].set_xlabel('Frames')
        axes[1].set_title('Contrast (99 - 1 percentile)', size='small')
        axes[1].imshow(contrasts)
        axes[1].set_xlabel('Frames')
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename, bbox_inches='tight')
        plt.close(fig)

        msg = ('quality images for {animal_id}-{session}-{stack_idx} channel '
               '{channel}').format(**key)
        slack_user.notify(file=img_filename, file_title=msg)


@schema
class CorrectionChannel(dj.Manual):
    definition = """ # channel to use for raster and motion correction

    -> experiment.Stack
    ---
    -> shared.Channel
    """

    def fill(self, key, channel=1):
        for stack_key in (StackInfo() & key).fetch(dj.key):
            self.insert1({**stack_key, 'channel': channel}, ignore_extra_fields=True,
                         skip_duplicates=True)


@schema
class RasterCorrection(dj.Computed):
    definition = """ # raster correction for bidirectional resonant scans

    -> StackInfo.ROI                         # animal_id, session, stack_idx, roi_id, version
    -> CorrectionChannel                     # animal_id, session, stack_idx
    ---
    raster_phase            : float          # difference between expected and recorded scan angle
    raster_std              : float          # standard deviation among raster phases in different slices
    """

    def _make_tuples(self, key):
        """ Compute raster phase discarding top and bottom 15% of slices and tapering
        edges to avoid edge artifacts."""
        print('Computing raster correction for ROI', key)

        # Get some params
        res = (StackInfo.ROI() & key).fetch1('bidirectional', 'roi_px_height',
                                             'roi_px_width', 'field_ids')
        is_bidirectional, image_height, image_width, field_ids = res
        correction_channel = (CorrectionChannel() & key).fetch1('channel') - 1

        if is_bidirectional:
            # Read the ROI
            filename_rel = (experiment.Stack.Filename() & (StackInfo.ROI() & key))
            roi_filename = filename_rel.local_filenames_as_wildcard
            roi = scanreader.read_scan(roi_filename)

            # Compute some parameters
            skip_fields = max(1, int(round(len(field_ids) * 0.10)))
            taper = np.sqrt(np.outer(signal.tukey(image_height, 0.4),
                                     signal.tukey(image_width, 0.4)))

            # Compute raster phase for each slice and take the median
            raster_phases = []
            for field_id in field_ids[skip_fields: -2 * skip_fields]:
                # Create template (average frame tapered to avoid edge artifacts)
                slice_ = roi[field_id, :, :, correction_channel, :].astype(np.float32,
                                                                           copy=False)
                anscombed = 2 * np.sqrt(slice_ - slice_.min(axis=(0, 1)) + 3 / 8)  # anscombe transform
                template = np.mean(anscombed, axis=-1) * taper

                # Compute raster correction
                raster_phases.append(galvo_corrections.compute_raster_phase(template,
                                                                            roi.temporal_fill_fraction))
            raster_phase = np.median(raster_phases)
            raster_std = np.std(raster_phases)
        else:
            raster_phase = 0
            raster_std = 0

        # Insert
        self.insert1({**key, 'raster_phase': raster_phase, 'raster_std': raster_std})

        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        msg = ('raster phase for {animal_id}-{session}-{stack_idx} roi {roi_id}: '
               '{phase}').format(**key, phase=(self & key).fetch1('raster_phase'))
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg)

    def correct(self, roi):
        """ Correct roi with parameters extracted from self. In place.

        :param np.array roi: ROI (fields, image_height, image_width, frames).
        """
        raster_phase = self.fetch1('raster_phase')
        fill_fraction = (StackInfo() & self).fetch1('fill_fraction')
        if abs(raster_phase) < 1e-7:
            corrected = roi.astype(np.float32, copy=False)
        else:
            corrected = roi  # in_place
            for i, field in enumerate(roi):
                corrected[i] = galvo_corrections.correct_raster(field, raster_phase,
                                                                fill_fraction)
        return corrected


@schema
class MotionCorrection(dj.Computed):
    definition = """ # motion correction for each slice in the stack

    -> RasterCorrection
    ---
    y_shifts            : longblob      # y motion correction shifts (num_slices x num_frames)
    x_shifts            : longblob      # x motion correction shifts (num_slices x num_frames)
    """

    def _make_tuples(self, key):
        """ Compute motion shifts to align frames over time and over slices."""
        print('Computing motion correction for ROI', key)

        # Get some params
        res = (StackInfo.ROI() & key).fetch1('nframes', 'roi_px_height', 'roi_px_width',
                                             'field_ids')
        num_frames, image_height, image_width, field_ids = res
        correction_channel = (CorrectionChannel() & key).fetch1('channel') - 1

        y_shifts = np.zeros([len(field_ids), num_frames])
        x_shifts = np.zeros([len(field_ids), num_frames])
        if num_frames > 1:
            # Read the ROI
            filename_rel = (experiment.Stack.Filename() & (StackInfo.ROI() & key))
            roi_filename = filename_rel.local_filenames_as_wildcard
            roi = scanreader.read_scan(roi_filename)

            # Compute some params
            skip_rows = int(round(image_height * 0.10))
            skip_cols = int(round(image_width * 0.10))

            # Map: Compute shifts in parallel
            f = performance.parallel_motion_stack  # function to map
            raster_phase = (RasterCorrection() & key).fetch1('raster_phase')
            fill_fraction = (StackInfo() & key).fetch1('fill_fraction')
            max_y_shift, max_x_shift = 20 / (StackInfo.ROI() & key).microns_per_pixel[1:]
            results = performance.map_fields(f, roi, field_ids=field_ids,
                                             channel=correction_channel,
                                             kwargs={'raster_phase': raster_phase,
                                                     'fill_fraction': fill_fraction,
                                                     'skip_rows': skip_rows,
                                                     'skip_cols': skip_cols,
                                                     'max_y_shift': max_y_shift,
                                                     'max_x_shift': max_x_shift})

            # Reduce: Collect results
            for field_idx, y_shift, x_shift in results:
                y_shifts[field_idx] = y_shift
                x_shifts[field_idx] = x_shift

        # Insert
        self.insert1({**key, 'y_shifts': y_shifts, 'x_shifts': x_shifts})

        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        y_shifts, x_shifts = (MotionCorrection() & key).fetch1('y_shifts', 'x_shifts')
        fps, is_slow_stack = (StackInfo.ROI() & key).fetch1('fps', 'is_slow')
        num_slices, num_frames = y_shifts.shape
        fps = fps * (num_slices if is_slow_stack else 1)
        seconds = np.arange(num_frames) / fps

        fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True, sharey=True)
        axes[0].set_title('Shifts in y for all slices')
        axes[0].set_ylabel('Pixels')
        axes[0].plot(seconds, y_shifts.T)
        axes[1].set_title('Shifts in x for all slices')
        axes[1].set_ylabel('Pixels')
        axes[1].set_xlabel('Seconds')
        axes[1].plot(seconds, x_shifts.T)
        fig.tight_layout()
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)

        msg = 'motion shifts for {animal_id}-{session}-{stack_idx} roi {roi_id}'.format(
            **key)
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=img_filename, file_title=msg)

    def save_as_tiff(self, filename='roi.tif', channel=1):
        """ Correct roi and save as a tiff file.

        :param int channel: What channel to use. Starts at 1
        """
        from tifffile import imsave

        # Get some params
        res = (StackInfo.ROI() & self).fetch1('field_ids', 'roi_px_depth',
                                              'roi_px_height', 'roi_px_width')
        field_ids, px_depth, px_height, px_width = res

        # Load ROI
        roi_filename = (experiment.Stack.Filename() & self).local_filenames_as_wildcard
        roi = scanreader.read_scan(roi_filename)

        # Map: Apply corrections to each field in parallel
        f = performance.parallel_correct_stack  # function to map
        raster_phase = (RasterCorrection() & self).fetch1('raster_phase')
        fill_fraction = (StackInfo() & self).fetch1('fill_fraction')
        y_shifts, x_shifts = self.fetch1('y_shifts', 'x_shifts')
        results = performance.map_fields(f, roi, field_ids=field_ids, channel=channel,
                                         kwargs={'raster_phase': raster_phase,
                                                 'fill_fraction': fill_fraction,
                                                 'y_shifts': y_shifts,
                                                 'x_shifts': x_shifts})

        # Reduce: Collect results
        corrected_roi = np.empty((px_depth, px_height, px_width), dtype=np.float32)
        for field_idx, corrected_field in results:
            corrected_roi[field_idx] = corrected_field

        print('Saving file at:', filename)
        imsave(filename, corrected_roi)


@schema
class Stitching(dj.Computed):
    definition = """ # stitches together overlapping rois

    -> StackInfo
    """

    @property
    def key_source(self):
        return StackInfo() - (StackInfo.ROI() - MotionCorrection())  # run iff all ROIs have been processed

    class Volume(dj.Part):
        definition = """ # union of ROIs from a stack (usually one volume per stack)

        -> Stitching
        volume_id       : tinyint       # id of this volume
        """

    class ROICoordinates(dj.Part):
        definition = """ # coordinates for each ROI in the stitched volume

        -> Stitching                    # animal_id, session, stack_idx, version
        -> MotionCorrection             # animal_id, session, stack_idx, version, roi_id
        ---
        -> Stitching.Volume             # volume to which this ROI belongs
        stitch_ys        : blob         # (px) center of each slice in a volume-wise coordinate system
        stitch_xs        : blob         # (px) center of each slice in a volume-wise coordinate system
        """

    def _make_tuples(self, key):
        """ Stitch overlapping ROIs together and correct slice-to-slice alignment.

        Iteratively stitches two overlapping ROIs if the overlapping dimension has the
        same length (up to some relative tolerance). Stitching params are calculated per
        slice.

        Edge case: when two overlapping ROIs have different px/micron resolution
            They won't be joined even if true height are the same (as pixel heights will
            not match) or pixel heights could happen to match even if true heights are
            different and thus they'll be erroneously stitched.
        """
        print('Stitching ROIs for stack', key)

        # Get some params
        correction_channel = (CorrectionChannel() & key).fetch1('channel') - 1

        # Read and correct ROIs forming this stack
        print('Correcting ROIs...')
        rois = []
        for roi_tuple in (StackInfo.ROI() & key).fetch():
            # Load ROI
            roi_filename = (experiment.Stack.Filename() &
                            roi_tuple).local_filenames_as_wildcard
            roi = scanreader.read_scan(roi_filename)

            # Map: Apply corrections to each field in parallel
            f = performance.parallel_correct_stack  # function to map
            raster_phase = (RasterCorrection() & roi_tuple).fetch1('raster_phase')
            fill_fraction = (StackInfo() & roi_tuple).fetch1('fill_fraction')
            y_shifts, x_shifts = (MotionCorrection() & roi_tuple).fetch1('y_shifts',
                                                                         'x_shifts')
            field_ids = roi_tuple['field_ids']
            results = performance.map_fields(f, roi, field_ids=field_ids,
                                             channel=correction_channel,
                                             kwargs={'raster_phase': raster_phase,
                                                     'fill_fraction': fill_fraction,
                                                     'y_shifts': y_shifts,
                                                     'x_shifts': x_shifts,
                                                     'apply_anscombe': True})

            # Reduce: Collect results
            corrected_roi = np.empty((roi_tuple['roi_px_depth'],
                                      roi_tuple['roi_px_height'],
                                      roi_tuple['roi_px_width']), dtype=np.float32)
            for field_idx, corrected_field in results:
                corrected_roi[field_idx] = corrected_field

            # Create ROI object
            um_per_px = (StackInfo.ROI() & (StackInfo.ROI().proj() &
                                            roi_tuple)).microns_per_pixel
            px_z, px_y, px_x = np.array([roi_tuple['roi_{}'.format(dim)] for dim in
                                         ['z', 'y', 'x']]) / um_per_px
            rois.append(stitching.StitchedROI(corrected_roi, x=px_x, y=px_y, z=px_z,
                                              id_=roi_tuple['roi_id']))

        def enhance(image, sigmas):
            """ Enhance 2p image. See enhancement.py for details."""
            return enhancement.sharpen_2pimage(enhancement.lcn(image, sigmas))

        def join_rows(rois_):
            """ Iteratively join all rois that overlap in the same row."""
            sorted_rois = sorted(rois_, key=lambda roi: (roi.x, roi.y))

            prev_num_rois = float('inf')
            while len(sorted_rois) < prev_num_rois:
                prev_num_rois = len(sorted_rois)

                for left, right in itertools.combinations(sorted_rois, 2):
                    if left.is_aside_to(right):
                        roi_key = {**key, 'roi_id': left.roi_coordinates[0].id}
                        um_per_px = (StackInfo.ROI() & roi_key).microns_per_pixel

                        # Compute stitching shifts
                        neighborhood_size = 25 / um_per_px[1:]
                        left_ys, left_xs = [], []
                        for l, r in zip(left.slices, right.slices):
                            left_slice = enhance(l.slice, neighborhood_size)
                            right_slice = enhance(r.slice, neighborhood_size)
                            delta_y, delta_x = stitching.linear_stitch(left_slice,
                                                                       right_slice,
                                                                       r.x - l.x)
                            left_ys.append(r.y - delta_y)
                            left_xs.append(r.x - delta_x)

                        # Fix outliers
                        max_y_shift, max_x_shift = 10 / um_per_px[1:]
                        left_ys, left_xs, _ = galvo_corrections.fix_outliers(
                            np.array(left_ys), np.array(left_xs), max_y_shift,
                            max_x_shift, method='linear')

                        # Stitch together
                        right.join_with(left, left_xs, left_ys)
                        sorted_rois.remove(left)
                        break  # restart joining

            return sorted_rois

        # Stitch overlapping rois recursively
        print('Computing stitching parameters...')
        prev_num_rois = float('Inf')  # to enter the loop at least once
        while len(rois) < prev_num_rois:
            prev_num_rois = len(rois)

            # Join rows
            rois = join_rows(rois)

            # Join columns
            [roi.rot90() for roi in rois]
            rois = join_rows(rois)
            [roi.rot270() for roi in rois]

        # Compute slice-to slice alignment
        print('Computing slice-to-slice alignment...')
        for roi in rois:
            big_volume = roi.volume
            num_slices, image_height, image_width = big_volume.shape
            roi_key = {**key, 'roi_id': roi.roi_coordinates[0].id}
            um_per_px = (StackInfo.ROI() & roi_key).microns_per_pixel

            # Enhance
            neighborhood_size = 25 / um_per_px[1:]
            for i in range(num_slices):
                big_volume[i] = enhance(big_volume[i], neighborhood_size)

            # Drop 10% of the image borders
            skip_rows = int(round(image_height * 0.1))
            skip_columns = int(round(image_width * 0.1))
            big_volume = big_volume[:, skip_rows:-skip_rows, skip_columns: -skip_columns]

            y_aligns = np.zeros(num_slices)
            x_aligns = np.zeros(num_slices)
            for i in range(1, num_slices):
                # Align current slice to previous one
                y_aligns[i], x_aligns[i] = galvo_corrections.compute_motion_shifts(
                    big_volume[i], big_volume[i - 1], in_place=False)

            # Fix outliers
            max_y_shift, max_x_shift = 15 / um_per_px[1:]
            y_fixed, x_fixed, _ = galvo_corrections.fix_outliers(y_aligns, x_aligns,
                                                                 max_y_shift, max_x_shift)

            # Accumulate shifts so shift i is shift in i -1 plus shift to align i to i-1
            y_cumsum, x_cumsum = np.cumsum(y_fixed), np.cumsum(x_fixed)

            # Detrend to discard influence of vessels going through the slices
            filter_size = int(round(60 / um_per_px[0]))  # 60 microns in z
            filter_size += 1 if filter_size % 2 == 0 else 0
            if len(y_cumsum) > filter_size:
                smoothing_filter = signal.hann(filter_size)
                smoothing_filter /= sum(smoothing_filter)
                y_detrend = y_cumsum - mirrconv(y_cumsum, smoothing_filter)
                x_detrend = x_cumsum - mirrconv(x_cumsum, smoothing_filter)
            else:
                y_detrend = y_cumsum - y_cumsum.mean()
                x_detrend = x_cumsum - x_cumsum.mean()

            # Apply alignment shifts in roi
            for slice_, y_align, x_align in zip(roi.slices, y_detrend, x_detrend):
                slice_.y -= y_align
                slice_.x -= x_align
            for roi_coord in roi.roi_coordinates:
                roi_coord.ys = [prev_y - y_align for prev_y, y_align in zip(roi_coord.ys,
                                                                            y_detrend)]
                roi_coord.xs = [prev_x - x_align for prev_x, x_align in zip(roi_coord.xs,
                                                                            x_detrend)]

        # Insert in Stitching
        print('Inserting...')
        self.insert1(key)

        # Insert each stitched volume
        for volume_id, roi in enumerate(rois, start=1):
            self.Volume().insert1({**key, 'volume_id': volume_id})

            # Insert coordinates of each ROI forming this volume
            for roi_coord in roi.roi_coordinates:
                tuple_ = {**key, 'roi_id': roi_coord.id, 'volume_id': volume_id,
                          'stitch_xs': roi_coord.xs, 'stitch_ys': roi_coord.ys}
                self.ROICoordinates().insert1(tuple_)

        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        slack_user = (notify.SlackUser() & (experiment.Session() & key))
        for volume_key in (self.Volume() & key).fetch('KEY'):
            for roi_coord in (self.ROICoordinates() & volume_key).fetch(as_dict=True):
                center_z, num_slices, um_depth = (StackInfo.ROI() & roi_coord).fetch1(
                    'roi_z', 'roi_px_depth', 'roi_um_depth')
                first_z = center_z - um_depth / 2 + (um_depth / num_slices) / 2
                depths = first_z + (um_depth / num_slices) * np.arange(num_slices)

                fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
                axes[0].set_title('Center position (x)')
                axes[0].plot(depths, roi_coord['stitch_xs'])
                axes[1].set_title('Center position (y)')
                axes[1].plot(depths, roi_coord['stitch_ys'])
                axes[0].set_ylabel('Pixels')
                axes[0].set_xlabel('Depths')
                fig.tight_layout()
                img_filename = '/tmp/' + key_hash(key) + '.png'
                fig.savefig(img_filename, bbox_inches='tight')
                plt.close(fig)

                msg = ('stitch traces for {animal_id}-{session}-{stack_idx} volume '
                       '{volume_id} roi {roi_id}').format(**roi_coord)
                slack_user.notify(file=img_filename, file_title=msg)


@schema
class CorrectedStack(dj.Computed):
    definition = """ # all slices of each stack after corrections.

    -> Stitching.Volume                 # animal_id, session, stack_idx, volume_id
    ---
    z               : float             # (um) center of volume in the motor coordinate system (cortex is at 0)
    y               : float             # (um) center of volume in the motor coordinate system
    x               : float             # (um) center of volume in the motor coordinate system
    px_depth        : smallint          # number of slices
    px_height       : smallint          # lines per frame
    px_width        : smallint          # pixels per line
    um_depth        : float             # depth in microns 
    um_height       : float             # height in microns
    um_width        : float             # width in microns
    surf_z          : float             # (um) depth of first slice - half a z step (cortex is at z=0)     
    """

    class Slice(dj.Part):
        definition = """ # single slice of one stack

        -> CorrectedStack
        -> shared.Channel
        islice              : smallint          # index of slice in volume
        ---
        slice               : longblob          # image (height x width)
        """

    def _make_tuples(self, key):
        print('Correcting stack', key)

        for channel in range((StackInfo() & key).fetch1('nchannels')):
            # Correct ROIs
            rois = []
            for roi_tuple in (StackInfo.ROI() * Stitching.ROICoordinates() & key).fetch():
                # Load ROI
                roi_filename = (experiment.Stack.Filename() &
                                roi_tuple).local_filenames_as_wildcard
                roi = scanreader.read_scan(roi_filename)

                # Map: Apply corrections to each field in parallel
                f = performance.parallel_correct_stack  # function to map
                raster_phase = (RasterCorrection() & roi_tuple).fetch1('raster_phase')
                fill_fraction = (StackInfo() & key).fetch1('fill_fraction')
                y_shifts, x_shifts = (MotionCorrection() & roi_tuple).fetch1('y_shifts',
                                                                             'x_shifts')
                field_ids = roi_tuple['field_ids']
                results = performance.map_fields(f, roi, field_ids=field_ids,
                                                 channel=channel,
                                                 kwargs={'raster_phase': raster_phase,
                                                         'fill_fraction': fill_fraction,
                                                         'y_shifts': y_shifts,
                                                         'x_shifts': x_shifts})

                # Reduce: Collect results
                corrected_roi = np.empty((roi_tuple['roi_px_depth'],
                                          roi_tuple['roi_px_height'],
                                          roi_tuple['roi_px_width']), dtype=np.float32)
                for field_idx, corrected_field in results:
                    corrected_roi[field_idx] = corrected_field

                # Create ROI object (with pixel x, y, z coordinates)
                px_z = roi_tuple['roi_z'] * (roi_tuple['roi_px_depth'] /
                                             roi_tuple['roi_um_depth'])
                ys = list(roi_tuple['stitch_ys'])
                xs = list(roi_tuple['stitch_xs'])
                rois.append(stitching.StitchedROI(corrected_roi, x=xs, y=ys, z=px_z,
                                                  id_=roi_tuple['roi_id']))

            def join_rows(rois_):
                """ Iteratively join all rois that overlap in the same row."""
                sorted_rois = sorted(rois_, key=lambda roi: (roi.x, roi.y))

                prev_num_rois = float('inf')
                while len(sorted_rois) < prev_num_rois:
                    prev_num_rois = len(sorted_rois)

                    for left, right in itertools.combinations(sorted_rois, 2):
                        if left.is_aside_to(right):
                            left_xs = [s.x for s in left.slices]
                            left_ys = [s.y for s in left.slices]
                            right.join_with(left, left_xs, left_ys)
                            sorted_rois.remove(left)
                            break  # restart joining

                return sorted_rois

            # Stitch all rois together. This is convoluted because smooth blending in
            # join_with assumes rois are next to (not below or atop of) each other
            prev_num_rois = float('Inf')  # to enter the loop at least once
            while len(rois) < prev_num_rois:
                prev_num_rois = len(rois)

                # Join rows
                rois = join_rows(rois)

                # Join columns
                [roi.rot90() for roi in rois]
                rois = join_rows(rois)
                [roi.rot270() for roi in rois]

            # Check stitching went alright
            if len(rois) > 1:
                msg = 'ROIs for volume {} could not be stitched properly'.format(key)
                raise PipelineException(msg)
            stitched = rois[0]

            # Insert in CorrectedStack
            roi_info = StackInfo.ROI() & key & {'roi_id': stitched.roi_coordinates[0].id}
            um_per_px = roi_info.microns_per_pixel
            tuple_ = key.copy()
            tuple_['z'] = stitched.z * um_per_px[0]
            tuple_['y'] = stitched.y * um_per_px[1]
            tuple_['x'] = stitched.x * um_per_px[2]
            tuple_['px_depth'] = stitched.depth
            tuple_['px_height'] = stitched.height
            tuple_['px_width'] = stitched.width
            tuple_['um_depth'] = roi_info.fetch1('roi_um_depth')  # same as original rois
            tuple_['um_height'] = stitched.height * um_per_px[1]
            tuple_['um_width'] = stitched.width * um_per_px[2]
            tuple_['surf_z'] = (stitched.z - stitched.depth / 2) * um_per_px[0]
            self.insert1(tuple_, skip_duplicates=True)

            # Insert each slice
            for i, slice_ in enumerate(stitched.volume):
                self.Slice().insert1({**key, 'channel': channel + 1, 'islice': i + 1,
                                      'slice': slice_})

            self.notify({**key, 'channel': channel + 1})

    @notify.ignore_exceptions
    def notify(self, key):
        import imageio

        volume = (self & key).get_stack(channel=key['channel'])
        volume = volume[:: int(volume.shape[0] / 8)]  # volume at 8 diff depths
        video_filename = '/tmp/' + key_hash(key) + '.gif'
        imageio.mimsave(video_filename, float2uint8(volume), duration=1)

        msg = ('corrected stack for {animal_id}-{session}-{stack_idx} volume {volume_id} '
               'channel {channel}').format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=video_filename, file_title=msg,
                          channel='#pipeline_quality')

    @property
    def microns_per_pixel(self):
        """ Returns an array with microns per pixel in depth, height and width. """
        um_dims = self.fetch1('um_depth', 'um_height', 'um_width')
        px_dims = self.fetch1('px_depth', 'px_height', 'px_width')
        return np.array([um_dim / px_dim for um_dim, px_dim in zip(um_dims, px_dims)])

    def get_stack(self, channel=1):
        """ Get full stack (num_slices, height, width).

        :param int channel: What channel to use. Starts at 1

        :returns The stack: a (num_slices, image_height, image_width) array.
        :rtype: np.array (float32)
        """
        slice_rel = (CorrectedStack.Slice() & self & {'channel': channel})
        slices = slice_rel.fetch('slice', order_by='islice')
        return np.stack(slices)

    def save_as_tiff(self, filename='stack.tif'):
        """ Save current stack as a tiff file."""
        from tifffile import imsave

        # Create a composite interleaving channels
        height, width, depth = self.fetch1('px_height', 'px_width', 'px_depth')
        num_channels = (StackInfo() & self).fetch1('nchannels')
        composite = np.zeros([num_channels * depth, height, width], dtype=np.float32)
        for i in range(num_channels):
            composite[i::num_channels] = self.get_stack(i + 1)

        # Save
        print('Saving file at:', filename)
        imsave(filename, composite)

    def save_video(self, filename='stack.mp4', channel=1, fps=10, dpi=250):
        """ Creates an animation video showing a fly-over of the stack (top to bottom).

        :param string filename: Output filename (path + filename)
        :param int channel: What channel to use. Starts at 1
        :param int start_index: Where in the scan to start the video.
        :param int fps: Number of slices shown per second.
        :param int dpi: Dots per inch, controls the quality of the video.

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        from matplotlib import animation

        stack = self.get_stack(channel=channel)
        num_slices = stack.shape[0]

        fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
        im = fig.gca().imshow(stack[int(num_slices / 2)])
        video = animation.FuncAnimation(fig, lambda i: im.set_data(stack[i]), num_slices,
                                        interval=1000 / fps)
        fig.tight_layout()

        if not filename.endswith('.mp4'):
            filename += '.mp4'
        print('Saving video at:', filename)
        print('If this takes too long, stop it and call again with dpi <', dpi,
              '(default)')
        video.save(filename, dpi=dpi)

        return fig


@schema
class PreprocessedStack(dj.Computed):
    definition = """ # Resize to 1 um^3, apply local contrast normalization and sharpen

    -> CorrectedStack
    -> shared.Channel
    ---
    resized:        external-stack      # original stack resized to 1 um^3
    lcned:          external-stack      # local contrast normalized stack. Filter size: (3, 25, 25)
    sharpened:      external-stack      # sharpened stack. Filter size: 1
    """

    @property
    def key_source(self):
        # restrict each stack to its channels
        return (CorrectedStack * shared.Channel).proj() & CorrectedStack.Slice.proj()

    def make(self, key):
        from .utils import registration
        from .utils import enhancement

        # Load stack
        stack = (CorrectedStack() & key).get_stack(key['channel'])

        # Resize to be 1 um^3
        um_sizes = (CorrectedStack & key).fetch1('um_depth', 'um_height', 'um_width')
        resized = registration.resize(stack, um_sizes, desired_res=1)

        # Enhance
        lcned = enhancement.lcn(resized, (3, 25, 25))

        # Sharpen
        sharpened = enhancement.sharpen_2pimage(lcned, 1)

        # Insert
        self.insert1({**key, 'resized': resized, 'lcned': lcned, 'sharpened': sharpened})


@schema
class Surface(dj.Computed):
    definition = """ # Calculated surface of the brain

    -> PreprocessedStack
    -> shared.SurfaceMethod
    ---
    guessed_points    : longblob              # Array of guessed depths stored in (z,y,x) format
    surface_im        : longblob          # Matrix of fitted depth for each pixel in stack. Value is number of pixels to surface from top of array.
    lower_bound_im    : longblob          # Lower bound of 95th percentile confidence interval
    upper_bound_im    : longblob          # Upper bound of 95th percentile confidence interval
    """

    def make(self, key):

        # WARNINGS
        #  - This code assumes the surface will be in the top half of the stack
        #      - Only the top half of z-values are analyzed
        #  - Points along the edge are dropped to avoid errors due to blank space left by stack registration
        #  - This code assumes the surface median intensity should be in the bottom 60% of the range of values over z
        #      - ex. Intensities ranges from 10-20. Surface points must have an intensity < .6*(20-10) + 10 = 17.5
        #      - This is within the 2r x 2r window being analyzed
        #  - This code drops any 2r x 2r field where the first median value is above the 30th-percentile of the whole stack.
        #  - Windows where the final median intensity is below 10 are removed
        #      - Attempts to replace this with a percentile all fail
        #  - This code drops guessed depths > 95th-percentile and < 5th-percentile to be more robust to outliers

        valid_method_ids = [1]  # Used to check if method is implemented

        # SETTINGS
        # Note: Intial parameters for fitting set further down
        r = 50  # Radius of square in pixels
        upper_threshold_percent = 0.6  # Surface median intensity should be in the bottom X% of the *range* of medians
        gaussian_blur_size = 5  # Size of gaussian blur applied to slice
        min_points_allowed = 10  # If there are less than X points after filtering, throw an error
        bounds = ([0, 0, np.NINF, np.NINF, np.NINF], [np.Inf, np.Inf, np.Inf, np.Inf, np.Inf])  # Bounds for paraboloid fit
        ss_percent = 0.40  # Percentage of points to subsample for robustness check
        num_iterations = 1000  # Number of iterations to use for robustness check

        # DEFINITIONS
        def surface_eqn(data, a, b, c, d, f):
            x, y = data
            return a * x ** 2 + b * y ** 2 + c * x + d * y + f

        # MAIN BODY
        if int(key['surface_method_id']) not in valid_method_ids:
            raise PipelineException(f'Error: surface_method_id {key["surface_method_id"]} is not implemented')

        print('Calculating surface of brain for stack', key)
        full_stack = (PreprocessedStack & key).fetch1('resized')
        depth, height, width = full_stack.shape

        surface_guess_map = []
        r_xs = np.arange(r, width - width % r, r * 2)[1:-1]
        r_ys = np.arange(r, height - height % r, r * 2)[1:-1]
        full_mesh_x, full_mesh_y = np.meshgrid(np.arange(width), np.arange(height))

        # Surface z should be below this value
        z_lim = int(depth / 2)
        # Mean intensity of the first frame in the slice should be less than this value
        z_0_upper_threshold = np.percentile(full_stack, 30)

        for x in r_xs:
            for y in r_ys:
                stack_slice_medians = np.percentile(full_stack[0:z_lim, y - r:y + r, x - r:x + r], 50, axis=(1, 2))
                blurred_slice = ndimage.gaussian_filter1d(stack_slice_medians, gaussian_blur_size)

                upper_threshold_value = upper_threshold_percent * (
                        (blurred_slice.max() - blurred_slice.min()) - blurred_slice.min())
                upper_threshold_idx = np.where(blurred_slice > upper_threshold_value)[0][0]

                stack_slice_derivative = ndimage.sobel(blurred_slice)
                surface_z = np.argmax(stack_slice_derivative)

                if ((surface_z < upper_threshold_idx) and (blurred_slice[0] < z_0_upper_threshold) and
                        (blurred_slice[-1] > 10)):
                    surface_guess_map.append((surface_z, y, x))

        if len(surface_guess_map) < min_points_allowed:
            raise PipelineException(f"Surface calculation could not find enough valid points for {key}. Only "
                                    f"{len(surface_guess_map)} detected")

        # Drop the z-values lower than 5th-percentile or greater than 95th-percentile
        arr = np.array(surface_guess_map)
        top = np.percentile(arr[:, 0], 95)
        bot = np.percentile(arr[:, 0], 5)
        surface_guess_map = arr[np.logical_and(arr[:, 0] > bot, arr[:, 0] < top)]

        # Guess for initial parameters
        initial = [1, 1, int(width / 2), int(height / 2), 1]

        popt, pcov = optimize.curve_fit(surface_eqn, (surface_guess_map[:, 2], surface_guess_map[:, 1]),
                                        surface_guess_map[:, 0], p0=initial, maxfev=10000, bounds=bounds)

        calculated_surface_map = surface_eqn((full_mesh_x, full_mesh_y), *popt)

        all_sub_fitted_z = np.zeros((num_iterations, height, width))
        for i in np.arange(num_iterations):
            indices = np.random.choice(surface_guess_map.shape[0], int(surface_guess_map.shape[0] * ss_percent),
                                       replace=False)
            subsample = surface_guess_map[indices]
            sub_popt, sub_pcov = optimize.curve_fit(surface_eqn, (subsample[:, 2], subsample[:, 1]), subsample[:, 0],
                                                    p0=initial, maxfev=10000, bounds=bounds)
            all_sub_fitted_z[i, :, :] = surface_eqn((full_mesh_x, full_mesh_y), *sub_popt)

        z_min_matrix = np.percentile(all_sub_fitted_z, 5, axis=0)
        z_max_matrix = np.percentile(all_sub_fitted_z, 95, axis=0)

        surface_key = {**key, 'guessed_points': surface_guess_map, 'surface_im': calculated_surface_map,
                       'lower_bound_im': z_min_matrix, 'upper_bound_im': z_max_matrix}

        self.insert1(surface_key)

    def plot_surface3d(self, fig_height=7, fig_width=9):
        """ Plot guessed surface points and fitted surface mesh in 3D

        :param fig_height: Height of returned figure
        :param fig_width: Width of returned figure
        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        surface_guess_map, fitted_surface = self.fetch1('guessed_points', 'surface_im')
        surface_height, surface_width = fitted_surface.shape
        mesh_x, mesh_y = np.meshgrid(np.arange(surface_width), np.arange(surface_height))

        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.gca(projection='3d')

        surf = ax.plot_surface(mesh_x, mesh_y, fitted_surface, cmap=cm.coolwarm, linewidth=0, antialiased=False,
                               alpha=0.5)
        ax.scatter(surface_guess_map[:, 2], surface_guess_map[:, 1], surface_guess_map[:, 0], color='grey')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.invert_zaxis()

        return fig

    def plot_surface2d(self, r=50, z=None, fig_height=10, fig_width=20):
        """ Plot grid of guessed points and fitted surface depths spaced 2r apart on top of stack slice at depth = z

        :param r: Defines radius of square for each grid point
        :param z: Pixel depth of stack to show behind depth grid
        :param fig_height: Height of returned figure
        :param fig_width: Width of returned figure
        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """

        from matplotlib import cm

        full_stack = (PreprocessedStack & self).fetch1('resized')
        stack_depth, stack_height, stack_width = full_stack.shape
        surface_guess_map, fitted_surface = self.fetch1('guessed_points', 'surface_im')
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

        r_xs = np.arange(r, stack_width - stack_width % r, r * 2)
        r_ys = np.arange(r, stack_height - stack_height % r, r * 2)
        r_mesh_x, r_mesh_y = np.meshgrid(r_xs, r_ys)

        # Using median of depth to pick slice of stack to show if not defined
        if z is None:
            z = np.median(fitted_surface)
        if z < 0 or z > stack_depth:
            raise PipelineException(f'Error: Z parameter {z} is out of bounds for stack with depth {depth}')

        vmin = np.min((np.min(fitted_surface), np.min(surface_guess_map[:, 0])))
        vmax = np.max((np.max(fitted_surface), np.max(surface_guess_map[:, 0])))
        guessed_scatter = axes[0].scatter(x=surface_guess_map[:, 2], y=surface_guess_map[:, 1],
                                          c=surface_guess_map[:, 0], cmap=cm.hot, vmin=vmin, vmax=vmax)
        fitted_scatter = axes[1].scatter(x=r_mesh_x, y=r_mesh_y, c=fitted_surface[r_mesh_y, r_mesh_x], cmap=cm.hot,
                                         vmin=vmin, vmax=vmax)

        for point in surface_guess_map:
            axes[0].annotate(int(point[0]), (point[2], point[1]), color='white')
        for x in r_xs:
            for y in r_ys:
                axes[1].annotate(int(fitted_surface[y, x]), (x, y), color='white')

        fig.colorbar(guessed_scatter, ax=axes[0], fraction=0.05)
        axes[0].set_title(f'Guessed Depth, Z = {int(z)}')
        fig.colorbar(fitted_scatter, ax=axes[1], fraction=0.05)
        axes[1].set_title(f'Fitted Depth, Z = {int(z)}')
        for ax in axes:
            ax.imshow(full_stack[int(z), :, :])
            ax.set_axis_off()

        return fig


@schema
class SegmentationTask(dj.Manual):
    definition = """ # defines the target, the method and the channel to use for segmentation

   -> CorrectedStack
   -> shared.Channel
   -> shared.StackSegmMethod
   ---
   -> experiment.Compartment
   """

    def fill(self, key, channel=1, stacksegm_method=2, compartment='soma'):
        for stack_key in (CorrectedStack() & key).fetch(dj.key):
            tuple_ = {**stack_key, 'channel': channel,
                      'stacksegm_method': stacksegm_method,
                      'compartment': compartment}
            self.insert1(tuple_, ignore_extra_fields=True, skip_duplicates=True)


@schema
class Segmentation(dj.Computed):
    definition = """ # 3-d stack segmentation
    
    -> PreprocessedStack
    -> SegmentationTask
    ---
    segmentation            : external-stack # voxel-wise cell-ids (0 for background)
    nobjects                : int            # number of cells found            
    """

    class ConvNet(dj.Part):
        definition = """ # attributes particular to convnet based methods
        -> master
        ---
        centroids           : external-stack # voxel-wise probability of centroids
        probs               : external-stack # voxel-wise probability of cell nuclei 
        seg_threshold       : float          # threshold used for the probability maps
        min_voxels          : int            # minimum number of voxels (in cubic microns)
        max_voxels          : int            # maximum number of voxels (in cubic microns)
        compactness_factor  : float          # compactness factor used for the watershed segmentation
        """

    def _make_tuples(self, key):
        from .utils import segmentation3d

        # Set params
        seg_threshold = 0.8
        min_voxels = 65  # sphere of diameter 5
        max_voxels = 4186  # sphere of diameter 20
        compactness_factor = 0.05 # bigger produces rounder cells
        pad_mode = 'reflect'  # any valid mode in np.pad

        # Get stack at 1 um**3 voxels
        resized = (PreprocessedStack & key).fetch1('resized')

        # Segment
        if key['stacksegm_method'] not in [1, 2]:
            raise PipelineException('Unrecognized stack segmentation method: {}'.format(
                key['stacksegm_method']))
        method = 'single' if key['stacksegm_method'] == 1 else 'ensemble'
        centroids, probs, segmentation = segmentation3d.segment(resized, method, pad_mode,
                                                                seg_threshold, min_voxels,
                                                                max_voxels,
                                                                compactness_factor)

        # Insert
        self.insert1({**key, 'nobjects': segmentation.max(),
                      'segmentation': segmentation})
        self.ConvNet().insert1({**key, 'centroids': centroids, 'probs': probs,
                                'seg_threshold': seg_threshold, 'min_voxels': min_voxels,
                                'max_voxels': max_voxels,
                                'compactness_factor': compactness_factor})
        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        import imageio
        from bl3d import utils

        volume = (self & key).fetch1('segmentation')
        volume = volume[:: int(volume.shape[0] / 8)]  # volume at 8 diff depths
        colored = utils.colorize_label(volume)
        video_filename = '/tmp/' + key_hash(key) + '.gif'
        imageio.mimsave(video_filename, colored, duration=1)

        msg = 'segmentation for {animal_id}-{session}-{stack_idx}'.format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=video_filename, file_title=msg,
                          channel='#pipeline_quality')


@schema
class RegistrationTask(dj.Manual):
    definition = """ # declare scan fields to register to a stack as well as channels and method used

    -> CorrectedStack.proj(stack_session='session') # animal_id, stack_session, stack_idx, volume_id
    -> shared.Channel.proj(stack_channel='channel')
    -> experiment.Scan.proj(scan_session='session')  # animal_id, scan_session, scan_idx
    -> shared.Channel.proj(scan_channel='channel')
    -> shared.Field
    -> shared.RegistrationMethod
    """

    def fill(self, stack_key, scan_key, stack_channel=1, scan_channel=1, method=5):
        # Add stack attributes
        stack_rel = CorrectedStack() & stack_key
        if len(stack_rel) > 1:
            raise PipelineException('More than one stack match stack_key {}'.format(
                stack_key))
        tuple_ = stack_rel.proj(stack_session='session').fetch1()

        # Add common attributes
        tuple_['stack_channel'] = stack_channel
        tuple_['scan_channel'] = scan_channel
        tuple_['registration_method'] = method

        # Add scan attributes
        fields_rel = reso.ScanInfo.Field.proj() + meso.ScanInfo.Field.proj() & scan_key
        scan_animal_ids = np.unique(fields_rel.fetch('animal_id'))
        if len(scan_animal_ids) > 1 or scan_animal_ids[0] != tuple_['animal_id']:
            raise PipelineException('animal_id of stack and scan do not match.')
        for field in fields_rel.fetch():
            RegistrationTask().insert1({**tuple_, 'scan_session': field['session'],
                                        'scan_idx': field['scan_idx'],
                                        'field': field['field']}, skip_duplicates=True)


@schema
class Registration(dj.Computed):
    """ Our affine matrix A is represented as the usual 4 x 4 matrix using homogeneous
    coordinates, i.e., each point p is an [x, y, z, 1] vector.

    Because each field is flat, the original z coordinate will be the same at each grid
    position (zero) and thus it won't affect its final position, so our affine matrix has
    only 9 parameters: a11, a21, a31, a12, a22, a32, a14, a24 and a34.
    """
    definition = """ # align a 2-d scan field to a stack

    -> PreprocessedStack.proj(stack_session='session', stack_channel='channel')
    -> RegistrationTask
    """
    @property
    def key_source(self):
        stacks = PreprocessedStack.proj(stack_session='session', stack_channel='channel')
        
        return (stacks * RegistrationTask & {'registration_method': 5}) & (meso.SummaryImages + reso.SummaryImages).proj(scan_session='session')


    class Rigid(dj.Part):
        definition = """ # 3-d template matching keeping the stack straight

        -> master
        ---
        reg_x       : float         # (um) center of field in motor coordinate system
        reg_y       : float         # (um) center of field in motor coordinate system
        reg_z       : float         # (um) center of field in motor coordinate system
        score       : float         # cross-correlation score (-1 to 1)
        reg_field   : longblob      # extracted field from the stack in the specified position
        """

    class Affine(dj.Part):
        definition = """ # affine matrix learned via gradient ascent

        -> master
        ---
        a11             : float         # (um) element in row 1, column 1 of the affine matrix
        a21             : float         # (um) element in row 2, column 1 of the affine matrix
        a31             : float         # (um) element in row 3, column 1 of the affine matrix
        a12             : float         # (um) element in row 1, column 2 of the affine matrix
        a22             : float         # (um) element in row 2, column 2 of the affine matrix
        a32             : float         # (um) element in row 3, column 2 of the affine matrix
        reg_x           : float         # (um) element in row 1, column 4 of the affine matrix
        reg_y           : float         # (um) element in row 2, column 4 of the affine matrix
        reg_z           : float         # (um) element in row 3, column 4 of the affine matrix
        score           : float         # cross-correlation score (-1 to 1)
        reg_field       : longblob      # extracted field from the stack in the specified position
        """

    class NonRigid(dj.Part):
        definition = """ # affine plus deformation field learned via gradient descent

        -> master
        ---
        a11             : float         # (um) element in row 1, column 1 of the affine matrix
        a21             : float         # (um) element in row 2, column 1 of the affine matrix
        a31             : float         # (um) element in row 3, column 1 of the affine matrix
        a12             : float         # (um) element in row 1, column 2 of the affine matrix
        a22             : float         # (um) element in row 2, column 2 of the affine matrix
        a32             : float         # (um) element in row 3, column 2 of the affine matrix
        reg_x           : float         # (um) element in row 1, column 4 of the affine matrix
        reg_y           : float         # (um) element in row 2, column 4 of the affine matrix
        reg_z           : float         # (um) element in row 3, column 4 of the affine matrix
        landmarks       : longblob      # (um) x, y position of each landmark (num_landmarks x 2) assuming center of field is at (0, 0)
        deformations    : longblob      # (um) x, y, z deformations per landmark (num_landmarks x 3)
        score           : float         # cross-correlation score (-1 to 1)
        reg_field       : longblob      # extracted field from the stack in the specified position
        """

    class Params(dj.Part):
        definition = """ # document some parameters used for the registration

        -> master
        ---
        rigid_zrange    : int           # microns above and below experimenter's estimate (in z) to search for rigid registration
        lr_linear       : float         # learning rate for the linear part of the affine matrix
        lr_translation  : float         # learning rate for the translation vector
        affine_iters    : int           # number of iterations to learn the affine registration
        random_seed     : int           # seed used to initialize landmark deformations
        landmark_gap    : int           # number of microns between landmarks
        rbf_radius      : int           # critical radius for the gaussian radial basis function
        lr_deformations : float         # learning rate for the deformation values
        wd_deformations : float         # regularization term to control size of the deformations
        smoothness_factor : float       # regularization term to control curvature of warping field
        nonrigid_iters  : int           # number of iterations to optimize for the non-rigid parameters
        """

    def make(self, key):
        from .utils import registration
        from .utils import enhancement

        # Set params
        rigid_zrange = 80  # microns to search above and below estimated z for rigid registration
        lr_linear = 0.001  # learning rate / step size for the linear part of the affine matrix
        lr_translation = 1  # learning rate / step size for the translation vector
        affine_iters = 200  # number of optimization iterations to learn the affine parameters
        random_seed = 1234  # seed for torch random number generator (used to initialize deformations)
        landmark_gap = 100  # spacing for the landmarks
        rbf_radius = 150  # critical radius for the gaussian rbf
        lr_deformations = 0.1  # learning rate / step size for deformation values
        wd_deformations = 1e-4  # weight decay for deformations; controls their size
        smoothness_factor = 0.01  # factor to keep the deformation field smooth
        nonrigid_iters = 200  # number of optimization iterations for the nonrigid parameters

        # Get enhanced stack
        stack_key = {'animal_id': key['animal_id'], 'session': key['stack_session'],
                     'stack_idx': key['stack_idx'], 'volume_id': key['volume_id'],
                     'channel': key['stack_channel']}
        original_stack = (PreprocessedStack & stack_key).fetch1('resized')
        stack = (PreprocessedStack & stack_key).fetch1('sharpened')

        # Get field
        field_key = {'animal_id': key['animal_id'], 'session': key['scan_session'],
                     'scan_idx': key['scan_idx'], 'field': key['field'],
                     'channel': key['scan_channel']}
        pipe = (reso if reso.ScanInfo & field_key else meso if meso.ScanInfo & field_key
                else None)
        original_field = (pipe.SummaryImages.Average & field_key).fetch1(
            'average_image').astype(np.float32)

        # Enhance field
        field_dims = ((reso.ScanInfo if pipe == reso else meso.ScanInfo.Field) &
                      field_key).fetch1('um_height', 'um_width')
        original_field = registration.resize(original_field, field_dims, desired_res=1)
        field = enhancement.sharpen_2pimage(enhancement.lcn(original_field, (15, 15)), 1)

        # Drop some edges to avoid artifacts
        field = field[15:-15, 15:-15]
        stack = stack[5:-5, 15:-15, 15:-15]


        # RIGID REGISTRATION
        from skimage import feature

        # Get initial estimate of field depth from experimenters
        field_z = (pipe.ScanInfo.Field & field_key).fetch1('z')
        stack_z = (CorrectedStack & stack_key).fetch1('z')
        z_limits = stack_z - stack.shape[0] / 2, stack_z + stack.shape[0] / 2
        if field_z < z_limits[0] or field_z > z_limits[1]:
            print('Warning: Estimated depth ({}) outside stack range ({}-{}).'.format(
                field_z, *z_limits))

        # Run registration with no rotations
        px_z = field_z - stack_z + stack.shape[0] / 2 - 0.5
        mini_stack = stack[max(0, int(round(px_z - rigid_zrange))): int(round(
            px_z + rigid_zrange))]
        corrs = np.stack([feature.match_template(s, field, pad_input=True) for s in
                          mini_stack])
        smooth_corrs = ndimage.gaussian_filter(corrs, 0.7)

        # Get results
        min_z = max(0, int(round(px_z - rigid_zrange)))
        min_y = int(round(0.05 * stack.shape[1]))
        min_x = int(round(0.05 * stack.shape[2]))
        mini_corrs = smooth_corrs[:, min_y:-min_y, min_x:-min_x]
        rig_z, rig_y, rig_x = np.unravel_index(np.argmax(mini_corrs), mini_corrs.shape)

        # Rewrite coordinates with respect to original z
        rig_z = (min_z + rig_z + 0.5) - stack.shape[0] / 2
        rig_y = (min_y + rig_y + 0.5) - stack.shape[1] / 2
        rig_x = (min_x + rig_x + 0.5) - stack.shape[2] / 2

        del (field_z, stack_z, z_limits, px_z, mini_stack, corrs, smooth_corrs, min_z,
             min_y, min_x, mini_corrs)


        # AFFINE REGISTRATION
        import torch
        from torch import optim

        # Create field grid (height x width x 2)
        grid = registration.create_grid(field.shape)

        # Create torch tensors
        stack_ = torch.as_tensor(stack, dtype=torch.float32)
        field_ = torch.as_tensor(field, dtype=torch.float32)
        grid_ = torch.as_tensor(grid, dtype=torch.float32)

        # Define parameters and optimizer
        linear = torch.nn.Parameter(torch.eye(3)[:, :2])  # first two columns of rotation matrix
        translation = torch.nn.Parameter(torch.tensor([rig_x, rig_y, rig_z]))  # translation vector
        affine_optimizer = optim.Adam([{'params': linear, 'lr': lr_linear},
                                       {'params': translation, 'lr': lr_translation}])

        # Optimize
        for i in range(affine_iters):
            # Zero gradients
            affine_optimizer.zero_grad()

            # Compute gradients
            pred_grid = registration.affine_product(grid_, linear, translation) # w x h x 3
            pred_field = registration.sample_grid(stack_, pred_grid)
            corr_loss = -(pred_field * field_).sum() / (torch.norm(pred_field) *
                                                        torch.norm(field_))
            print('Corr at iteration {}: {:5.4f}'.format(i, -corr_loss))
            corr_loss.backward()

            # Update
            affine_optimizer.step()

        # Save em (originals will be modified during non-rigid registration)
        affine_linear = linear.detach().clone()
        affine_translation = translation.detach().clone()


        # NON-RIGID REGISTRATION
        # Inspired by the the Demon's Algorithm (Thirion, 1998)
        torch.manual_seed(random_seed) # we use random initialization below

        # Create landmarks (and their corresponding deformations)
        first_y = int(round((field.shape[0] % landmark_gap) / 2))
        first_x = int(round((field.shape[1] % landmark_gap) / 2))
        landmarks = grid_[first_x::landmark_gap, first_y::landmark_gap].contiguous().view(
            -1, 2)  # num_landmarks x 2

        # Compute rbf scores between landmarks and grid coordinates and between landmarks
        grid_distances = torch.norm(grid_.unsqueeze(-2) - landmarks, dim=-1)
        grid_scores = torch.exp(-(grid_distances * (1 / rbf_radius)) ** 2)  # w x h x num_landmarks
        landmark_distances = torch.norm(landmarks.unsqueeze(-2) - landmarks, dim=-1)
        landmark_scores = torch.exp(-(landmark_distances * (1 / 200)) ** 2)  # num_landmarks x num_landmarks

        # Define parameters and optimizer
        deformations = torch.nn.Parameter(torch.randn((landmarks.shape[0], 3)) / 10)  # N(0, 0.1)
        nonrigid_optimizer = optim.Adam([deformations], lr=lr_deformations,
                                        weight_decay=wd_deformations)

        # Optimize
        for i in range(nonrigid_iters):
            # Zero gradients
            affine_optimizer.zero_grad()  # we reuse affine_optimizer so the affine matrix changes slowly
            nonrigid_optimizer.zero_grad()

            # Compute grid with radial basis
            affine_grid = registration.affine_product(grid_, linear, translation)
            warping_field = torch.einsum('whl,lt->wht', (grid_scores, deformations))
            pred_grid = affine_grid + warping_field
            pred_field = registration.sample_grid(stack_, pred_grid)

            # Compute loss
            corr_loss = -(pred_field * field_).sum() / (torch.norm(pred_field) *
                                                        torch.norm(field_))

            # Compute cosine similarity between landmarks (and weight em by distance)
            norm_deformations = deformations / torch.norm(deformations, dim=-1,
                                                          keepdim=True)
            cosine_similarity = torch.mm(norm_deformations, norm_deformations.t())
            reg_term = -((cosine_similarity * landmark_scores).sum() /
                         landmark_scores.sum())

            # Compute gradients
            loss = corr_loss + smoothness_factor * reg_term
            print('Corr/loss at iteration {}: {:5.4f}/{:5.4f}'.format(i, -corr_loss,
                                                                      loss))
            loss.backward()

            # Update
            affine_optimizer.step()
            nonrigid_optimizer.step()

        # Save final results
        nonrigid_linear = linear.detach().clone()
        nonrigid_translation = translation.detach().clone()
        nonrigid_landmarks = landmarks.clone()
        nonrigid_deformations = deformations.detach().clone()


        # COMPUTE SCORES (USING THE ENHANCED AND CROPPED VERSION OF THE FIELD)
        # Rigid
        pred_grid = registration.affine_product(grid_, torch.eye(3)[:, :2],
                                                torch.tensor([rig_x, rig_y, rig_z]))
        pred_field = registration.sample_grid(stack_, pred_grid).numpy()
        rig_score = np.corrcoef(field.ravel(), pred_field.ravel())[0, 1]

        # Affine
        pred_grid = registration.affine_product(grid_, affine_linear, affine_translation)
        pred_field = registration.sample_grid(stack_, pred_grid).numpy()
        affine_score = np.corrcoef(field.ravel(), pred_field.ravel())[0, 1]

        # Non-rigid
        affine_grid = registration.affine_product(grid_, nonrigid_linear,
                                                  nonrigid_translation)
        warping_field = torch.einsum('whl,lt->wht', (grid_scores, nonrigid_deformations))
        pred_grid = affine_grid + warping_field
        pred_field = registration.sample_grid(stack_, pred_grid).numpy()
        nonrigid_score = np.corrcoef(field.ravel(), pred_field.ravel())[0, 1]


        # FIND FIELDS IN STACK
        # Create grid of original size (h x w x 2)
        original_grid = registration.create_grid(original_field.shape)

        # Create torch tensors
        original_stack_ = torch.as_tensor(original_stack, dtype=torch.float32)
        original_grid_ = torch.as_tensor(original_grid, dtype=torch.float32)

        # Rigid
        pred_grid = registration.affine_product(original_grid_, torch.eye(3)[:, :2],
                                                torch.tensor([rig_x, rig_y, rig_z]))
        rig_field = registration.sample_grid(original_stack_, pred_grid).numpy()

        # Affine
        pred_grid = registration.affine_product(original_grid_, affine_linear,
                                                affine_translation)
        affine_field = registration.sample_grid(original_stack_, pred_grid).numpy()

        # Non-rigid
        affine_grid = registration.affine_product(original_grid_, nonrigid_linear,
                                                  nonrigid_translation)
        original_grid_distances = torch.norm(original_grid_.unsqueeze(-2) -
                                             nonrigid_landmarks, dim=-1)
        original_grid_scores = torch.exp(-(original_grid_distances * (1 / rbf_radius)) ** 2)
        warping_field = torch.einsum('whl,lt->wht', (original_grid_scores,
                                                     nonrigid_deformations))
        pred_grid = affine_grid + warping_field
        nonrigid_field = registration.sample_grid(original_stack_, pred_grid).numpy()


        # Insert
        stack_z, stack_y, stack_x = (CorrectedStack & stack_key).fetch1('z', 'y', 'x')
        self.insert1(key)
        self.Params.insert1({**key, 'rigid_zrange': rigid_zrange, 'lr_linear': lr_linear,
                             'lr_translation': lr_translation,
                             'affine_iters': affine_iters, 'random_seed': random_seed,
                             'landmark_gap': landmark_gap, 'rbf_radius': rbf_radius,
                             'lr_deformations': lr_deformations,
                             'wd_deformations': wd_deformations,
                             'smoothness_factor': smoothness_factor,
                             'nonrigid_iters': nonrigid_iters})
        self.Rigid.insert1({**key, 'reg_x': stack_x + rig_x, 'reg_y': stack_y + rig_y,
                            'reg_z': stack_z + rig_z, 'score': rig_score,
                            'reg_field': rig_field})
        self.Affine.insert1({**key, 'a11': affine_linear[0, 0].item(),
                             'a21': affine_linear[1, 0].item(),
                             'a31': affine_linear[2, 0].item(),
                             'a12': affine_linear[0, 1].item(),
                             'a22': affine_linear[1, 1].item(),
                             'a32': affine_linear[2, 1].item(),
                             'reg_x': stack_x + affine_translation[0].item(),
                             'reg_y': stack_y + affine_translation[1].item(),
                             'reg_z': stack_z + affine_translation[2].item(),
                             'score': affine_score, 'reg_field': affine_field})
        self.NonRigid.insert1({**key, 'a11': nonrigid_linear[0, 0].item(),
                               'a21': nonrigid_linear[1, 0].item(),
                               'a31': nonrigid_linear[2, 0].item(),
                               'a12': nonrigid_linear[0, 1].item(),
                               'a22': nonrigid_linear[1, 1].item(),
                               'a32': nonrigid_linear[2, 1].item(),
                               'reg_x': stack_x + nonrigid_translation[0].item(),
                               'reg_y': stack_y + nonrigid_translation[1].item(),
                               'reg_z': stack_z + nonrigid_translation[2].item(),
                               'landmarks': nonrigid_landmarks.numpy(),
                               'deformations': nonrigid_deformations.numpy(),
                               'score': nonrigid_score, 'reg_field': nonrigid_field})
        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        # No notifications
        pass

    def get_grid(self, type='affine', desired_res=1):
        """ Get registered grid for this registration. """
        import torch
        from .utils import registration

        # Get field
        field_key = self.proj(session='scan_session')
        field_dims = (reso.ScanInfo & field_key or meso.ScanInfo.Field &
                      field_key).fetch1('um_height', 'um_width')

        # Create grid at desired resolution
        grid = registration.create_grid(field_dims, desired_res=desired_res)  # h x w x 2
        grid = torch.as_tensor(grid, dtype=torch.float32)

        # Apply required transform
        if type == 'rigid':
            params = (Registration.Rigid & self).fetch1('reg_x', 'reg_y', 'reg_z')
            delta_x, delta_y, delta_z = params
            linear = torch.eye(3)[:, :2]
            translation = torch.tensor([delta_x, delta_y, delta_z])

            pred_grid = registration.affine_product(grid, linear, translation)
        elif type == 'affine':
            params = (Registration.Affine & self).fetch1('a11', 'a21', 'a31', 'a12',
                                                         'a22', 'a32', 'reg_x', 'reg_y',
                                                         'reg_z')
            a11, a21, a31, a12, a22, a32, delta_x, delta_y, delta_z = params
            linear = torch.tensor([[a11, a12], [a21, a22], [a31, a32]])
            translation = torch.tensor([delta_x, delta_y, delta_z])

            pred_grid = registration.affine_product(grid, linear, translation)
        elif type == 'nonrigid':
            params = (Registration.NonRigid & self).fetch1('a11', 'a21', 'a31', 'a12',
                                                           'a22', 'a32', 'reg_x', 'reg_y',
                                                           'reg_z', 'landmarks',
                                                           'deformations')
            rbf_radius = (Registration.Params & self).fetch1('rbf_radius')
            a11, a21, a31, a12, a22, a32, delta_x, delta_y, delta_z, landmarks, deformations = params
            linear = torch.tensor([[a11, a12], [a21, a22], [a31, a32]])
            translation = torch.tensor([delta_x, delta_y, delta_z])
            landmarks = torch.from_numpy(landmarks)
            deformations = torch.from_numpy(deformations)

            affine_grid = registration.affine_product(grid, linear, translation)
            grid_distances = torch.norm(grid.unsqueeze(-2) - landmarks, dim=-1)
            grid_scores = torch.exp(-(grid_distances * (1 / rbf_radius)) ** 2)
            warping_field = torch.einsum('whl,lt->wht', (grid_scores, deformations))

            pred_grid = affine_grid + warping_field
        else:
            raise PipelineException('Unrecognized registration.')

        return pred_grid.numpy()

    def plot_grids(self, desired_res=5):
        """ Plot the grids for this different registrations as 3-d surfaces."""
        # Get grids at desired resoultion
        rig_grid = self.get_grid('rigid', desired_res)
        affine_grid = self.get_grid('affine', desired_res)
        nonrigid_grid = self.get_grid('nonrigid', desired_res)

        # Plot surfaces
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D

        fig = plt.figure(figsize=plt.figaspect(0.5) * 1.5)
        ax = fig.gca(projection='3d')
        ax.plot_surface(rig_grid[..., 0], rig_grid[..., 1], rig_grid[..., 2], alpha=0.5)
        ax.plot_surface(affine_grid[..., 0], affine_grid[..., 1], affine_grid[..., 2],
                        alpha=0.5)
        ax.plot_surface(nonrigid_grid[..., 0], nonrigid_grid[..., 1],
                        nonrigid_grid[..., 2], alpha=0.5)
        ax.set_aspect('equal')
        ax.invert_zaxis()

        return fig


@schema
class FieldSegmentation(dj.Computed):
    definition = """ # structural segmentation of a 2-d field (using the affine registration)
    
    -> Segmentation.proj(stack_session='session', stacksegm_channel='channel')
    -> Registration
    ---
    segm_field          : longblob      # field (image x height) of cell ids at 1 um/px
    """

    class StackUnit(dj.Part):
        definition = """ # single unit from the stack that appears in the field

        -> master
        sunit_id        : int           # id in the stack segmentation
        ---
        depth           : int           # (um) size in z   
        height          : int           # (um) size in y
        width           : int           # (um) size in x
        volume          : float         # (um) volume of the 3-d unit
        area            : float         # (um) area of the 2-d mask  
        sunit_z         : float         # (um) centroid for the 3d unit in the motor coordinate system
        sunit_y         : float         # (um) centroid for the 3d unit in the motor coordinate system
        sunit_x         : float         # (um) centroid for the 3d unit in the motor coordinate system
        mask_z          : float         # (um) centroid for the 2d mask in the motor coordinate system
        mask_y          : float         # (um) centroid for the 2d mask in the motor coordinate system
        mask_x          : float         # (um) centroid for the 2d mask in the motor coordinate system
        distance        : float         # (um) euclidean distance between centroid of 2-d mask and 3-d unit
        """

    def _make_tuples(self, key):
        from skimage import measure

        # Get structural segmentation
        stack_key = {'animal_id': key['animal_id'], 'session': key['stack_session'],
                     'stack_idx': key['stack_idx'], 'volume_id': key['volume_id'],
                     'channel': key['stacksegm_channel']}
        instance = (Segmentation & stack_key).fetch1('segmentation')

        # Get segmented field
        grid = (Registration & key).get_grid(type='affine', desired_res=1)
        stack_center = np.array((CorrectedStack & stack_key).fetch1('z', 'y', 'x'))
        px_grid = (grid[..., ::-1] - stack_center - 0.5 + np.array(instance.shape) / 2)
        segmented_field = ndimage.map_coordinates(instance, np.moveaxis(px_grid, -1, 0),
                                                  order=0)  # nearest neighbor sampling

        # Insert in FieldSegmentation
        self.insert1({**key, 'segm_field': segmented_field})

        # Insert each StackUnit
        instance_props = measure.regionprops(instance)
        instance_labels = np.array([p.label for p in instance_props])
        for prop in measure.regionprops(segmented_field):
            sunit_id = prop.label
            instance_prop = instance_props[np.argmax(instance_labels == sunit_id)]

            depth = (instance_prop.bbox[3] - instance_prop.bbox[0])
            height = (instance_prop.bbox[4] - instance_prop.bbox[1])
            width = (instance_prop.bbox[5] - instance_prop.bbox[2])
            volume = instance_prop.area
            sunit_z, sunit_y, sunit_x = (stack_center + np.array(instance_prop.centroid) -
                                         np.array(instance.shape) / 2 + 0.5)

            binary_sunit = segmented_field == sunit_id
            area = np.count_nonzero(binary_sunit)
            px_y, px_x = ndimage.measurements.center_of_mass(binary_sunit)
            px_coords = np.array([[px_y], [px_x]])
            mask_x, mask_y, mask_z = [ndimage.map_coordinates(grid[..., i], px_coords,
                                                              order=1)[0] for i in
                                      range(3)]
            distance = np.sqrt((sunit_z - mask_z) ** 2 + (sunit_y - mask_y) ** 2 +
                               (sunit_x - mask_x) ** 2)

            # Insert in StackUnit
            self.StackUnit.insert1({**key, 'sunit_id': sunit_id, 'depth': depth,
                                    'height': height, 'width': width, 'volume': volume,
                                    'area': area, 'sunit_z': sunit_z, 'sunit_y': sunit_y,
                                    'sunit_x': sunit_x, 'mask_z': mask_z,
                                    'mask_y': mask_y, 'mask_x': mask_x,
                                    'distance': distance})


@schema
class RegistrationOverTime(dj.Computed):
    definition = """ # register a field at different timepoints of recording
    
    -> PreprocessedStack.proj(stack_session='session', stack_channel='channel')
    -> RegistrationTask
    """
    @property
    def key_source(self):
        stacks = PreprocessedStack.proj(stack_session='session', stack_channel='channel')
        return stacks * RegistrationTask & {'registration_method': 5}

    class Chunk(dj.Part):
        definition = """ # single registered chunk

        -> master
        frame_num       : int           # frame number of the frame in the middle of this chunk
        ---
        initial_frame   : int           # initial frame used in this chunk (1-based)
        final_frame     : int           # final frame used in this chunk (1-based)
        avg_chunk       : longblob      # average field used for registration
        """

        def get_grid(self, type='nonrigid', desired_res=1):
            # TODO: Taken verbatim from Registration (minor changes for formatting), refactor
            """ Get registered grid for this registration. """
            import torch
            from .utils import registration

            # Get field
            field_key = self.proj(session='scan_session')
            field_dims = (reso.ScanInfo & field_key or meso.ScanInfo.Field &
                          field_key).fetch1('um_height', 'um_width')

            # Create grid at desired resolution
            grid = registration.create_grid(field_dims, desired_res=desired_res)  # h x w x 2
            grid = torch.as_tensor(grid, dtype=torch.float32)

            # Apply required transform
            if type == 'rigid':
                params = (RegistrationOverTime.Rigid & self).fetch1('reg_x', 'reg_y',
                                                                    'reg_z')
                delta_x, delta_y, delta_z = params
                linear = torch.eye(3)[:, :2]
                translation = torch.tensor([delta_x, delta_y, delta_z])

                pred_grid = registration.affine_product(grid, linear, translation)
            elif type == 'affine':
                params = (RegistrationOverTime.Affine & self).fetch1('a11', 'a21', 'a31',
                                                                     'a12', 'a22', 'a32',
                                                                     'reg_x', 'reg_y',
                                                                     'reg_z')
                a11, a21, a31, a12, a22, a32, delta_x, delta_y, delta_z = params
                linear = torch.tensor([[a11, a12], [a21, a22], [a31, a32]])
                translation = torch.tensor([delta_x, delta_y, delta_z])

                pred_grid = registration.affine_product(grid, linear, translation)
            elif type == 'nonrigid':
                params = (RegistrationOverTime.NonRigid & self).fetch1('a11', 'a21',
                                                                       'a31', 'a12',
                                                                       'a22', 'a32',
                                                                       'reg_x', 'reg_y',
                                                                       'reg_z',
                                                                       'landmarks',
                                                                       'deformations')
                rbf_radius = (RegistrationOverTime.Params & self).fetch1('rbf_radius')
                (a11, a21, a31, a12, a22, a32, delta_x, delta_y, delta_z, landmarks,
                 deformations) = params
                linear = torch.tensor([[a11, a12], [a21, a22], [a31, a32]])
                translation = torch.tensor([delta_x, delta_y, delta_z])
                landmarks = torch.from_numpy(landmarks)
                deformations = torch.from_numpy(deformations)

                affine_grid = registration.affine_product(grid, linear, translation)
                grid_distances = torch.norm(grid.unsqueeze(-2) - landmarks, dim=-1)
                grid_scores = torch.exp(-(grid_distances * (1 / rbf_radius)) ** 2)
                warping_field = torch.einsum('whl,lt->wht', (grid_scores, deformations))

                pred_grid = affine_grid + warping_field
            else:
                raise PipelineException('Unrecognized registration.')

            return pred_grid.numpy()

    class Rigid(dj.Part):
        definition = """ # rigid registration of a single chunk
        
        -> RegistrationOverTime.Chunk
        ---
        reg_x       : float         # (um) center of field in motor coordinate system
        reg_y       : float         # (um) center of field in motor coordinate system
        reg_z       : float         # (um) center of field in motor coordinate system
        score       : float         # cross-correlation score (-1 to 1)
        reg_field   : longblob      # extracted field from the stack in the specified position
        """

    class Affine(dj.Part):
        definition = """ # affine matrix learned via gradient ascent

        -> RegistrationOverTime.Chunk
        ---
        a11             : float         # (um) element in row 1, column 1 of the affine matrix
        a21             : float         # (um) element in row 2, column 1 of the affine matrix
        a31             : float         # (um) element in row 3, column 1 of the affine matrix
        a12             : float         # (um) element in row 1, column 2 of the affine matrix
        a22             : float         # (um) element in row 2, column 2 of the affine matrix
        a32             : float         # (um) element in row 3, column 2 of the affine matrix
        reg_x           : float         # (um) element in row 1, column 4 of the affine matrix
        reg_y           : float         # (um) element in row 2, column 4 of the affine matrix
        reg_z           : float         # (um) element in row 3, column 4 of the affine matrix
        score           : float         # cross-correlation score (-1 to 1)
        reg_field       : longblob      # extracted field from the stack in the specified position
        """

    class NonRigid(dj.Part):
        definition = """ # affine plus deformation field learned via gradient descent

        -> RegistrationOverTime.Chunk
        ---
        a11             : float         # (um) element in row 1, column 1 of the affine matrix
        a21             : float         # (um) element in row 2, column 1 of the affine matrix
        a31             : float         # (um) element in row 3, column 1 of the affine matrix
        a12             : float         # (um) element in row 1, column 2 of the affine matrix
        a22             : float         # (um) element in row 2, column 2 of the affine matrix
        a32             : float         # (um) element in row 3, column 2 of the affine matrix
        reg_x           : float         # (um) element in row 1, column 4 of the affine matrix
        reg_y           : float         # (um) element in row 2, column 4 of the affine matrix
        reg_z           : float         # (um) element in row 3, column 4 of the affine matrix
        landmarks       : longblob      # (um) x, y position of each landmark (num_landmarks x 2) assuming center of field is at (0, 0)
        deformations    : longblob      # (um) x, y, z deformations per landmark (num_landmarks x 3)
        score           : float         # cross-correlation score (-1 to 1)
        reg_field       : longblob      # extracted field from the stack in the specified position
        """

    class Params(dj.Part):
        definition = """ # document some parameters used for the registration

        -> master
        ---
        rigid_zrange    : int           # microns above and below experimenter's estimate (in z) to search for rigid registration
        lr_linear       : float         # learning rate for the linear part of the affine matrix
        lr_translation  : float         # learning rate for the translation vector
        affine_iters    : int           # number of iterations to learn the affine registration
        random_seed     : int           # seed used to initialize landmark deformations
        landmark_gap    : int           # number of microns between landmarks
        rbf_radius      : int           # critical radius for the gaussian radial basis function
        lr_deformations : float         # learning rate for the deformation values
        wd_deformations : float         # regularization term to control size of the deformations
        smoothness_factor : float       # regularization term to control curvature of warping field
        nonrigid_iters  : int           # number of iterations to optimize for the non-rigid parameters
        """

    def make(self, key):
        from .utils import registration
        from .utils import enhancement

        # Set params
        rigid_zrange = 80  # microns to search above and below estimated z for rigid registration
        lr_linear = 0.001  # learning rate / step size for the linear part of the affine matrix
        lr_translation = 1  # learning rate / step size for the translation vector
        affine_iters = 200  # number of optimization iterations to learn the affine parameters
        random_seed = 1234  # seed for torch random number generator (used to initialize deformations)
        landmark_gap = 100  # spacing for the landmarks
        rbf_radius = 150  # critical radius for the gaussian rbf
        lr_deformations = 0.1  # learning rate / step size for deformation values
        wd_deformations = 1e-4  # weight decay for deformations; controls their size
        smoothness_factor = 0.01  # factor to keep the deformation field smooth
        nonrigid_iters = 200  # number of optimization iterations for the nonrigid parameters

        # Get enhanced stack
        stack_key = {'animal_id': key['animal_id'], 'session': key['stack_session'],
                     'stack_idx': key['stack_idx'], 'volume_id': key['volume_id'],
                     'channel': key['stack_channel']}
        original_stack = (PreprocessedStack & stack_key).fetch1('resized')
        stack = (PreprocessedStack & stack_key).fetch1('sharpened')
        stack = stack[5:-5, 15:-15, 15:-15]  # drop some edges

        # Get corrected scan
        field_key = {'animal_id': key['animal_id'], 'session': key['scan_session'],
                     'scan_idx': key['scan_idx'], 'field': key['field'],
                     'channel': key['scan_channel']}
        pipe = (reso if reso.ScanInfo & field_key else meso if meso.ScanInfo & field_key
        else None)
        scan = RegistrationOverTime._get_corrected_scan(field_key)

        # Get initial estimate of field depth from experimenters
        field_z = (pipe.ScanInfo.Field & field_key).fetch1('z')
        stack_z = (CorrectedStack & stack_key).fetch1('z')
        z_limits = stack_z - stack.shape[0] / 2, stack_z + stack.shape[0] / 2
        if field_z < z_limits[0] or field_z > z_limits[1]:
            print('Warning: Estimated depth ({}) outside stack range ({}-{}).'.format(
                field_z, *z_limits))

        # Compute best chunk size: each lasts the same (~15 minutes)
        fps = (pipe.ScanInfo & field_key).fetch1('fps')
        num_frames = scan.shape[-1]
        overlap = int(round(3 * 60 * fps))  # ~ 3 minutes
        num_chunks = int(np.ceil((num_frames - overlap) / (15 * 60 * fps - overlap)))
        chunk_size = int(np.floor((num_frames - overlap) / num_chunks + overlap))  # *
        # * distributes frames in the last (incomplete) chunk to the other chunks

        # Insert in RegistrationOverTime and Params (once per field)
        self.insert1(key)
        self.Params.insert1(
            {**key, 'rigid_zrange': rigid_zrange, 'lr_linear': lr_linear,
             'lr_translation': lr_translation, 'affine_iters': affine_iters,
             'random_seed': random_seed, 'landmark_gap': landmark_gap,
             'rbf_radius': rbf_radius, 'lr_deformations': lr_deformations,
             'wd_deformations': wd_deformations, 'smoothness_factor': smoothness_factor,
             'nonrigid_iters': nonrigid_iters})

        # Iterate over chunks
        for initial_frame in range(0, num_frames - chunk_size, chunk_size - overlap):
            # Get next chunk
            final_frame = initial_frame + chunk_size
            chunk = scan[..., initial_frame: final_frame]

            # Enhance field
            field_dims = ((reso.ScanInfo if pipe == reso else meso.ScanInfo.Field) &
                          field_key).fetch1('um_height', 'um_width')
            original_field = registration.resize(chunk.mean(-1), field_dims,
                                                 desired_res=1)
            field = enhancement.sharpen_2pimage(enhancement.lcn(original_field, 15), 1)
            field = field[15:-15, 15:-15] # drop some edges


            # TODO: From here until Insert is taken verbatim from Registration, refactor
            #  RIGID REGISTRATION
            from skimage import feature

            # Run registration with no rotations
            px_z = field_z - stack_z + stack.shape[0] / 2 - 0.5
            mini_stack = stack[max(0, int(round(px_z - rigid_zrange))): int(round(
                px_z + rigid_zrange))]
            corrs = np.stack([feature.match_template(s, field, pad_input=True) for s in
                              mini_stack])
            smooth_corrs = ndimage.gaussian_filter(corrs, 0.7)

            # Get results
            min_z = max(0, int(round(px_z - rigid_zrange)))
            min_y = int(round(0.05 * stack.shape[1]))
            min_x = int(round(0.05 * stack.shape[2]))
            mini_corrs = smooth_corrs[:, min_y:-min_y, min_x:-min_x]
            rig_z, rig_y, rig_x = np.unravel_index(np.argmax(mini_corrs),
                                                   mini_corrs.shape)

            # Rewrite coordinates with respect to original z
            rig_z = (min_z + rig_z + 0.5) - stack.shape[0] / 2
            rig_y = (min_y + rig_y + 0.5) - stack.shape[1] / 2
            rig_x = (min_x + rig_x + 0.5) - stack.shape[2] / 2

            del px_z, mini_stack, corrs, smooth_corrs, min_z, min_y, min_x, mini_corrs

            # AFFINE REGISTRATION
            import torch
            from torch import optim

            # Create field grid (height x width x 2)
            grid = registration.create_grid(field.shape)

            # Create torch tensors
            stack_ = torch.as_tensor(stack, dtype=torch.float32)
            field_ = torch.as_tensor(field, dtype=torch.float32)
            grid_ = torch.as_tensor(grid, dtype=torch.float32)

            # Define parameters and optimizer
            linear = torch.nn.Parameter(torch.eye(3)[:, :2])  # first two columns of rotation matrix
            translation = torch.nn.Parameter(torch.tensor([rig_x, rig_y, rig_z]))  # translation vector
            affine_optimizer = optim.Adam([{'params': linear, 'lr': lr_linear},
                                           {'params': translation, 'lr': lr_translation}])

            # Optimize
            for i in range(affine_iters):
                # Zero gradients
                affine_optimizer.zero_grad()

                # Compute gradients
                pred_grid = registration.affine_product(grid_, linear, translation)  # w x h x 3
                pred_field = registration.sample_grid(stack_, pred_grid)
                corr_loss = -(pred_field * field_).sum() / (torch.norm(pred_field) *
                                                            torch.norm(field_))
                print('Corr at iteration {}: {:5.4f}'.format(i, -corr_loss))
                corr_loss.backward()

                # Update
                affine_optimizer.step()

            # Save them (originals will be modified during non-rigid registration)
            affine_linear = linear.detach().clone()
            affine_translation = translation.detach().clone()

            # NON-RIGID REGISTRATION
            # Inspired by the the Demon's Algorithm (Thirion, 1998)
            torch.manual_seed(random_seed)  # we use random initialization below

            # Create landmarks (and their corresponding deformations)
            first_y = int(round((field.shape[0] % landmark_gap) / 2))
            first_x = int(round((field.shape[1] % landmark_gap) / 2))
            landmarks = grid_[first_x::landmark_gap,
                              first_y::landmark_gap].contiguous().view(-1, 2)  # num_landmarks x 2

            # Compute rbf scores between landmarks and grid coordinates and between landmarks
            grid_distances = torch.norm(grid_.unsqueeze(-2) - landmarks, dim=-1)
            grid_scores = torch.exp(-(grid_distances * (1 / rbf_radius)) ** 2)  # w x h x num_landmarks
            landmark_distances = torch.norm(landmarks.unsqueeze(-2) - landmarks, dim=-1)
            landmark_scores = torch.exp(-(landmark_distances * (1 / 200)) ** 2)  # num_landmarks x num_landmarks

            # Define parameters and optimizer
            deformations = torch.nn.Parameter(torch.randn((landmarks.shape[0], 3)) / 10)  # N(0, 0.1)
            nonrigid_optimizer = optim.Adam([deformations], lr=lr_deformations,
                                            weight_decay=wd_deformations)

            # Optimize
            for i in range(nonrigid_iters):
                # Zero gradients
                affine_optimizer.zero_grad()  # we reuse affine_optimizer so the affine matrix changes slowly
                nonrigid_optimizer.zero_grad()

                # Compute grid with radial basis
                affine_grid = registration.affine_product(grid_, linear, translation)
                warping_field = torch.einsum('whl,lt->wht', (grid_scores, deformations))
                pred_grid = affine_grid + warping_field
                pred_field = registration.sample_grid(stack_, pred_grid)

                # Compute loss
                corr_loss = -(pred_field * field_).sum() / (torch.norm(pred_field) *
                                                            torch.norm(field_))

                # Compute cosine similarity between landmarks (and weight em by distance)
                norm_deformations = deformations / torch.norm(deformations, dim=-1,
                                                              keepdim=True)
                cosine_similarity = torch.mm(norm_deformations, norm_deformations.t())
                reg_term = -((cosine_similarity * landmark_scores).sum() /
                             landmark_scores.sum())

                # Compute gradients
                loss = corr_loss + smoothness_factor * reg_term
                print('Corr/loss at iteration {}: {:5.4f}/{:5.4f}'.format(i, -corr_loss,
                                                                          loss))
                loss.backward()

                # Update
                affine_optimizer.step()
                nonrigid_optimizer.step()

            # Save final results
            nonrigid_linear = linear.detach().clone()
            nonrigid_translation = translation.detach().clone()
            nonrigid_landmarks = landmarks.clone()
            nonrigid_deformations = deformations.detach().clone()

            # COMPUTE SCORES (USING THE ENHANCED AND CROPPED VERSION OF THE FIELD)
            # Rigid
            pred_grid = registration.affine_product(grid_, torch.eye(3)[:, :2],
                                                    torch.tensor([rig_x, rig_y, rig_z]))
            pred_field = registration.sample_grid(stack_, pred_grid).numpy()
            rig_score = np.corrcoef(field.ravel(), pred_field.ravel())[0, 1]

            # Affine
            pred_grid = registration.affine_product(grid_, affine_linear,
                                                    affine_translation)
            pred_field = registration.sample_grid(stack_, pred_grid).numpy()
            affine_score = np.corrcoef(field.ravel(), pred_field.ravel())[0, 1]

            # Non-rigid
            affine_grid = registration.affine_product(grid_, nonrigid_linear,
                                                      nonrigid_translation)
            warping_field = torch.einsum('whl,lt->wht', (grid_scores, nonrigid_deformations))
            pred_grid = affine_grid + warping_field
            pred_field = registration.sample_grid(stack_, pred_grid).numpy()
            nonrigid_score = np.corrcoef(field.ravel(), pred_field.ravel())[0, 1]

            # FIND FIELDS IN STACK
            # Create grid of original size (h x w x 2)
            original_grid = registration.create_grid(original_field.shape)

            # Create torch tensors
            original_stack_ = torch.as_tensor(original_stack, dtype=torch.float32)
            original_grid_ = torch.as_tensor(original_grid, dtype=torch.float32)

            # Rigid
            pred_grid = registration.affine_product(original_grid_, torch.eye(3)[:, :2],
                                                    torch.tensor([rig_x, rig_y, rig_z]))
            rig_field = registration.sample_grid(original_stack_, pred_grid).numpy()

            # Affine
            pred_grid = registration.affine_product(original_grid_, affine_linear,
                                                    affine_translation)
            affine_field = registration.sample_grid(original_stack_, pred_grid).numpy()

            # Non-rigid
            affine_grid = registration.affine_product(original_grid_, nonrigid_linear,
                                                      nonrigid_translation)
            original_grid_distances = torch.norm(original_grid_.unsqueeze(-2) -
                                                 nonrigid_landmarks, dim=-1)
            original_grid_scores = torch.exp(-(original_grid_distances *
                                               (1 / rbf_radius)) ** 2)
            warping_field = torch.einsum('whl,lt->wht', (original_grid_scores,
                                                         nonrigid_deformations))
            pred_grid = affine_grid + warping_field
            nonrigid_field = registration.sample_grid(original_stack_, pred_grid).numpy()


            # Insert chunk
            stack_z, stack_y, stack_x = (CorrectedStack & stack_key).fetch1('z', 'y', 'x')
            frame_num = int(round((initial_frame + final_frame) / 2))
            self.Chunk.insert1({**key, 'frame_num': frame_num + 1,
                                'initial_frame': initial_frame + 1,
                                'final_frame': final_frame, 'avg_chunk': original_field})
            self.Rigid.insert1({**key, 'frame_num': frame_num + 1,
                                'reg_x': stack_x + rig_x, 'reg_y': stack_y + rig_y,
                                'reg_z': stack_z + rig_z, 'score': rig_score,
                                'reg_field': rig_field})
            self.Affine.insert1({**key, 'frame_num': frame_num + 1,
                                 'a11': affine_linear[0, 0].item(),
                                 'a21': affine_linear[1, 0].item(),
                                 'a31': affine_linear[2, 0].item(),
                                 'a12': affine_linear[0, 1].item(),
                                 'a22': affine_linear[1, 1].item(),
                                 'a32': affine_linear[2, 1].item(),
                                 'reg_x': stack_x + affine_translation[0].item(),
                                 'reg_y': stack_y + affine_translation[1].item(),
                                 'reg_z': stack_z + affine_translation[2].item(),
                                 'score': affine_score,
                                 'reg_field': affine_field})
            self.NonRigid.insert1({**key, 'frame_num': frame_num + 1,
                                   'a11': nonrigid_linear[0, 0].item(),
                                   'a21': nonrigid_linear[1, 0].item(),
                                   'a31': nonrigid_linear[2, 0].item(),
                                   'a12': nonrigid_linear[0, 1].item(),
                                   'a22': nonrigid_linear[1, 1].item(),
                                   'a32': nonrigid_linear[2, 1].item(),
                                   'reg_x': stack_x + nonrigid_translation[0].item(),
                                   'reg_y': stack_y + nonrigid_translation[1].item(),
                                   'reg_z': stack_z + nonrigid_translation[2].item(),
                                   'landmarks': nonrigid_landmarks.numpy(),
                                   'deformations': nonrigid_deformations.numpy(),
                                   'score': nonrigid_score, 'reg_field': nonrigid_field})
        # self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        frame_num, zs, scores = (self.Affine & key).fetch('frame_num', 'reg_z', 'score')

        plt.plot(frame_num, -zs, zorder=1)
        plt.scatter(frame_num, -zs, marker='*', s=scores * 70, zorder=2, color='r')
        plt.title('Registration over time (star size represents confidence)')
        plt.ylabel('z (surface at 0)')
        plt.xlabel('Frames')
        img_filename = '/tmp/{}.png'.format(key_hash(key))
        plt.savefig(img_filename)
        plt.close()

        msg = ('registration over time of {animal_id}-{scan_session}-{scan_idx} field '
               '{field} to {animal_id}-{stack_session}-{stack_idx}')
        msg = msg.format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key &
                                           {'session': key['stack_session']})
        slack_user.notify(file=img_filename, file_title=msg)

    def _get_corrected_scan(key):
        # Read scan
        scan_filename = (experiment.Scan & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)

        # Get some params
        pipe = reso if (reso.ScanInfo() & key) else meso

        # Map: Correct scan in parallel
        f = performance.parallel_correct_scan  # function to map
        raster_phase = (pipe.RasterCorrection & key).fetch1('raster_phase')
        fill_fraction = (pipe.ScanInfo & key).fetch1('fill_fraction')
        y_shifts, x_shifts = (pipe.MotionCorrection & key).fetch1('y_shifts', 'x_shifts')
        kwargs = {'raster_phase': raster_phase, 'fill_fraction': fill_fraction,
                  'y_shifts': y_shifts, 'x_shifts': x_shifts}
        results = performance.map_frames(f, scan, field_id=key['field'] - 1,
                                         channel=key['channel'] - 1, kwargs=kwargs)

        # Reduce: Make a single array (height x width x num_frames)
        height, width, _ = results[0][1].shape
        corrected_scan = np.zeros([height, width, scan.num_frames], dtype=np.float32)
        for frames, chunk in results:
            corrected_scan[..., frames] = chunk

        return corrected_scan

    def session_plot(self):
        """ Create a registration plot for the session"""
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        # Check that plot is restricted to a single stack and a single session
        regot_key = self.fetch('KEY', limit=1)[0]
        stack_key = {n: regot_key[n] for n in ['animal_id', 'stack_session', 'stack_idx',
                                               'volume_id']}
        session_key = {n: regot_key[n] for n in ['animal_id', 'scan_session']}
        if len(self & stack_key) != len(self):
            raise PipelineException('Plot can only be generated for one stack at a time')
        if len(self & session_key) != len(self):
            raise PipelineException('Plot can only be generated for one session at a '
                                    'time')

        # Get field times and depths
        ts = []
        zs = []
        session_ts = (experiment.Session & regot_key &
                      {'session': regot_key['scan_session']}).fetch1('session_ts')
        for key in self.fetch('KEY'):
            field_key = {'animal_id': key['animal_id'], 'session': key['scan_session'],
                         'scan_idx': key['scan_idx'], 'field': key['field']}
            scan_ts = (experiment.Scan & field_key).fetch1('scan_ts')
            fps = (reso.ScanInfo & field_key or meso.ScanInfo & field_key).fetch1('fps')

            frame_nums, field_zs = (RegistrationOverTime.Affine & key).fetch('frame_num',
                                                                             'reg_z')
            field_ts = (scan_ts - session_ts).seconds + frame_nums / fps  # in seconds

            ts.append(field_ts)
            zs.append(field_zs)

        # Plot
        fig = plt.figure(figsize=(20, 8))
        for ts_, zs_ in zip(ts, zs):
            plt.plot(ts_ / 3600, zs_)
        plt.title('Registered zs for {animal_id}-{scan_session} into {animal_id}-'
                  '{stack_session}-{stack_idx} starting at {t}'.format(t=session_ts,
                                                                       **regot_key))
        plt.ylabel('Registered zs')
        plt.xlabel('Hours')

        # Plot formatting
        plt.gca().invert_yaxis()
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))
        plt.grid(linestyle='--', alpha=0.8)

        return fig


@schema
class Drift(dj.Computed):
    definition = """ # assuming a linear drift, compute the rate of drift (of the affine registration)
    
    -> RegistrationOverTime
    ---
    z_slope                 : float            # (um/hour) drift of the center of the field
    y_slope                 : float            # (um/hour) drift of the center of the field
    x_slope                 : float            # (um/hour) drift of the center of the field
    z_rmse                  : float            # (um) root mean squared error of the fit
    y_rmse                  : float            # (um) root mean squared error of the fit
    x_rmse                  : float            # (um) root mean squared error of the fit
    """
    @property
    def key_source(self):
        return RegistrationOverTime.aggr(RegistrationOverTime.Chunk.proj(),
                                         nchunks='COUNT(*)') & 'nchunks > 1'

    def _make_tuples(self, key):
        from sklearn import linear_model

        # Get drifts per axis
        frame_nums, zs, ys, xs = (RegistrationOverTime.Affine & key).fetch('frame_num',
                'reg_z', 'reg_y', 'reg_x')

        # Get scan fps
        field_key = {**key, 'session': key['scan_session']}
        fps = (reso.ScanInfo() & field_key or meso.ScanInfo() & field_key).fetch1('fps')

        # Fit a line through the values (robust regression)
        slopes = []
        rmses = []
        X = frame_nums.reshape(-1, 1)
        for y in [zs, ys, xs]:
            model = linear_model.TheilSenRegressor()
            model.fit(X, y)
            slopes.append(model.coef_[0] * fps * 3600)
            rmses.append(np.sqrt(np.mean(zs - model.predict(X)) ** 2))

        self.insert1({**key, 'z_slope': slopes[0], 'y_slope': slopes[1],
                      'x_slope': slopes[2], 'z_rmse': rmses[0], 'y_rmse': rmses[1],
                      'x_rmse': rmses[2]})

    def session_plot(self):
        """ Create boxplots for the session (one per scan)."""
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        # Check that plot is restricted to a single stack and a single session
        regot_key = self.fetch('KEY', limit=1)[0]
        stack_key = {n: regot_key[n] for n in ['animal_id', 'stack_session', 'stack_idx',
                                               'volume_id']}
        session_key = {n: regot_key[n] for n in ['animal_id', 'scan_session']}
        if len(self & stack_key) != len(self):
            raise PipelineException('Plot can only be generated for one stack at a time')
        if len(self & session_key) != len(self):
            raise PipelineException('Plot can only be generated for one session at a '
                                    'time')

        # Get field times and depths
        z_slopes = []
        scan_idxs = np.unique(self.fetch('scan_idx'))
        for scan_idx in scan_idxs:
            scan_slopes = (self & {**session_key, 'scan_idx': scan_idx}).fetch('z_slope')
            z_slopes.append(scan_slopes)

        # Plot
        fig = plt.figure(figsize=(7, 4))
        plt.boxplot(z_slopes)
        plt.title('Z drift for {animal_id}-{scan_session} into {animal_id}-'
                  '{stack_session}-{stack_idx}'.format(**regot_key))
        plt.ylabel('Z drift (um/hour)')
        plt.xlabel('Scans')
        plt.xticks(range(1, len(scan_idxs) + 1), scan_idxs)

        # Plot formatting
        plt.gca().invert_yaxis()
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(5))
        plt.grid(linestyle='--', alpha=0.8)

        return fig


@schema
class StackSet(dj.Computed):
    definition = """ # match segmented masks by proximity in the stack
    
    -> CorrectedStack.proj(stack_session='session')  # animal_id, stack_session, stack_idx, volume_id
    -> shared.RegistrationMethod
    -> shared.SegmentationMethod
    ---
    min_distance            :tinyint        # distance used as threshold to accept two masks as the same
    max_height              :tinyint        # maximum allowed height of a joint mask
    """

    @property
    def key_source(self):
        return (CorrectedStack.proj(stack_session='session') *
                shared.RegistrationMethod.proj() * shared.SegmentationMethod.proj() &
                Registration & {'segmentation_method': 6})

    class Unit(dj.Part):
        definition = """ # a unit in the stack
        
        -> master
        munit_id            :int        # unique id in the stack
        ---
        munit_x             :float      # (um) position of centroid in motor coordinate system
        munit_y             :float      # (um) position of centroid in motor coordinate system
        munit_z             :float      # (um) position of centroid in motor coordinate system
        """

    class Match(dj.Part):
        definition = """ # Scan unit to stack unit match (n:1 relation)
        
        -> master
        -> experiment.Scan.proj(scan_session='session')  # animal_id, scan_session, scan_idx
        unit_id             :int        # unit id from ScanSet.Unit
        ---
        -> StackSet.Unit
        """

    class MatchedUnit():
        """ Coordinates for a set of masks that form a single cell."""

        def __init__(self, key, x, y, z, plane_id):
            self.keys = [key]
            self.xs = [x]
            self.ys = [y]
            self.zs = [z]
            self.plane_ids = [plane_id]
            self.centroid = [x, y, z]

        def join_with(self, other):
            self.keys += other.keys
            self.xs += other.xs
            self.ys += other.ys
            self.zs += other.zs
            self.plane_ids += other.plane_ids
            self.centroid = [np.mean(self.xs), np.mean(self.ys), np.mean(self.zs)]

        def __lt__(self, other):
            """ Used for sorting. """
            return True

    def make(self, key):
        from scipy.spatial import distance
        import bisect

        # Set some params
        min_distance = 10
        max_height = 20

        # Create list of units
        units = []  # stands for matched units
        for field in Registration & key:
            # Edge case: when two channels are registered, we don't know which to use
            if len(Registration.proj(ignore='scan_channel') & field) > 1:
                msg = ('More than one channel was registered for {animal_id}-'
                       '{scan_session}-{scan_idx} field {field}'.format(**field))
                raise PipelineException(msg)

            # Get registered grid
            field_key = {'animal_id': field['animal_id'],
                         'session': field['scan_session'], 'scan_idx': field['scan_idx'],
                         'field': field['field']}
            pipe = reso if reso.ScanInfo & field_key else meso
            um_per_px = ((reso.ScanInfo if pipe == reso else meso.ScanInfo.Field) &
                         field_key).microns_per_pixel
            grid = (Registration & field).get_grid(type='affine', desired_res=um_per_px)

            # Create cell objects
            for channel_key in (pipe.ScanSet & field_key &
                                {'segmentation_method': key['segmentation_method']}):  # *
                somas = pipe.MaskClassification.Type & {'type': 'soma'}
                field_somas = pipe.ScanSet.Unit & channel_key & somas
                unit_keys, xs, ys = (pipe.ScanSet.UnitInfo & field_somas).fetch('KEY',
                        'px_x', 'px_y')
                px_coords = np.stack([ys, xs])
                xs, ys, zs = [ndimage.map_coordinates(grid[..., i], px_coords, order=1)
                              for i in range(3)]
                units += [StackSet.MatchedUnit(*args, key_hash(channel_key)) for args in
                          zip(unit_keys, xs, ys, zs)]
            # * Separating masks per channel allows masks in diff channels to be matched
        print(len(units), 'initial units')

        def find_close_units(centroid, centroids, min_distance):
            """ Finds centroids that are closer than min_distance to centroid. """
            dists = distance.cdist(np.expand_dims(centroid, 0), centroids)
            indices = np.flatnonzero(dists < min_distance)
            return indices, dists[0, indices]

        def is_valid(unit1, unit2, max_height):
            """ Checks that units belong to different fields and that the resulting unit
            would not be bigger than 20 microns."""
            different_fields = len(set(unit1.plane_ids) & set(unit2.plane_ids)) == 0
            acceptable_height = (max(unit1.zs + unit2.zs) - min(
                unit1.zs + unit2.zs)) < max_height
            return different_fields and acceptable_height

        # Create distance matrix
        # For memory efficiency we use an adjacency list with only the units at less than 10 microns
        centroids = np.stack([u.centroid for u in units])
        distance_list = []  # list of triples (distance, unit1, unit2)
        for i in range(len(units)):
            indices, distances = find_close_units(centroids[i], centroids[i + 1:],
                                                  min_distance)
            for dist, j in zip(distances, i + 1 + indices):
                if is_valid(units[i], units[j], max_height):
                    bisect.insort(distance_list, (dist, units[i], units[j]))
        print(len(distance_list), 'possible pairings')

        # Join units
        while (len(distance_list) > 0):
            # Get next pair of units
            d, unit1, unit2 = distance_list.pop(0)

            # Remove them from lists
            units.remove(unit1)
            units.remove(unit2)
            f = lambda x: (unit1 not in x[1:]) and (unit2 not in x[1:])
            distance_list = list(filter(f, distance_list))

            # Join them
            unit1.join_with(unit2)

            # Recalculate distances
            centroids = [u.centroid for u in units]
            indices, distances = find_close_units(unit1.centroid, centroids, min_distance)
            for dist, j in zip(distances, indices):
                if is_valid(unit1, units[j], max_height):
                    bisect.insort(distance_list, (d, unit1, units[j]))

            # Insert new unit
            units.append(unit1)
        print(len(units), 'number of final masks')

        # Insert
        self.insert1({**key, 'min_distance': min_distance, 'max_height': max_height})
        for munit_id, munit in zip(itertools.count(start=1), units):
            new_unit = {**key, 'munit_id': munit_id, 'munit_x': munit.centroid[0],
                        'munit_y': munit.centroid[1], 'munit_z': munit.centroid[2]}
            self.Unit().insert1(new_unit)
            for subunit_key in munit.keys:
                new_match = {**key, 'munit_id': munit_id, **subunit_key,
                             'scan_session': subunit_key['session']}
                self.Match().insert1(new_match, ignore_extra_fields=True)

        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        fig = (StackSet() & key).plot_centroids3d()
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)

        msg = ('StackSet for {animal_id}-{stack_session}-{stack_idx}: {num_units} final '
               'units').format(**key, num_units=len(self.Unit & key))
        slack_user = notify.SlackUser & (experiment.Session & key &
                                         {'session': key['stack_session']})
        slack_user.notify(file=img_filename, file_title=msg)

    def plot_centroids3d(self):
        """ Plots the centroids of all units in the motor coordinate system (in microns)

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        from mpl_toolkits.mplot3d import Axes3D

        # Get centroids
        xs, ys, zs = (StackSet.Unit & self).fetch('munit_x', 'munit_y', 'munit_z')

        # Plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')
        ax.scatter(xs, ys, zs, alpha=0.5)
        ax.invert_zaxis()
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.set_zlabel('z (um)')

        return fig


@schema
class Area(dj.Computed):
    definition = """ # transform area masks from annotated retinotopic maps into stack space

    -> PreprocessedStack.proj(stack_session='session',stack_channel='channel')
    -> experiment.Scan.proj(scan_session='session')
    -> shared.Channel.proj(scan_channel='channel')
    -> shared.RegistrationMethod
    -> shared.AreaMaskMethod
    ret_idx              : smallint                     # retinotopy map index for each animal
    ret_hash             : varchar(32)                  # single attribute representation of the key (used to avoid going over 16 attributes in the key)
    ---
    """

    class Mask(dj.Part):
        definition = """ # mask per area indicating membership

        -> master
        -> anatomy.Area
        ---
        mask             : blob                        # 2D mask of pixel area membership
        """

    @property
    def key_source(self):
        # anatomy code outputs masks per field for aim 2pScan and per concatenated plane for aim widefield
        map_rel = (anatomy.AreaMask.proj('ret_idx', scan_session='session') &
                   (experiment.Scan & 'aim="2pScan"').proj(scan_session='session'))
        stack_rel = Registration & 'registration_method = 5'

        heading = list(set(list(map_rel.heading.attributes) + list(stack_rel.heading.attributes)))
        heading.remove('field')
        heading.remove('brain_area')
        key_source = dj.U(*heading, 'mask_method') & (map_rel * stack_rel * shared.AreaMaskMethod)

        return key_source

    def make(self, key):
        from scipy.interpolate import griddata
        import cv2

        #same as key source but retains brain area attribute
        key['ret_hash'] = key_hash(key)
        map_rel = (anatomy.AreaMask.proj('ret_idx', scan_session='session') &
                   (experiment.Scan & 'aim="2pScan"').proj(stack_session='session'))
        stack_rel = Registration & 'registration_method = 5'

        heading = list(set(list(map_rel.heading.attributes) + list(stack_rel.heading.attributes)))
        heading.remove('field')
        area_keys = (dj.U(*heading, 'mask_method') & (map_rel * stack_rel * shared.AreaMaskMethod) & key).fetch('KEY')


        fetch_str = ['x', 'y', 'um_width', 'um_height', 'px_width', 'px_height']
        stack_rel = CorrectedStack.proj(*fetch_str, stack_session='session') & key
        cent_x, cent_y, um_w, um_h, px_w, px_h = stack_rel.fetch1(*fetch_str)

        # subtract edges so that all coordinates are relative to the field
        stack_edges = np.array((cent_x - um_w / 2, cent_y - um_h / 2))
        stack_px_dims = np.array((px_w, px_h))
        stack_um_dims = np.array((um_w, um_h))

        # 0.5 displacement returns the center of each pixel
        stack_px_grid = np.meshgrid(*[np.arange(d) + 0.5 for d in stack_px_dims])

        # for each area, transfer mask from all fields into the stack
        area_masks = []
        for area_key in area_keys:
            mask_rel = anatomy.AreaMask & area_key
            field_keys, masks = mask_rel.fetch('KEY', 'mask')
            stack_masks = []
            for field_key, field_mask in zip(field_keys, masks):
                field_res = (meso.ScanInfo.Field & field_key).microns_per_pixel
                grid_key = {**key, 'field': field_key['field']}

                # fetch transformation grid using built in function
                field2stack_um = (Registration & grid_key).get_grid(type='affine', desired_res=field_res)
                field2stack_um = (field2stack_um[..., :2]).transpose([2, 0, 1])

                # convert transformation grid into stack pixel space
                field2stack_px = [(grid - edge) * px_per_um for grid, edge, px_per_um
                                  in zip(field2stack_um, stack_edges, stack_px_dims / stack_um_dims)]


                grid_locs = np.array([f2s.ravel() for f2s in field2stack_px]).T
                grid_vals = field_mask.ravel()
                grid_query = np.array([stack_grid.ravel() for stack_grid in stack_px_grid]).T

                # griddata because scipy.interpolate.interp2d wasn't working for some reason
                # linear because nearest neighbor doesn't handle nans at the edge of the image
                stack_mask = griddata(grid_locs, grid_vals, grid_query, method='linear')
                stack_mask = np.round(np.reshape(stack_mask, (px_h, px_w)))

                stack_masks.append(stack_mask)

            # flatten all masks for area
            stack_masks = np.array(stack_masks)
            stack_masks[np.isnan(stack_masks)] = 0
            area_mask = np.max(stack_masks, axis=0)

            # close gaps in mask with 100 um kernel
            kernel_width = 100
            kernel = np.ones(np.round(kernel_width * (stack_px_dims / stack_um_dims)).astype(int))
            area_mask = cv2.morphologyEx(area_mask, cv2.MORPH_CLOSE, kernel)

            area_masks.append(area_mask)

        # locate areas where masks overlap and set to nan
        overlap_locs = np.sum(area_masks, axis=0) > 1

        # create reference map of non-overlapping area masks
        mod_masks = np.stack(area_masks.copy())
        mod_masks[:, overlap_locs] = np.nan
        ref_mask = np.max([mm * (i + 1) for i, mm in enumerate(mod_masks)], axis=0)

        # interpolate overlap pixels into reference mask
        non_nan_idx = np.invert(np.isnan(ref_mask))
        grid_locs = np.array([stack_grid[non_nan_idx].ravel() for stack_grid in stack_px_grid]).T
        grid_vals = ref_mask[non_nan_idx].ravel()
        grid_query = np.array([stack_grid[overlap_locs] for stack_grid in stack_px_grid]).T

        mask_assignments = griddata(grid_locs, grid_vals, grid_query, method='nearest')

        for loc, assignment in zip((np.array(grid_query) - 0.5).astype(int), mask_assignments):
            mod_masks[:, loc[1], loc[0]] = 0
            mod_masks[int(assignment - 1)][loc[1]][loc[0]] = 1

        area_keys = [{**area_key,**key,'mask': mod_mask} for area_key, mod_mask in zip(area_keys, mod_masks)]

        self.insert1(key)
        self.Mask.insert(area_keys)
