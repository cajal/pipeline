""" Schemas for structural stacks. """
import datajoint as dj
from datajoint.jobs import key_hash
import matplotlib.pyplot as plt
import numpy as np
import scanreader
from scipy import signal
import itertools

from . import experiment, notify, shared, reso, meso
from .utils import galvo_corrections, stitching, performance, enhancement
from .utils.signal import mirrconv, float2uint8
from .exceptions import PipelineException


schema = dj.schema('pipeline_stack', locals(), create_tables=False)
CURRENT_VERSION = 1


@schema
class Version(dj.Lookup):
    definition = """ # versions for the stack pipeline

    -> shared.PipelineVersion
    ---
    description = ''                : varchar(256)      # any notes on this version
    date = CURRENT_TIMESTAMP        : timestamp         # automatic
    """


@schema
class StackInfo(dj.Imported):
    definition = """ # master table with general data about the stacks

    -> experiment.Stack
    -> Version                          # stack version
    ---
    nrois           : tinyint           # number of ROIs
    nchannels       : tinyint           # number of channels
    z_step          : float             # (um) distance in z between adjacent slices (always positive)
    fill_fraction   : float             # raster scan temporal fill fraction (see scanimage)
    """
    @property
    def key_source(self):
        return experiment.Stack() * (Version() & {'pipe_version': CURRENT_VERSION})

    class ROI(dj.Part):
        definition = """ # 3-D volumes that compose this stack (usually tiled to form a bigger fov)

        -> StackInfo
        roi_id          : tinyint           # same as ScanImage's
        ---
        -> experiment.Stack.Filename
        field_ids           : blob              # list of field_ids (0-index) sorted from shallower to deeper
        roi_x               : float             # (um) center of ROI in the motor coordinate system
        roi_y               : float             # (um) center of ROI in the motor coordinate system
        roi_z               : float             # (um) initial depth in the motor coordinate system
        roi_px_height       : smallint          # lines per frame
        roi_px_width        : smallint          # pixels per line
        roi_px_depth        : smallint          # number of slices
        roi_um_height       : float             # height in microns
        roi_um_width        : float             # width in microns
        roi_um_depth        : float             # depth in microns
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
                field_ids = [i for i, field_roi in enumerate(stack.field_rois) if id_in_file in field_roi]
                field_depths = [stack.field_depths[i] - surf_z for i in field_ids]
            else:
                field_ids = range(stack.num_scanning_depths)
                motor_zero = surf_z - stack.motor_position_at_zero[2]
                if stack.is_slow_stack and not stack.is_slow_stack_with_fastZ: # using motor
                    initial_fastZ = stack.initial_secondary_z or 0
                    field_depths = [motor_zero - stack.field_depths[i] + 2 * initial_fastZ for i in field_ids]
                else: # using fastZ
                    field_depths = [motor_zero + stack.field_depths[i] for i in field_ids]
            field_depths, field_ids = zip(*sorted(zip(field_depths, field_ids)))
            tuple_['field_ids'] = field_ids

            # Get reso/meso specific coordinates
            x_zero, y_zero, _ = stack.motor_position_at_zero  # motor x, y at ScanImage's 0
            if stack.is_multiROI:
                tuple_['roi_x'] = x_zero + stack._degrees_to_microns(stack.fields[field_ids[0]].x)
                tuple_['roi_y'] = y_zero + stack._degrees_to_microns(stack.fields[field_ids[0]].y)
                tuple_['roi_px_height'] = stack.field_heights[field_ids[0]]
                tuple_['roi_px_width'] = stack.field_widths[field_ids[0]]
                tuple_['roi_um_height'] = stack.field_heights_in_microns[field_ids[0]]
                tuple_['roi_um_width'] = stack.field_widths_in_microns[field_ids[0]]
            else:
                tuple_['roi_x'] = x_zero
                tuple_['roi_y'] = y_zero #TODO: Add sign flip if ys in reso point upwards
                tuple_['roi_px_height'] = stack.image_height
                tuple_['roi_px_width'] = stack.image_width

                # Estimate height and width in microns using measured FOVs for similar setups
                fov_rel = (experiment.FOV() * experiment.Session() * experiment.Stack() &
                           key & 'session_date>=fov_ts')
                zooms = fov_rel.fetch('mag').astype(np.float32)  # zooms measured in same setup
                closest_zoom = zooms[np.argmin(np.abs(np.log(zooms / stack.zoom)))]
                dims = (fov_rel & 'ABS(mag - {}) < 1e-4'.format(closest_zoom)).fetch1('height', 'width')
                um_height, um_width = [float(um) * (closest_zoom / stack.zoom) for um in dims]
                tuple_['roi_um_height'] = um_height * stack._y_angle_scale_factor
                tuple_['roi_um_width'] = um_width * stack._x_angle_scale_factor

            # Get common parameters
            tuple_['roi_z'] = field_depths[0] #TODO: Add surf_depth
            tuple_['roi_px_depth'] = len(field_ids)
            tuple_['roi_um_depth'] = field_depths[-1] - field_depths[0] + 1
            tuple_['nframes'] = stack.num_frames
            tuple_['fps'] = stack.fps
            tuple_['bidirectional'] = stack.is_bidirectional
            tuple_['is_slow'] = stack.is_slow_stack

            self.insert1(tuple_)

        @property
        def microns_per_pixel(self):
            """ Returns an array with microns per pixel in height and width. """
            dims = self.fetch1('roi_um_height', 'roi_px_height', 'roi_um_width', 'roi_px_width')
            um_height, px_height, um_width, px_width = dims
            return np.array([um_height / px_height, um_width / px_width])

    def _make_tuples(self, key):
        """ Read and store stack information."""
        print('Reading header...')

        # Read files forming this stack
        filename_keys = (experiment.Stack.Filename() & key).fetch(dj.key)
        stacks = []
        for filename_key in filename_keys:
            stack_filename = (experiment.Stack.Filename() & filename_key).local_filenames_as_wildcard
            stacks.append(scanreader.read_scan(stack_filename))
        num_rois_per_file = [(stack.num_rois if stack.is_multiROI else 1) for stack in stacks]

        # Create Stack tuple
        tuple_ = key.copy()
        tuple_['nrois'] = np.sum(num_rois_per_file)
        tuple_['nchannels'] = stacks[0].num_channels
        tuple_['z_step'] = abs(stacks[0].scanning_depths[1] - stacks[0].scanning_depths[0])
        tuple_['fill_fraction'] = stacks[0].temporal_fill_fraction

        # Insert Stack
        StackInfo().insert1(tuple_)

        # Insert ROIs
        roi_id = 1
        for filename_key, num_rois, stack in zip(filename_keys, num_rois_per_file, stacks):
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
    @property
    def key_source(self):
        return StackInfo() & {'pipe_version': CURRENT_VERSION}

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
            roi_filename = (experiment.Stack.Filename() & roi_tuple).local_filenames_as_wildcard
            roi = scanreader.read_scan(roi_filename)

            for channel in range((StackInfo() & key).fetch1('nchannels')):
                # Map: Compute quality metrics in each field
                f = performance.parallel_quality_stack # function to map
                field_ids = roi_tuple['field_ids']
                results = performance.map_fields(f, roi, field_ids=field_ids, channel=channel)

                # Reduce: Collect results
                mean_intensities = np.empty((roi_tuple['roi_px_depth'], roi_tuple['nframes']))
                contrasts = np.empty((roi_tuple['roi_px_depth'], roi_tuple['nframes']))
                for field_idx, field_mis, field_contrasts, _ in results:
                    mean_intensities[field_idx] = field_mis
                    contrasts[field_idx] = field_contrasts
                frames = [res[3] for res in sorted(results, key=lambda res: res[0])]
                frames = np.stack(frames[:: int(len(frames) / 8)], axis=-1) # frames at 8 diff depths

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

        msg = 'summary frames for {animal_id}-{session}-{stack_idx} channel {channel}'.format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=video_filename, file_title=msg)

        # Send intensity and contrasts
        figsize = (min(4, contrasts.shape[1] / 10 + 1),  contrasts.shape[0] / 30 + 1) # set heuristically
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

        msg = 'quality images for {animal_id}-{session}-{stack_idx} channel {channel}'.format(**key)
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
    @property
    def key_source(self):
        return StackInfo.ROI() * CorrectionChannel() & {'pipe_version': CURRENT_VERSION}

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
            for field_id in field_ids[skip_fields: -2*skip_fields]:
                # Create template (average frame tapered to avoid edge artifacts)
                slice_ = roi[field_id, :, :, correction_channel, :].astype(np.float32, copy=False)
                anscombed = 2 * np.sqrt(slice_ - slice_.min(axis=(0, 1)) + 3 / 8) # anscombe transform
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
        msg = 'raster phase for {animal_id}-{session}-{stack_idx} roi {roi_id}: {phase}'.format(
                   **key, phase=(self & key).fetch1('raster_phase'))
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
            corrected = roi # in_place
            for i, field in enumerate(roi):
                corrected[i] = galvo_corrections.correct_raster(field, raster_phase, fill_fraction)
        return corrected


@schema
class MotionCorrection(dj.Computed):
    definition = """ # motion correction for each slice in the stack

    -> RasterCorrection
    ---
     y_shifts            : longblob      # y motion correction shifts (num_slices x num_frames)
     x_shifts            : longblob      # x motion correction shifts (num_slices x num_frames)
    """
    @property
    def key_source(self):
        return RasterCorrection() & {'pipe_version': CURRENT_VERSION}

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
            f = performance.parallel_motion_stack # function to map
            raster_phase = (RasterCorrection() & key).fetch1('raster_phase')
            fill_fraction = (StackInfo() & key).fetch1('fill_fraction')
            max_y_shift, max_x_shift = 20 / (StackInfo.ROI() & key).microns_per_pixel
            results = performance.map_fields(f, roi, field_ids=field_ids, channel=correction_channel,
                                             kwargs={'raster_phase': raster_phase, 'fill_fraction': fill_fraction,
                                                     'skip_rows': skip_rows, 'skip_cols': skip_cols,
                                                     'max_y_shift': max_y_shift, 'max_x_shift': max_x_shift})

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

        msg = 'motion shifts for {animal_id}-{session}-{stack_idx} roi {roi_id}'.format(**key)
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
        f = performance.parallel_correct_stack # function to map
        raster_phase = (RasterCorrection() & self).fetch1('raster_phase')
        fill_fraction = (StackInfo() & self).fetch1('fill_fraction')
        y_shifts, x_shifts = self.fetch1('y_shifts', 'x_shifts')
        results = performance.map_fields(f, roi, field_ids=field_ids, channel=channel,
                                         kwargs={'raster_phase': raster_phase,
                                                 'fill_fraction': fill_fraction,
                                                 'y_shifts': y_shifts, 'x_shifts': x_shifts})

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
        # run iff all ROIs have been processed
        stacks = StackInfo() - (StackInfo.ROI() - MotionCorrection())
        return stacks & {'pipe_version': CURRENT_VERSION}

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
        stitch_xs        : blob         # (px) center of each slice in the volume-wise coordinate system
        stitch_ys        : blob         # (px) center of each slice in the volume-wise coordinate system
        stitch_z         : float        # (um) initial depth in the motor coordinate system
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
            roi_filename = (experiment.Stack.Filename() & roi_tuple).local_filenames_as_wildcard
            roi = scanreader.read_scan(roi_filename)

            # Map: Apply corrections to each field in parallel
            f = performance.parallel_correct_stack # function to map
            raster_phase = (RasterCorrection() & roi_tuple).fetch1('raster_phase')
            fill_fraction = (StackInfo() & roi_tuple).fetch1('fill_fraction')
            y_shifts, x_shifts = (MotionCorrection() & roi_tuple).fetch1('y_shifts', 'x_shifts')
            field_ids = roi_tuple['field_ids']
            results = performance.map_fields(f, roi, field_ids=field_ids, channel=correction_channel,
                                             kwargs={'raster_phase': raster_phase,
                                                     'fill_fraction': fill_fraction,
                                                     'y_shifts': y_shifts, 'x_shifts': x_shifts,
                                                     'apply_anscombe': True})

            # Reduce: Collect results
            corrected_roi = np.empty((roi_tuple['roi_px_depth'], roi_tuple['roi_px_height'],
                                      roi_tuple['roi_px_width']), dtype=np.float32)
            for field_idx, corrected_field in results:
                corrected_roi[field_idx] = corrected_field

            # Create ROI object
            um_per_px = (StackInfo.ROI() & (StackInfo.ROI().proj() & roi_tuple)).microns_per_pixel
            px_y, px_x = (roi_tuple['roi_y'], roi_tuple['roi_x']) / um_per_px # in pixels
            rois.append(stitching.StitchedROI(corrected_roi, x=px_x, y=px_y,
                                              z=roi_tuple['roi_z'], id_=roi_tuple['roi_id']))

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

                        # Compute stitching shifts
                        neighborhood_size = 25 / (StackInfo.ROI() & roi_key).microns_per_pixel
                        left_ys, left_xs = [], []
                        for l, r in zip(left.slices, right.slices):
                            left_slice = enhance(l.slice, neighborhood_size)
                            right_slice = enhance(r.slice, neighborhood_size)
                            delta_y, delta_x = stitching.linear_stitch(left_slice, right_slice,
                                                                       r.x - l.x)
                            left_ys.append(r.y - delta_y)
                            left_xs.append(r.x - delta_x)

                        # Fix outliers
                        max_y_shift, max_x_shift = 10 / (StackInfo.ROI() & roi_key).microns_per_pixel
                        left_ys, left_xs, _ = galvo_corrections.fix_outliers(np.array(left_ys),
                                np.array(left_xs), max_y_shift, max_x_shift, method='linear')

                        # Stitch together
                        right.join_with(left, left_xs, left_ys)
                        sorted_rois.remove(left)
                        break # restart joining

            return sorted_rois

        # Stitch overlapping rois recursively
        print('Computing stitching parameters...')
        prev_num_rois = float('Inf') # to enter the loop at least once
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

            # Enhance
            neighborhood_size = 25 / (StackInfo.ROI() & roi_key).microns_per_pixel
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
                y_aligns[i], x_aligns[i] = galvo_corrections.compute_motion_shifts(big_volume[i],
                                                                 big_volume[i-1], in_place=False)

            # Fix outliers
            max_y_shift, max_x_shift = 15 / (StackInfo.ROI() & roi_key).microns_per_pixel
            y_fixed, x_fixed, _ = galvo_corrections.fix_outliers(y_aligns, x_aligns,
                                                                 max_y_shift, max_x_shift)

            # Accumulate shifts so shift i is shift in i -1 plus shift to align i to i-1
            y_cumsum, x_cumsum = np.cumsum(y_fixed), np.cumsum(x_fixed)

            # Detrend to discard influence of vessels going through the slices
            filter_size = int(round(60 / (StackInfo() & key).fetch1('z_step'))) # 60 microns in z
            if len(y_cumsum) > filter_size:
                smoothing_filter = signal.hann(filter_size + (1 if filter_size % 2 == 0 else 0))
                y_detrend = y_cumsum - mirrconv(y_cumsum, smoothing_filter / sum(smoothing_filter))
                x_detrend = x_cumsum - mirrconv(x_cumsum, smoothing_filter / sum(smoothing_filter))
            else:
                y_detrend = y_cumsum - y_cumsum.mean()
                x_detrend = x_cumsum - x_cumsum.mean()

            # Apply alignment shifts in roi
            for slice_, y_align, x_align in zip(roi.slices, y_detrend, x_detrend):
                slice_.y -= y_align
                slice_.x -= x_align
            for roi_coord in roi.roi_coordinates:
                roi_coord.ys = [prev_y - y_align for prev_y, y_align in zip(roi_coord.ys, y_detrend)]
                roi_coord.xs = [prev_x - x_align for prev_x, x_align in zip(roi_coord.xs, x_detrend)]

        # Insert in Stitching
        print('Inserting...')
        self.insert1(key)

        # Insert each stitched volume
        for volume_id, roi in enumerate(rois):
            self.Volume().insert1({**key, 'volume_id': volume_id + 1})

            # Insert coordinates of each ROI forming this volume
            for roi_coord in roi.roi_coordinates:
                tuple_ = {**key, 'roi_id': roi_coord.id, 'volume_id': volume_id + 1,
                          'stitch_xs': roi_coord.xs, 'stitch_ys': roi_coord.ys,
                          'stitch_z': roi.z}
                self.ROICoordinates().insert1(tuple_)

        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        slack_user = (notify.SlackUser() & (experiment.Session() & key))
        z_step = (StackInfo() & key).fetch1('z_step')
        for volume_key in (self.Volume() & key).fetch('KEY'):
            for roi_coord in (self.ROICoordinates() & volume_key).fetch(as_dict=True):
                first_z, num_slices = (StackInfo.ROI() & roi_coord).fetch1('roi_z', 'roi_px_depth')
                depths = first_z + z_step * np.arange(num_slices)

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

    -> Stitching.Volume                 # animal_id, session, stack_idx, volume_id, pipe_version
    ---
    x               : float             # (px) center of volume in a volume-wise coordinate system
    y               : float             # (px) center of volume in a volume-wise coordinate system
    z               : float             # (um) initial depth in the motor coordinate system
    px_height       : smallint          # lines per frame
    px_width        : smallint          # pixels per line
    px_depth        : smallint          # number of slices
    um_height       : float             # height in microns
    um_width        : float             # width in microns
    um_depth        : float             # depth in microns
    """
    @property
    def key_source(self):
        return Stitching.Volume() & {'pipe_version': CURRENT_VERSION}

    class Slice(dj.Part):
        definition = """ # single slice of one stack

        -> CorrectedStack
        -> shared.Channel
        islice              : smallint          # index of slice in volume
        ---
        slice               : longblob          # image (height x width)
        slice_z             : float             # slice depth in volume-wise coordinate system
        """

    def _make_tuples(self, key):
        print('Correcting stack', key)

        for channel in range((StackInfo() & key).fetch1('nchannels')):
            # Correct ROIs
            rois = []
            for roi_tuple in (StackInfo.ROI() * Stitching.ROICoordinates() & key).fetch():
                # Load ROI
                roi_filename = (experiment.Stack.Filename() & roi_tuple).local_filenames_as_wildcard
                roi = scanreader.read_scan(roi_filename)

                # Map: Apply corrections to each field in parallel
                f = performance.parallel_correct_stack # function to map
                raster_phase = (RasterCorrection() & roi_tuple).fetch1('raster_phase')
                fill_fraction = (StackInfo() & key).fetch1('fill_fraction')
                y_shifts, x_shifts = (MotionCorrection() & roi_tuple).fetch1('y_shifts', 'x_shifts')
                field_ids = roi_tuple['field_ids']
                results = performance.map_fields(f, roi, field_ids=field_ids, channel=channel,
                                                 kwargs={'raster_phase': raster_phase,
                                                         'fill_fraction': fill_fraction,
                                                         'y_shifts': y_shifts, 'x_shifts': x_shifts})

                # Reduce: Collect results
                corrected_roi = np.empty((roi_tuple['roi_px_depth'], roi_tuple['roi_px_height'],
                                          roi_tuple['roi_px_width']), dtype=np.float32)
                for field_idx, corrected_field in results:
                    corrected_roi[field_idx] = corrected_field

                # Create ROI object
                xs, ys = list(roi_tuple['stitch_xs']), list(roi_tuple['stitch_ys'])
                rois.append(stitching.StitchedROI(corrected_roi, x=xs, y=ys, z=roi_tuple['stitch_z'],
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
                            break # restart joining

                return sorted_rois

            # Stitch all rois together. This is convoluted because smooth blending in
            # join_with assumes rois are next to (not below or atop of) each other
            prev_num_rois = float('Inf') # to enter the loop at least once
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
            roi_info = StackInfo.ROI() & key & {'roi_id': stitched.roi_coordinates[0].id} # one roi from this volume
            tuple_ = {**key, 'x': stitched.x, 'y': stitched.y, 'z': stitched.z,
                      'px_height': stitched.height, 'px_width': stitched.width}
            tuple_['um_height'] = stitched.height * roi_info.microns_per_pixel[0]
            tuple_['um_width'] = stitched.width * roi_info.microns_per_pixel[1]
            tuple_['px_depth'] = roi_info.fetch1('roi_px_depth') # same as original rois
            tuple_['um_depth'] = roi_info.fetch1('roi_um_depth') # same as original rois
            self.insert1(tuple_, skip_duplicates=True)

            # Insert each slice
            initial_z = stitched.z
            z_step = (StackInfo() & key).fetch1('z_step')
            for i, slice_ in enumerate(stitched.volume):
                self.Slice().insert1({**key, 'channel': channel + 1, 'islice': i + 1,
                                      'slice': slice_, 'slice_z': initial_z + i * z_step})

            self.notify({**key, 'channel': channel + 1})

    @notify.ignore_exceptions
    def notify(self, key):
        import imageio

        volume = (self & key).get_stack(channel=key['channel'])
        volume = volume[:: int(volume.shape[0] / 8)] # volume at 8 diff depths
        video_filename = '/tmp/' + key_hash(key) + '.gif'
        imageio.mimsave(video_filename, float2uint8(volume), duration=1)

        msg = ('corrected stack for {animal_id}-{session}-{stack_idx} volume {volume_id} '
               'channel {channel}').format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=video_filename, file_title=msg, channel='#pipeline_quality')

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

        fig = plt.figure()
        fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
        im = fig.gca().imshow(stack[int(num_slices / 2)])
        video = animation.FuncAnimation(fig, lambda i: im.set_data(stack[i]), num_slices,
                                        interval=1000 / fps)
        fig.tight_layout()

        if not filename.endswith('.mp4'):
            filename += '.mp4'
        print('Saving video at:', filename)
        print('If this takes too long, stop it and call again with dpi <', dpi, '(default)')
        video.save(filename, dpi=dpi)

        return fig


@schema
class RegistrationTask(dj.Manual):
    definition = """ # declare scan fields to register to a stack as well as channels and method used

    (stack_session) -> CorrectedStack(session)  # animal_id, stack_session, stack_idx, pipe_version, volume_id
    (scan_session) -> experiment.Scan(session)  # animal_id, scan_session, scan_idx
    -> shared.Field
    (stack_channel) -> shared.Channel(channel)
    (scan_channel) -> shared.Channel(channel)
    -> shared.RegistrationMethod
    """
    def fill(self, stack_key, scan_key, stack_channel=1, scan_channel=1, method=3):
        # Add stack attributes
        stack_rel = CorrectedStack() & stack_key
        if len(stack_rel) > 1:
            raise PipelineException('More than one stack match stack_key {}'.format(stack_key))
        tuple_ = stack_rel.proj(stack_session='session').fetch1()

        # Add common attributes
        tuple_['stack_channel'] = stack_channel
        tuple_['scan_channel'] = scan_channel
        tuple_['registration_method'] = method

        # Add scan attributes
        fields_rel = reso.ScanInfo.Field().proj() + meso.ScanInfo.Field().proj() & scan_key
        scan_animal_ids = np.unique(fields_rel.fetch('animal_id'))
        if len(scan_animal_ids) > 1 or scan_animal_ids[0] != tuple_['animal_id']:
            raise PipelineException('animal_id of stack and scan do not match.')
        for field in fields_rel.fetch():
            RegistrationTask().insert1({**tuple_, 'scan_session': field['session'],
                                        'scan_idx': field['scan_idx'],
                                        'field': field['field']}, skip_duplicates=True)


@schema
class InitialRegistration(dj.Computed):
    definition = """ # register a 2-d scan field to a stack (rigid registration with 100 microns range)

    (stack_session) -> CorrectedStack(session)  # animal_id, stack_session, stack_idx, pipe_version, volume_id
    (scan_session) -> experiment.Scan(session)  # animal_id, scan_session, scan_idx
    -> shared.Field
    (stack_channel) -> shared.Channel(channel)
    (scan_channel) -> shared.Channel(channel)
    ---
    init_x          : float         # (px) center of scan in stack coordinates
    init_y          : float         # (px) center of scan in stack coordinates
    init_z          : float         # (um) depth of scan in stack coordinates
    score           : float         # cross-correlation score (-1 to 1)
    common_res      : float         # (um/px) common resolution used for registration
    """
    @property
    def key_source(self):
        all_stacks = (CorrectedStack() * shared.Channel()).proj(stack_session='session',
                                                                stack_channel='channel')
        all_fields = experiment.Scan() * shared.Field() * shared.Channel()
        processed_fields = reso.SummaryImages() + meso.SummaryImages()
        fields = (all_fields & processed_fields).proj(scan_session='session',
                                                      scan_channel='channel')
        keys = (all_stacks * fields) & RegistrationTask() & {'pipe_version': CURRENT_VERSION}

        return keys

    class FieldInStack(dj.Part):
        definition = """ # cut out of the field in the stack after registration
        -> master
        ---
        init_field  : longblob    # 2-d field taken from the stack
        """

    def _make_tuples(self, key):
        from scipy import ndimage
        from .utils import registration

        print('Registering', key)

        # Get stack
        stack_rel = (CorrectedStack() & key & {'session': key['stack_session']})
        stack = stack_rel.get_stack(key['stack_channel'])

        # Get average field
        field_key = {'animal_id': key['animal_id'], 'session': key['scan_session'],
                     'scan_idx': key['scan_idx'], 'field': key['field'],
                     'channel': key['scan_channel']} # no pipe_version
        pipe = reso if reso.ScanInfo() & field_key else meso if meso.ScanInfo() & field_key else None
        mean_image = (pipe.SummaryImages.Average() & field_key).fetch1('average_image')

        # Get field and stack resolution
        field_res = ((reso.ScanInfo() & field_key).microns_per_pixel if pipe == reso else
                     (meso.ScanInfo.Field() & field_key).microns_per_pixel)
        dims = stack_rel.fetch1('um_depth', 'px_depth', 'um_height', 'px_height',
                                'um_width', 'px_width')
        stack_res = np.array([dims[0] / dims[1], dims[2] / dims[3], dims[4] / dims[5]])

        # Drop some edges (only x and y) to avoid artifacts and black edges
        skip_dims = [max(1, int(round(s * 0.025))) for s in stack.shape]
        stack = stack[:, skip_dims[1] : -skip_dims[1], skip_dims[2]: -skip_dims[2]]
        skip_dims = [max(1, int(round(s * 0.025))) for s in mean_image.shape]
        field = mean_image[skip_dims[0] : -skip_dims[0], skip_dims[1]: -skip_dims[1]]

        # Apply local contrast normalization (improves contrast and gets rid of big vessels)
        norm_stack = enhancement.lcn(stack, np.array([3, 25, 25]) / stack_res)
        norm_field = enhancement.lcn(field, 20 / field_res)

        # Rescale to match lowest resolution  (isotropic pixels/voxels)
        common_res = max(*field_res, *stack_res) # minimum available resolution
        common_stack = ndimage.zoom(norm_stack, stack_res / common_res, order=1)
        common_field = ndimage.zoom(norm_field, field_res / common_res, order=1)

        # Get estimated depth of the field (from experimenters)
        stack_x, stack_y, stack_z = stack_rel.fetch1('x', 'y', 'z') # z of the first slice (zero is at surface depth)
        field_z = (pipe.ScanInfo.Field() & field_key).fetch1('z') # measured in microns (zero is at surface depth)
        if field_z < stack_z or field_z > stack_z + dims[0]:
            msg_template = 'Warning: Estimated depth ({}) outside stack range ({}-{}).'
            print(msg_template.format(field_z, stack_z , stack_z + dims[0]))
        estimated_px_z = (field_z - stack_z + 0.5) / common_res # in pixels

        # Run rigid registration with no rotations searching 100 microns up and down
        px_estimate = (0, 0, estimated_px_z - common_stack.shape[0] / 2) # (0, 0, 0) in center of stack
        px_range = (0.45 * common_stack.shape[2], 0.45 * common_stack.shape[1], 100 / common_res)
        result = registration.register_rigid(common_stack, common_field, px_estimate, px_range)
        score, (x, y, z), _ = result

        # Get field in stack (after registration)
        stack = ndimage.zoom(stack, stack_res / common_res, order=1)
        common_shape = np.round(np.array(mean_image.shape) * field_res / common_res).astype(int)
        reg_field = registration.find_field_in_stack(stack, *common_shape, x, y, z)
        reg_field = ndimage.zoom(reg_field, common_res / field_res, order=1) # *
        # * this could differ from original shape but it should be pretty close

        # Map back to stack coordinates
        final_x = stack_x + x * (common_res / stack_res[2]) # in stack pixels
        final_y = stack_y + y * (common_res / stack_res[1]) # in stack pixels
        final_z = stack_z + (z + common_stack.shape[0] / 2) * common_res # in microns*
        #* Best match in slice 0 will not result in z = 0 but 0.5 * z_step.

        # Insert
        self.insert1({**key,'init_x': final_x, 'init_y': final_y, 'init_z': final_z,
                      'score': score, 'common_res': common_res,})
        self.FieldInStack().insert1({**key, 'init_field': reg_field})

        self.notify(key, mean_image, reg_field)

    @notify.ignore_exceptions
    def notify(self, key, original_field, registered_field):
        import imageio
        from pipeline.utils import signal

        orig_clipped = np.clip(original_field, *np.percentile(original_field, [1, 99.8]))
        reg_clipped = np.clip(registered_field, *np.percentile(registered_field, [1, 99.8]))

        overlay = np.zeros([*original_field.shape, 3], dtype=np.uint8)
        overlay[:, :, 0] = signal.float2uint8(-reg_clipped) # stack in red
        overlay[:, :, 1] = signal.float2uint8(-orig_clipped) # original in green
        img_filename = '/tmp/{}.png'.format(key_hash(key))
        imageio.imwrite(img_filename, overlay)

        msg = ('initial registration of {animal_id}-{scan_session}-{scan_idx} field '
               '{field} to {animal_id}-{stack_session}-{stack_idx}').format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key &
                                           {'session': key['stack_session']})
        slack_user.notify(file=img_filename, file_title=msg)


@schema
class CuratedRegistration(dj.Computed):
    definition = """ # curate the initial registration estimates before final registration

    -> InitialRegistration
    -> shared.CurationMethod
    ---
    cur_x          : float         # (px) center of scan in stack coordinates
    cur_y          : float         # (px) center of scan in stack coordinates
    cur_z          : float         # (um) depth of scan in stack coordinates
    """
    def _make_tuples(self, key):
        if key['curation_method'] == 1:
            x, y, z = (InitialRegistration & key).fetch1('init_x', 'init_y', 'init_z')
            self.insert1({**key, 'cur_x': x, 'cur_y': y, 'cur_z': z})
        if key['curation_method'] == 2:
            print('Warning: Interface for manual curation written in Matlab.')


@schema
class FieldRegistration(dj.Computed):
    """
    Note: We stick with this conventions to define rotations:
    http://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html

    "To summarize, we will employ a Tait-Bryan Euler angle convention using active,
    intrinsic rotations around the axes in the order z-y-x". We use a right-handed
    coordinate system (x points to the right, y points forward and z points downwards)
    with right-handed/clockwise rotations.

    To register the field to the stack:
        1. Scale the field & stack to have isotropic pixels & voxels that match the
            lowest pixels-per-microns resolution among the two (common_res).
        2. Rotate the field over x -> y -> z (extrinsic roll->pitch->yaw is equivalent to
            an intrinsic yaw->pitch-roll rotation) using clockwise rotations and taking
            center of the field as (0, 0, 0).
        3. Translate to final x, y, z position (accounting for the previous stack scaling).
    """
    definition = """ # align a 2-d scan field to a stack

    -> CuratedRegistration
    -> RegistrationTask
    ---
    reg_x       : float         # (px) center of scan in stack coordinates
    reg_y       : float         # (px) center of scan in stack coordinates
    reg_z       : float         # (um) depth of scan in stack coordinates
    yaw=0       : float         # degrees of rotation over the z axis
    pitch=0     : float         # degrees of rotation over the y axis
    roll=0      : float         # degrees of rotation over the x axis
    score       : float         # cross-correlation score (-1 to 1)
    common_res  : float         # (um/px) common resolution used for registration
    """
    class FieldInStack(dj.Part):
        definition = """ # cut out of the field in the stack after registration
        -> master
        ---
        reg_field       : longblob    # 2-d field taken from the stack
        """

    class AffineResults(dj.Part):
        definition = """ # some intermediate results from affine registration
        -> master
        ---
        score_map       : longblob     # 3-d map of best correlation scores for each yaw, pitch, rol combination
        position_map    : longblob     # 3-d map of best positions (x, y, z) for each yaw, pitch, roll combination
        """

    def _make_tuples(self, key):
        from scipy import ndimage
        from .utils import registration

        print('Registering', key)

        # Get stack
        stack_rel = (CorrectedStack() & key & {'session': key['stack_session']})
        stack = stack_rel.get_stack(key['stack_channel'])

        # Get average field
        field_key = {'animal_id': key['animal_id'], 'session': key['scan_session'],
                     'scan_idx': key['scan_idx'], 'field': key['field'],
                     'channel': key['scan_channel']} # no pipe_version
        pipe = reso if reso.ScanInfo() & field_key else meso if meso.ScanInfo() & field_key else None
        mean_image = (pipe.SummaryImages.Average() & field_key).fetch1('average_image')

        # Get field and stack resolution
        field_res = ((reso.ScanInfo() & field_key).microns_per_pixel if pipe == reso else
                     (meso.ScanInfo.Field() & field_key).microns_per_pixel)
        dims = stack_rel.fetch1('um_depth', 'px_depth', 'um_height', 'px_height',
                                'um_width', 'px_width')
        stack_res = np.array([dims[0] / dims[1], dims[2] / dims[3], dims[4] / dims[5]])

        # Drop some edges (only x and y) to avoid artifacts and black edges
        skip_dims = [max(1, int(round(s * 0.025))) for s in stack.shape]
        stack = stack[:, skip_dims[1] : -skip_dims[1], skip_dims[2]: -skip_dims[2]]
        skip_dims = [max(1, int(round(s * 0.025))) for s in mean_image.shape]
        field = mean_image[skip_dims[0] : -skip_dims[0], skip_dims[1]: -skip_dims[1]]

        # Apply local contrast normalization (improves contrast and gets rid of big vessels)
        norm_stack = enhancement.lcn(stack, np.array([3, 25, 25]) / stack_res)
        norm_field = enhancement.lcn(field, 20 / field_res)

        # Rescale to match lowest resolution  (isotropic pixels/voxels)
        common_res = max(*field_res, *stack_res) # minimum available resolution
        common_stack = ndimage.zoom(norm_stack, stack_res / common_res, order=1)
        common_field = ndimage.zoom(norm_field, field_res / common_res, order=1)

        # Get initial estimate from CuratedRegistration
        stack_x, stack_y, stack_z = stack_rel.fetch1('x', 'y', 'z') # z of the first slice (zero is at surface depth)
        cur_x, cur_y, cur_z = (CuratedRegistration() & key).fetch1('cur_x', 'cur_y', 'cur_z')
        estimated_px_x = (cur_x - stack_x) * stack_res[2] / common_res # in pixels
        estimated_px_y = (cur_y - stack_y) * stack_res[1] / common_res
        estimated_px_z = (cur_z - stack_z) / common_res - common_stack.shape[0] / 2

        # Register
        px_estimate = (estimated_px_x, estimated_px_y, estimated_px_z) # (0, 0, 0) in center of stack
        um_range = 40 if key['registration_method'] in [1, 3] else 100 # search 40/100 microns up and down
        px_range = (um_range / common_res, ) * 3
        if key['registration_method'] in [1, 2]: # rigid
            # Run rigid registration with no rotations
            result = registration.register_rigid(common_stack, common_field, px_estimate, px_range)
            score, (x, y, z), (yaw, pitch, roll) = result

        elif key['registration_method'] in [3, 4]: # rigid plus 3-d rotation
            # Run parallel registration searching for best rotation angles
            angles = np.linspace(-4, 4, 4 * 4 + 1) # -4, -3.5, -3, ... 3.5, 4
            results = performance.map_angles(common_stack, common_field, px_estimate, px_range, angles)
            score, (x, y, z), (yaw, pitch, roll) = sorted(results)[-1]

            # Create some intermediate results (inserted below)
            score_map = np.zeros([len(angles), len(angles), len(angles)])
            position_map = np.zeros([len(angles), len(angles), len(angles), 3])
            for rho, position, res_angles in results:
                idx1, idx2, idx3 = (np.where(angles == a)[0][0] for a in res_angles)
                score_map[idx1, idx2, idx3] = rho
                position_map[idx1, idx2, idx3] = position

        # Get field in stack (after registration)
        stack = ndimage.zoom(stack, stack_res / common_res, order=1)
        common_shape = np.round(np.array(mean_image.shape) * field_res / common_res).astype(int)
        reg_field = registration.find_field_in_stack(stack, *common_shape, x, y, z, yaw,
                                                     pitch, roll)
        reg_field = ndimage.zoom(reg_field, common_res / field_res, order=1) # *
        # * this could differ from original shape but it should be pretty close

        # Map back to stack coordinates
        final_x = stack_x + x * (common_res / stack_res[2]) # in stack pixels
        final_y = stack_y + y * (common_res / stack_res[1]) # in stack pixels
        final_z = stack_z + (z + common_stack.shape[0] / 2) * common_res # in microns*
        #* Best match in slice 0 will not result in z = 0 but 0.5 * z_step.

        # Insert
        self.insert1({**key, 'reg_x': final_x, 'reg_y': final_y, 'reg_z': final_z,
                      'yaw': yaw, 'pitch': pitch, 'roll': roll, 'common_res': common_res,
                      'score': score})
        self.FieldInStack().insert1({**key, 'reg_field': reg_field})
        if key['registration_method'] in [3, 4]: # store correlation values
            self.AffineResults().insert1({**key, 'score_map': score_map,
                                          'position_map': position_map})

        self.notify(key, mean_image, reg_field)

    @notify.ignore_exceptions
    def notify(self, key, original_field, registered_field):
        import imageio
        from pipeline.utils import signal

        orig_clipped = np.clip(original_field, *np.percentile(original_field, [1, 99.8]))
        reg_clipped = np.clip(registered_field, *np.percentile(registered_field, [1, 99.8]))

        overlay = np.zeros([*original_field.shape, 3], dtype=np.uint8)
        overlay[:, :, 0] = signal.float2uint8(-reg_clipped) # stack in red
        overlay[:, :, 1] = signal.float2uint8(-orig_clipped) # original in green
        img_filename = '/tmp/{}.png'.format(key_hash(key))
        imageio.imwrite(img_filename, overlay)

        msg = ('registration of {animal_id}-{scan_session}-{scan_idx} field {field} to '
               '{animal_id}-{stack_session}-{stack_idx} (method {registration_method})')
        msg = msg.format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key &
                                           {'session': key['stack_session']})
        slack_user.notify(file=img_filename, file_title=msg)


@schema
class StackSet(dj.Computed):
    definition=""" # give a unique id to segmented masks in the stack
    (stack_session) -> CorrectedStack(session)  # animal_id, stack_session, stack_idx, pipe_version, volume_id
    -> shared.CurationMethod
    -> shared.RegistrationMethod
    ---
    min_distance            :tinyint        # distance used as threshold to accept two masks as the same
    max_height              :tinyint        # maximum allowed height of a joint mask
    """
    #TODO: Make it automatic to delete itself and repopulate if a new field is registered to the stack
    @property
    def key_source(self):
        all_keys = CorrectedStack() * shared.CurationMethod().proj() * shared.RegistrationMethod()
        return all_keys.proj(stack_session='session') & FieldRegistration()

    class Unit(dj.Part):
        definition = """ # a unit in the stack
        -> master
        munit_id            :int        # unique id in the stack
        ---
        munit_x             :float      # (px) position of the centroid in the stack
        munit_y             :float      # (px) position of the centroid in the stack
        munit_z             :float      # (um) position of the centroid in the stack
        """

    class Match(dj.Part):
        definition = """ # Scan unit to stack unit match (n:1 relation)
        -> master
        (scan_session) -> experiment.Scan(session)  # animal_id, scan_session, scan_idx
        -> shared.SegmentationMethod
        unit_id             :int        # unit id from ScanSet.Unit
        ---
        -> StackSet.Unit
        """

    class MatchedUnit():
        """ Coordinates for a set of cells."""
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

    def _make_tuples(self, key):
        from .utils.registration import create_rotation_matrix
        from scipy.spatial import distance
        import bisect

        # Set some params
        min_distance = 10
        max_height = 20

        # Compute stack resolution
        stack_rel = (CorrectedStack() & key & {'session': key['stack_session']})
        dims = stack_rel.fetch1('um_depth', 'px_depth', 'um_height', 'px_height',
                                'um_width', 'px_width')
        stack_res = np.array([dims[0] / dims[1], dims[2] / dims[3], dims[4] / dims[5]])

        # Create list of units
        units = [] # stands for matched units
        registered_fields = FieldRegistration() & key
        for field in registered_fields.fetch():
            # Get field_key, field_hash and field_res
            field_key = {'animal_id': field['animal_id'], 'session': field['scan_session'],
                         'scan_idx': field['scan_idx'], 'field': field['field'],
                         'channel': field['scan_channel']} # no pipe_version
            field_hash = key_hash(field_key)
            pipe = reso if reso.ScanInfo() & field_key else meso if meso.ScanInfo() & field_key else None
            field_res = ((reso.ScanInfo() & field_key).microns_per_pixel if pipe == reso
                         else (meso.ScanInfo.Field() & field_key).microns_per_pixel)

            # Create transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = create_rotation_matrix(field['yaw'], field['pitch'],
                                                              field['roll'])
            transform_matrix[0, 3] = field['reg_x'] * stack_res[2] # 1 x 1 resolution
            transform_matrix[1, 3] = field['reg_y'] * stack_res[1]
            transform_matrix[2, 3] = field['reg_z'] * stack_res[0]

            # Create cell objects
            somas = (pipe.MaskClassification.Type() & {'type': 'soma'})
            field_somas = pipe.ScanSet.Unit() & field_key & somas
            unit_keys, xs, ys = (pipe.ScanSet.UnitInfo() & field_somas).fetch('KEY', 'px_x', 'px_y')
            coords = [xs * field_res[1], ys * field_res[0], np.zeros(len(xs)), np.ones(len(xs))]
            xs, ys, zs, _ = np.dot(transform_matrix, coords)
            units += [StackSet.MatchedUnit(*args, field_hash) for args in zip(unit_keys, xs, ys, zs)]
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
            acceptable_height = (max(unit1.zs + unit2.zs) - min(unit1.zs + unit2.zs)) < max_height
            return different_fields and acceptable_height

        # Create distance matrix
        # For memory efficiency we use an adjacency list with only the units at less than 10 microns
        centroids = np.stack(u.centroid for u in units)
        distance_list = [] # list of triples (distance, unit1, unit2)
        for i in range(len(units)):
            indices, distances = find_close_units(centroids[i], centroids[i+1:], min_distance)
            for dist, j in zip(distances, i + 1 + indices):
                if is_valid(units[i], units[j], max_height):
                    bisect.insort(distance_list, (dist, units[i], units[j]))
        print(len(distance_list), 'possible pairings')

        # Join units
        while(len(distance_list) > 0):
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
            centroid = munit.centroid / stack_res[::-1] # in stack coordinates
            self.Unit().insert1({**key, 'munit_id': munit_id, 'munit_x': centroid[0],
                                 'munit_y': centroid[1], 'munit_z': centroid[2]})
            for subunit_key in munit.keys:
                new_match = {**key, 'munit_id': munit_id,
                             **subunit_key, 'scan_session': subunit_key['session']}
                self.Match().insert1(new_match, ignore_extra_fields=True)

        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        fig = (StackSet() & key).plot_centroids3d()
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)

        msg = ('StackSet for {animal_id}-{stack_session}-{stack_idx}: {num_units} final '
               'units').format(**key, num_units=len(self.Unit() & key))
        slack_user = notify.SlackUser() & (experiment.Session() & key &
                                           {'session': key['stack_session']})
        slack_user.notify(file=img_filename, file_title=msg)

    def plot_centroids3d(self):
        """ Plots the centroids of all units in the motor coordinate system (in microns)

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        from mpl_toolkits.mplot3d import Axes3D

        # Get stack resolution
        stack_rel = CorrectedStack() & self.proj(session='stack_session')
        dims = stack_rel.fetch1('um_height', 'px_height', 'um_width', 'px_width')
        stack_res = [dims[0] / dims[1], dims[2] / dims[3]]

        # Get centroids
        xs, ys, zs = (StackSet.Unit() & self).fetch('munit_x', 'munit_y', 'munit_z')
        centroids = np.stack([xs * stack_res[1], ys * stack_res[0], zs], axis=1)

        # Plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2])
        ax.invert_zaxis()
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.set_zlabel('z (um)')

        return fig



# TODO: Add this in shared
#class SegmentationMethod(dj.Lookup):
#    defintion = """ # 3-d segmentation methods
#    """
#    # threshold: Just threshold the scan and postprocess (nuclear labels)
#    # blob: Gaussian blob detection
#    # covnent: 3-d convnet
#
#@schema
#class SegmentationTask(dj.Manual):
#    definition = """ # defines the target of segmentation and the channel to use
#
#    -> experiment.Stack
#    -> shared.Channel
#    -> shared.SegmentationMethod
#    ---
#    -> experiment.Compartment
#    """
