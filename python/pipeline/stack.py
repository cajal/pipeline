""" Schemas for structural stacks. """
import datajoint as dj
from datajoint.jobs import key_hash
import matplotlib.pyplot as plt
import numpy as np
import scanreader
from scipy import signal

from . import experiment, notify, shared
from .utils import galvo_corrections
from .utils import stitching
from .utils.signal import mirrconv, float2uint8


schema = dj.schema('pipeline_stack', locals())
CURRENT_VERSION = 0


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
        field_ids       : blob              # list of field_ids (0-index) sorted from shallower to deeper
        x               : float             # (um) center of ROI in the motor coordinate system
        y               : float             # (um) center of ROI in the motor coordinate system
        z               : float             # (um) initial depth in the motor coordinate system
        px_height       : smallint          # lines per frame
        px_width        : smallint          # pixels per line
        px_depth        : smallint          # number of slices
        um_height       : float             # height in microns
        um_width        : float             # width in microns
        um_depth        : float             # depth in microns
        nframes         : smallint          # number of recorded frames per plane
        fps             : float             # (Hz) volumes per second
        bidirectional   : boolean           # true = bidirectional scanning
        """

        def _make_tuples(self, key, stack, id_in_file):
            # Create results tuple
            tuple_ = key.copy()

            # Get field_ids in this ROI ordered from shallower to deeper
            if stack.is_multiROI:
                field_ids = [i for i, field_rois in enumerate(stack.field_rois) if id_in_file in field_rois]
                field_depths = [stack.field_depths[i] for i in field_ids]
                field_ids = [i for _, i in sorted(zip(field_depths, field_ids))]
            else: # for reso lower values mean deeper
                field_ids = range(stack.num_scanning_depths)
                field_depths = stack.field_depths
                field_ids = [idx for _, idx in sorted(zip(field_depths, field_ids), reverse=True)]
            tuple_['field_ids'] = field_ids

            # Get reso/meso specific coordinates
            x_zero, y_zero, z_zero = stack.motor_position_at_zero  # motor x, y, z at ScanImage's 0
            if stack.is_multiROI:
                tuple_['x'] = x_zero + stack._degrees_to_microns(stack.fields[field_ids[0]].x)
                tuple_['y'] = y_zero + stack._degrees_to_microns(stack.fields[field_ids[0]].y)
                tuple_['z'] = z_zero + stack.field_depths[field_ids[0]]
                tuple_['px_height'] = stack.field_heights[field_ids[0]]
                tuple_['px_width'] = stack.field_widths[field_ids[0]]
                tuple_['um_height'] = stack.field_heights_in_microns[field_ids[0]]
                tuple_['um_width'] = stack.field_widths_in_microns[field_ids[0]]
                tuple_['um_depth'] = stack.field_depths[field_ids[-1]] - stack.field_depths[field_ids[0]] + 1
            else:
                tuple_['x'] = x_zero
                tuple_['y'] = y_zero
                tuple_['z'] = -(z_zero + stack.field_depths[field_ids[0]]) # minus so deeper is more positive
                tuple_['px_height'] = stack.image_height
                tuple_['px_width'] = stack.image_width

                # Estimate height and width in microns using measured FOVs for similar setups
                fov_rel = (experiment.FOV() * experiment.Session() * experiment.Stack() & key
                           & 'session_date>=fov_ts')
                zooms = fov_rel.fetch('mag').astype(np.float32)  # zooms measured in same setup
                closest_zoom = zooms[np.argmin(np.abs(np.log(zooms / stack.zoom)))]
                dims = (fov_rel & 'ABS(mag - {}) < 1e-4'.format(closest_zoom)).fetch1('height', 'width')
                um_height, um_width = [float(um) * (closest_zoom / stack.zoom) for um in dims]
                tuple_['um_height'] = um_height * stack._y_angle_scale_factor
                tuple_['um_width'] = um_width * stack._x_angle_scale_factor
                tuple_['um_depth'] = stack.field_depths[field_ids[0]] - stack.field_depths[field_ids[-1]] + 1

            # Get common parameters
            tuple_['px_depth'] = len(field_ids)
            tuple_['nframes'] = stack.num_frames
            tuple_['fps'] = stack.fps
            tuple_['bidirectional'] = stack.is_bidirectional

            self.insert1(tuple_)

        @property
        def microns_per_pixel(self):
            """ Returns an array with microns per pixel in height and width. """
            um_height, px_height, um_width, px_width = self.fetch1('um_height', 'px_height',
                                                                   'um_width', 'px_width')
            return np.array([um_height / px_height, um_width / px_width])

    def _make_tuples(self, key):
        """ Read and store stack information."""
        print('Reading header...')

        # Read files forming this stack
        filename_keys = (experiment.Stack.Filename() & key).fetch(dj.key)
        stacks = []
        for filename_key in filename_keys:
            stack_filename = (experiment.Stack.Filename() & filename_key).local_filenames_as_wildcard
            stacks.append(scanreader.read_stack(stack_filename))
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
            for id_in_file in range(num_rois):
                roi_key = {**key, **filename_key, 'roi_id': roi_id}
                StackInfo.ROI()._make_tuples(roi_key, stack, id_in_file)
                roi_id += 1

        # Fill in CorrectionChannel if only one channel
        if stacks[0].num_channels == 1:
            CorrectionChannel().fill_in(key)

        self.notify(key)

    def notify(self, key):
        msg = 'StackInfo for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg)


@schema
class CorrectionChannel(dj.Manual):
    definition = """ # channel to use for raster and motion correction

    -> experiment.Stack
    ---
    -> shared.Channel
    """

    def fill_in(self, key, channel=1):
        for stack_key in (StackInfo() & key).fetch(dj.key):
            self.insert1({**stack_key, 'channel': channel}, ignore_extra_fields=True,
                          skip_duplicates=True)


@schema
class RasterCorrection(dj.Computed):
    definition = """ # raster correction for bidirectional resonant scans

    -> StackInfo.ROI                        # animal_id, session, stack_idx, roi_id, version
    -> CorrectionChannel                    # animal_id, session, stack_idx
    ---
    raster_phase           : float          # difference between expected and recorded scan angle
    raster_std             : float          # standard deviation among raster phases in different slices
    """

    @property
    def key_source(self):
        return StackInfo.ROI() * CorrectionChannel() & {'pipe_version': CURRENT_VERSION}

    def _make_tuples(self, key):
        print('Computing raster correction for ROI', key)

        # Get some params
        res = (StackInfo.ROI() & key).fetch1('bidirectional', 'px_height', 'px_width', 'field_ids')
        is_bidirectional, image_height, image_width, field_ids = res
        correction_channel = (CorrectionChannel() & key).fetch1('channel') - 1

        if is_bidirectional:
            # Read the ROI
            filename_rel = (experiment.Stack.Filename() & (StackInfo.ROI() & key))
            roi_filename = filename_rel.local_filenames_as_wildcard
            roi = scanreader.read_stack(roi_filename, dtype=np.float32)

            # Drop 10% of slices in the top and bottom and taper to avoid edge artifacts
            skip_fields = max(1, int(round(len(field_ids) * 0.10)))
            taper = np.sqrt(np.outer(signal.tukey(image_height, 0.4),
                                     signal.tukey(image_width, 0.4)))

            # Compute raster phase for each slice and take the mean
            raster_phases = []
            for field_id in field_ids[skip_fields:-skip_fields]:
                # Create template (average frame tapered to avoid edge artifacts)
                slice_= roi[field_id, :, :, correction_channel, :]
                anscombed = 2 * np.sqrt(slice_ - slice_.min() + 3 / 8) # anscombe transform
                template = np.mean(anscombed, axis=-1) * taper

                # Compute raster correction
                raster_phases.append(galvo_corrections.compute_raster_phase(template,
                                                             roi.temporal_fill_fraction))
            raster_phase = np.mean(raster_phases)
            raster_std = np.std(raster_phases)
            print('Raster phases per slice: ', raster_phases)
        else:
            raster_phase = 0
            raster_std = 0

        # Insert
        self.insert1({**key, 'raster_phase': raster_phase, 'raster_std': raster_std})

        self.notify(key)

    def notify(self, key):
        msg = 'RasterCorrection for `{}` has been populated.'.format(key)
        msg += '\nRaster phase: {}'.format((self & key).fetch1('raster_phase'))
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg)

    def get_correct_raster(self):
        """ Returns a function to perform raster correction on the 4-d ROI. """
        raster_phase = self.fetch1('raster_phase')
        fill_fraction = (StackInfo() & self).fetch1('fill_fraction')
        if raster_phase == 0:
            correct_raster = lambda roi: roi.astype(np.float32, copy=False)
        else:
            def correct_raster(roi):
                roi = roi.transpose([1, 2, 3, 0])
                corrected_roi = galvo_corrections.correct_raster(roi, raster_phase, fill_fraction)
                return corrected_roi.transpose([3, 0, 1, 2])
        return correct_raster


@schema
class MotionCorrection(dj.Computed):
    definition = """ # motion correction for each slice in the stack

    -> RasterCorrection
    ---
    y_shifts                        : longblob      # (pixels) y motion correction shifts (num_slices x num_frames)
    x_shifts                        : longblob      # (pixels) x motion correction shifts (num_slices x num_frames)
    """

    @property
    def key_source(self):
        return RasterCorrection() & {'pipe_version': CURRENT_VERSION}

    def _make_tuples(self, key):
        from scipy import ndimage

        print('Computing motion correction for ROI', key)

        # Get some params
        res = (StackInfo.ROI() & key).fetch1('nframes', 'px_height', 'px_width', 'field_ids')
        num_frames, image_height, image_width, field_ids = res
        correction_channel = (CorrectionChannel() & key).fetch1('channel') - 1

        y_shifts = np.zeros([len(field_ids), num_frames])
        x_shifts = np.zeros([len(field_ids), num_frames])
        if num_frames > 1:
            # Read the ROI
            filename_rel = (experiment.Stack.Filename() & (StackInfo.ROI() & key))
            roi_filename = filename_rel.local_filenames_as_wildcard
            roi = scanreader.read_stack(roi_filename, dtype=np.float32)
            roi_ = roi[field_ids, :, :, correction_channel, :]

            # Correct raster effects
            correct_raster = (RasterCorrection() & key).get_correct_raster()
            corrected_roi = correct_raster(roi_)

            # Discard some rows/cols to avoid edge artifacts
            skip_rows = int(round(image_height * 0.10))
            skip_cols = int(round(image_width * 0.10))
            roi_ = corrected_roi[:, skip_rows: -skip_rows, skip_cols: -skip_cols, :]
            roi_ -= roi_.min() # make nonnegative for fft

            # Compute smoothing window size
            size_in_ms = 300  # smooth over a 300 milliseconds window
            slices_per_second = roi.fps * roi.num_scanning_depths
            window_size = int(round(slices_per_second * (size_in_ms / 1000)))  # in frames

            for i, field in enumerate(roi_): # height x width x frames
                shifted = field.copy()
                for j in range(3):
                    # Create template
                    anscombed = 2 * np.sqrt(shifted + 3 / 8) # anscombe transform
                    template = np.mean(anscombed, axis=-1)
                    template = ndimage.gaussian_filter(template, 0.6)

                    # Compute motion correction shifts
                    results = galvo_corrections.compute_motion_shifts(field, template,
                        in_place=False, fix_outliers=False, smoothing_window_size=window_size)

                    # Center motions around zero
                    y_shifts[i] = results[0] - results[0].mean()
                    x_shifts[i] = results[1] - results[1].mean()

                    # Apply shifts
                    xy_shifts = np.stack([x_shifts[i], y_shifts[i]])
                    shifted = galvo_corrections.correct_motion(field, xy_shifts, in_place=False)

        # Insert
        self.insert1({**key, 'y_shifts': y_shifts, 'x_shifts': x_shifts})

        self.notify(key, y_shifts, x_shifts)

    def notify(self, key, y_shifts, x_shifts):
        import seaborn as sns

        num_slices, num_frames = y_shifts.shape
        fps = (StackInfo.ROI() & key).fetch1('fps') * num_slices
        seconds = np.arange(num_frames) / fps

        with sns.axes_style('white'):
            fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, sharey=True)

        axes[0].set_title('Shifts in y for all slices')
        axes[0].plot(seconds, y_shifts.T)
        axes[1].set_title('Shifts in x for all slices')
        axes[1].plot(seconds, x_shifts.T)
        axes[0].set_ylabel('Pixels')
        axes[1].set_ylabel('Pixels')
        axes[1].set_xlabel('Seconds')
        fig.tight_layout()
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)
        sns.reset_orig()

        msg = 'MotionCorrection for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg, file=img_filename,
                                                                   file_title='motion shifts')

    def get_correct_motion(self):
        """ Returns a function to perform motion correction in the 4-d ROI."""
        y_shifts, x_shifts = self.fetch1('y_shifts', 'x_shifts')
        def correct_motion(roi):
            corrected = []
            for i, field in enumerate(roi):
                xy_shifts = [x_shifts[i], y_shifts[i]]
                corrected.append(galvo_corrections.correct_motion(field, xy_shifts))
            return np.stack(corrected)
        return correct_motion

@schema
class Alignment(dj.Computed):
    definition = """ # inter-slice alignment
    -> MotionCorrection
    ---
    y_shifts                : longblob
    x_shifts                : longblob
    """

    @property
    def key_source(self):
        return MotionCorrection() & {'pipe_version': CURRENT_VERSION}

    def _make_tuples(self, key):
        print('Computing alignment for ROI', key)

        # Get some params
        res = (StackInfo.ROI() & key).fetch1('px_height', 'px_width', 'field_ids')
        image_height, image_width, field_ids = res
        correction_channel = (CorrectionChannel() & key).fetch1('channel') - 1
        num_slices = len(field_ids)

        # Read ROI
        filename_rel = (experiment.Stack.Filename() & (StackInfo.ROI() & key))
        roi_filename = filename_rel.local_filenames_as_wildcard
        roi = scanreader.read_stack(roi_filename, dtype=np.float32)
        roi_ = roi[field_ids, :, :, correction_channel, :]

        # Correct raster and motion effects
        correct_raster = (RasterCorrection() & key).get_correct_raster()
        correct_motion = (MotionCorrection() & key).get_correct_motion()
        corrected_roi = correct_motion(correct_raster(roi_)).mean(axis=-1)

        # Discard some rows/cols to avoid edge artifacts
        skip_rows = int(round(image_height * 0.10))
        skip_cols = int(round(image_width * 0.10))
        roi_ = corrected_roi[:, skip_rows: -skip_rows, skip_cols: -skip_cols]

#        # Increase contrast
#        enhanced_roi = np.empty(roi_.shape, dtype=np.float32)
#        for i, slice_ in enumerate(roi_):
#            a_min, a_max = np.percentile(slice_, (25, 75))
#            enhanced_roi[i] = np.clip(slice_, a_min=a_min, a_max=a_max)
#        roi_ = enhanced_roi

        # Compute shifts per slice
        y_shifts = np.zeros(num_slices)
        x_shifts = np.zeros(num_slices)
        for i in range(1, num_slices):
            results = galvo_corrections.compute_motion_shifts(roi_[i], roi_[i - 1],
                                fix_outliers=False, smooth_shifts=False, in_place=False)

            print('y, x shifts', results[0][0], results[1][0])

            # Reject shifts that are higher than 2.5% image height/width
            y_shift = results[0][0] if abs(results[0][0]) < image_height * 0.025 else 0
            x_shift = results[1][0] if abs(results[1][0]) < image_width * 0.025 else 0

            # Update shifts (shift of i -1 plus the shift to align i to i-1)
            y_shifts[i] = y_shifts[i - 1] + y_shift
            x_shifts[i] = x_shifts[i - 1] + x_shift

        # Detrend (to discard influence of vessels going through the slices)
        filter_size = int(round(40 / (StackInfo() & key).fetch1('z_step'))) # 40 microns in z
        smoothing_filter = signal.hann(filter_size + 1 if filter_size % 2 == 0 else 0)
        y_shifts -= mirrconv(y_shifts, smoothing_filter / sum(smoothing_filter))
        x_shifts -= mirrconv(x_shifts, smoothing_filter/ sum(smoothing_filter))

        # Center shifts
        y_shifts -= y_shifts.mean()
        x_shifts -= x_shifts.mean()

        # Insert
        self.insert1({**key, 'y_shifts': y_shifts, 'x_shifts': x_shifts})

        self.notify(key, y_shifts, x_shifts)

    def notify(self, key, y_shifts, x_shifts):
        import seaborn as sns

        initial_z = (StackInfo.ROI() & key).fetch1('z')
        z_step = (StackInfo() & key).fetch1('z_step')
        depths = initial_z + z_step * np.arange(len(y_shifts))

        with sns.axes_style('white'):
            fig = plt.figure(figsize=(15, 8))

        plt.plot(depths, y_shifts, label='Y shifts')
        plt.plot(depths, x_shifts, label='X shifts')
        plt.ylabel('Pixels')
        plt.xlabel('Depth')
        plt.legend()
        fig.tight_layout()
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)
        sns.reset_orig()

        msg = 'Alignment for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg, file=img_filename,
                                                                   file_title='alignment')

    def get_align(self):
        """ Returns a function to align slices in the stack. """
        y_shifts, x_shifts = self.fetch1('y_shifts', 'x_shifts')
        correct_alignment = lambda roi: np.stack([galvo_corrections.correct_motion(slice_, (x, y))
                                                  for slice_, x, y in zip(roi, x_shifts, y_shifts)])
        return correct_alignment

@schema
class Stitching(dj.Computed):
    definition = """ # stitches together overlapping rois
    -> StackInfo
    """
    @property
    def key_source(self):
        # run iff all ROIs have been processed
        stacks = StackInfo() - (StackInfo.ROI() - Alignment())
        return stacks & {'pipe_version': CURRENT_VERSION}

    class Volume(dj.Part):
        definition = """ # union of ROIs from a stack (usually one volume per stack)
        -> Stitching
        volume_id       : tinyint           # id of this volume
        ---
        x               : float             # (um) center of ROI in a volume-wise coordinate system
        y               : float             # (um) center of ROI in a volume-wise coordinate system
        z               : float             # (um) initial depth in a volume-wise coordinate system
        px_height       : smallint          # lines per frame
        px_width        : smallint          # pixels per line
        px_depth        : smallint          # number of slices
        um_height       : float             # height in microns
        um_width        : float             # width in microns
        um_depth        : float             # depth in microns
        """

#        def get_stitch(self):
#            """ Returns a function that stitches all ROIS of this volume. """
#            y_shifts, x_shifts = self.fetch1('y_shifts', 'x_shifts')
#
#            # What would be the ionput t this function
#            return lambda roi: np.stack([galvo_corrections.correct_motion(f, [x_shifts[i], y_shifts[i]])
#                                 for i, f in enumerate(roi)]):

    class ROICoordinates(dj.Part):
        definition = """ # coordinates for each ROI in a volume
        -> StackInfo.ROI
        ---
        -> Stitching.Volume
        x               : float             # (pixels) center of ROI in a volume-wise coordinate system
        y               : float             # (pixels) center of ROI in a volume-wise coordinate system
        z               : float             # (pixels) initial depth in a volume-wise coordinate system
        """

    def _make_tuples(self, key):
        import itertools

        print('Stitching ROIS for stack', key)

        # Get some params
        correction_channel = (CorrectionChannel() & key).fetch1('channel') - 1
        num_slices = (StackInfo.ROI() & key).fetch('px_depth')
        skip_fields = int(round(min(num_slices) * 0.10)) # to avoid artifacts near the top and bottom

        # Read ROIs forming this stack
        rois = []
        for roi_tuple in (StackInfo.ROI() & key).fetch():
            # Load ROI
            roi_filename = (experiment.Stack.Filename() & roi_tuple).local_filenames_as_wildcard
            roi = scanreader.read_stack(roi_filename, dtype=np.float32)
            roi_ = roi[roi_tuple['field_ids'], :, :, correction_channel, :]

            # Correct
            correct_raster = (RasterCorrection() & roi_tuple).get_correct_raster()
            correct_motion = (MotionCorrection() & roi_tuple).get_correct_motion()
            align = (Alignment() & roi_tuple).get_align()
            corrected_roi = align(correct_motion(correct_raster(roi_)).mean(axis=-1))

            # Discard some fields at the top and bottom and some pixels to avoid artifacts
            skip_rows = max(1, int(round(0.005 * corrected_roi.shape[1])))  # 0.5 %
            skip_columns = max(1, int(round(0.005 * corrected_roi.shape[2])))  # 0.5 %
            corrected_roi = corrected_roi[skip_fields: -skip_fields, skip_rows: -skip_rows,
                                          skip_columns: -skip_columns]

            # Create ROI object
            rois.append(stitching.StitchedROI(corrected_roi, x=roi_tuple['x'],
                                              y=roi_tuple['y'], z=roi_tuple['z'],
                                              id_=roi_tuple['roi_id']))


        two_rois_joined = True
        while two_rois_joined:
            two_rois_joined = False

            # Join rows
            for roi1, roi2 in itertools.combinations(rois, 2):
                if roi1.is_aside_to(roi2):
                    if roi1.left_or_right(roi2) == stitching.Position.LEFT: # 2|1
                        left, right = roi2, roi1
                    else: # 1|2
                        left, right = roi1, roi2

                    # Compute translation, join them and update roi list
                    delta_x, delta_y = stitching.linear_stitch(left.volume, right.volume)
                    left.join_with(right, left.x + delta_x, left.y + delta_y)
                    rois.remove(right)

                    two_rois_joined=True
                    break; # restart joining

            # Join columns
            [roi.rot90() for roi in rois]
            for roi1, roi2 in itertools.combinations(rois, 2):
                if roi1.is_aside_to(roi2):
                    if roi1.left_or_right(roi2) == stitching.Position.LEFT: # 2|1
                        left, right = roi2, roi1
                    else: # 1|2
                        left, right = roi1, roi2

                    # Compute translation, join them and update roi list
                    delta_x, delta_y = stitching.linear_stitch(left.volume, right.volume)
                    left.join_with(right, left.x + delta_x, left.y + delta_y)
                    rois.remove(right)

                    two_rois_joined=True
                    break; # restart joining
            [roi.rot270() for roi in rois]

        # Trim rois to delete black spaces in edges
        #[roi.trim() for roi in rois]

        # Insert in Stitching
        self.insert1(key)

        # Insert each stitched volume
        for volume_id, roi in enumerate(rois):
            tuple_ = {**key, 'volume_id': volume_id, 'x': roi.x, 'y': roi.y, 'z': roi.z,
                      'px_height': roi.height, 'px_width': roi.width}
            one_roi = StackInfo.ROI() & key & {'roi_id': roi.roi_coordinates[0].id} # get one roi from those forming the volume
            tuple_['px_depth'] = one_roi.fetch1('px_depth') # same as original rois
            tuple_['um_height'] = roi.height * one_roi.microns_per_pixel[0]
            tuple_['um_width'] = roi.width * one_roi.microns_per_pixel[1]
            tuple_['um_depth'] = one_roi.fetch1('um_depth') # same as original rois
            Stitching.Volume().insert1(tuple_)

            # Insert coordinates of each ROI forming this volume
            for roi_coord in roi.roi_coordinates:
                tuple_ = {**key, 'roi_id': roi_coord.id, 'volume_id': volume_id,
                          'x': roi_coord.x, 'y': roi_coord.y, 'z': roi_coord.z}
                Stitching.ROICoordinates().insert1(tuple_)


        # Save stitched volumes (datajoint)
        # TODO: Save here
        for i, roi in enumerate(rois):
            np.save('/data/pipeline/volume_{}.npy'.format(i), roi.volume)
            pass

        self.notify(key, rois)

    def notify(self, key, rois):
        import imageio

        msg = 'Stitching for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg)

        # Send a gif with the stitched roi
        for roi in rois:
            video_filename = '/tmp/' + key_hash(key) + '.gif'
            stitched_roi = roi.volume[:: round(roi.depth / 5)] # volume at 5 diff depths
            imageio.mimsave(video_filename, float2uint8(stitched_roi), duration=1)

            (notify.SlackUser() & (experiment.Session() & key)).notify(file=video_filename,
                                                                file_title='stitched ROI')

#class CorrectedStack to just correct all stackc

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
