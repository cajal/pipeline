""" Schemas for structural stacks. """
import datajoint as dj
from datajoint.jobs import key_hash
import matplotlib.pyplot as plt
import numpy as np
import scanreader
from scipy import signal
import itertools

from . import experiment, notify, shared
from .utils import galvo_corrections, stitching
from .utils.signal import mirrconv, float2uint8
from .exceptions import PipelineException


schema = dj.schema('pipeline_stack', locals())
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
class Corrections(dj.Computed):
    definition = """ # stack corrections
    -> StackInfo                    # animal_id, session, stack_idx, pipe_version
    -> CorrectionChannel            # animal_id, session, stack_idx
    """
    @property
    def key_source(self):
        return StackInfo() * CorrectionChannel() & {'pipe_version': CURRENT_VERSION}


    class Raster(dj.Part):
        definition = """ # raster correction for bidirectional resonant scans

        -> Corrections
        -> StackInfo.ROI                         # animal_id, session, stack_idx, roi_id, version
        ---
        raster_phase            : float          # difference between expected and recorded scan angle
        raster_std              : float          # standard deviation among raster phases in different slices
        """

        def _make_tuples(self, key, roi):
            """ Compute raster phase discarding top and bottom 10% of slices and tapering
            edges to avoid edge artifacts

            :param dict key: Dictionary with ROI key attributes.
            :param np.array roi: ROI (fields, image_height, image_width, frames).
            """
            if (StackInfo.ROI() & key).fetch1('bidirectional'):
                num_fields, image_height, image_width, num_frames = roi.shape

                # Compute some parameters
                fill_fraction = (StackInfo() & key).fetch1('fill_fraction')
                skip_fields = max(1, int(round(num_fields * 0.10)))
                taper = np.sqrt(np.outer(signal.tukey(image_height, 0.4),
                                         signal.tukey(image_width, 0.4)))

                # Compute raster phase for each slice and take the mean
                raster_phases = []
                for slice_ in roi[skip_fields: -skip_fields]:
                    # Create template (average frame tapered to avoid edge artifacts)
                    anscombed = 2 * np.sqrt(slice_ - slice_.min() + 3 / 8) # anscombe transform
                    template = np.mean(anscombed, axis=-1) * taper

                    # Compute raster correction
                    raster_phases.append(galvo_corrections.compute_raster_phase(template,
                                                                                fill_fraction))
                raster_phase = np.mean(raster_phases)
                raster_std = np.std(raster_phases)
            else:
                raster_phase = 0
                raster_std = 0

            # Insert
            self.insert1({**key, 'raster_phase': raster_phase, 'raster_std': raster_std})

        def correct(self, roi):
            """ Correct roi with parameters extracted from self.

            :param np.array roi: ROI (fields, image_height, image_width, frames).
            """
            raster_phase = self.fetch1('raster_phase')
            fill_fraction = (StackInfo() & self).fetch1('fill_fraction')
            if raster_phase == 0:
                corrected_roi = roi.astype(np.float32, copy=False)
            else:
                roi = roi.transpose([1, 2, 3, 0])
                corrected_roi = galvo_corrections.correct_raster(roi, raster_phase, fill_fraction)
                corrected_roi = corrected_roi.transpose([3, 0, 1, 2])
            return corrected_roi


    class Motion(dj.Part):
        definition = """ # motion correction for each slice in the stack (frame-to-frame and slice-to-slice)

        -> Corrections
        -> StackInfo.ROI
        ---
        y_shifts            : longblob      # y motion correction shifts (num_slices x num_frames)
        x_shifts            : longblob      # x motion correction shifts (num_slices x num_frames)
        y_aligns            : longblob      # isolated slice-to-slice alignment shifts (num_slices)
        x_aligns            : longblob      # isolated slice-to-slice alignment shifts (num_slices)
        """

        def _make_tuples(self, key, roi, sps):
            """ Compute motion shifts to align frames over time and over slices.

            :param dict key: Dictionary with ROI key attributes.
            :param np.array roi: Raster corrected ROI (fields, height, width, frames).
            :param float sps: Slices per second.
            """
            from scipy import ndimage

            # Discard some rows/cols to avoid edge artifacts
            num_fields, image_height, image_width, num_frames = roi.shape
            skip_rows = int(round(image_height * 0.10))
            skip_cols = int(round(image_width * 0.10))
            roi = roi[:, skip_rows: -skip_rows, skip_cols: -skip_cols, :]

            # Compute smoothing window size
            size_in_ms = 300  # smooth over a 300 milliseconds window
            window_size = int(round(sps * (size_in_ms / 1000)))  # in frames

            # Compute shifts
            y_shifts = np.zeros([num_fields, num_frames]) # frame to frame shifts
            x_shifts = np.zeros([num_fields, num_frames])
            y_aligns = np.zeros(num_fields) # slice to slice shifts
            x_aligns = np.zeros(num_fields)
            previous = None
            for i, field in enumerate(roi):
                corrected = field.copy() # leave input intact

                # Frame to frame alignment
                if num_frames > 1:
                    for j in range(3):
                        # Create template from previous
                        anscombed = 2 * np.sqrt(corrected - corrected.min() + 3 / 8) # anscombe transform
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
                        corrected = galvo_corrections.correct_motion(field, xy_shifts, in_place=False)

                # Interslice alignment
                if previous is not None:
                    # Align current slice to previous one
                    results = galvo_corrections.compute_motion_shifts(corrected.mean(axis=-1),
                        previous.mean(axis=-1), fix_outliers=False, smooth_shifts=False, in_place=False)

                    # Reject alignment shifts higher than 2.5% image height/width
                    y_align = results[0][0] if abs(results[0][0]) < image_height * 0.025 else 0
                    x_align = results[1][0] if abs(results[1][0]) < image_width * 0.025 else 0

                    # Update shifts (shift of i -1 plus the shift to align i to i-1)
                    y_aligns[i] = y_aligns[i - 1] + y_align
                    x_aligns[i] = x_aligns[i - 1] + x_align

                previous = corrected

            # Detrend alignment shifts (to decrease influence of vessels going through the slices)
            filter_size = int(round(40 / (StackInfo() & key).fetch1('z_step'))) # 40 microns in z
            smoothing_filter = signal.hann(filter_size + 1 if filter_size % 2 == 0 else 0)
            y_aligns -= mirrconv(y_aligns, smoothing_filter / sum(smoothing_filter))
            x_aligns -= mirrconv(x_aligns, smoothing_filter/ sum(smoothing_filter))

            # Compute final shifts (frame-to-frame plus slice-to-slice)
            y_shifts = y_shifts + np.expand_dims(y_aligns, axis=-1)
            x_shifts = x_shifts + np.expand_dims(x_aligns, axis=-1)

            # Insert
            self.insert1({**key, 'y_shifts': y_shifts, 'x_shifts': x_shifts,
                          'y_aligns': y_aligns, 'x_aligns': x_aligns})

        def correct(self, roi):
            """ Correct roi with parameters extracted from self.

                :param np.array roi: ROI (fields, image_height, image_width, frames).
            """
            y_shifts, x_shifts = self.fetch1('y_shifts', 'x_shifts')
            corrected = [galvo_corrections.correct_motion(field, (x_shifts[i], y_shifts[i]))
                         for i, field in enumerate(roi)]
            return np.stack(corrected)


    class Stitched(dj.Part):
        definition = """ # union of ROIs from a stack (usually one per stack)

        -> Corrections
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

        def _make_tuples(self, key, rois):
            # Stitch rois recursively
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
                        break # restart joining

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
                        break # restart joining
                [roi.rot270() for roi in rois]

            # Insert each stitched volume
            for volume_id, roi in enumerate(rois):
                tuple_ = {**key, 'volume_id': volume_id, 'x': roi.x, 'y': roi.y, 'z': roi.z,
                          'px_height': roi.height, 'px_width': roi.width}
                one_roi = StackInfo.ROI() & key & {'roi_id': roi.roi_coordinates[0].id} # get one roi from those forming the volume
                tuple_['px_depth'] = one_roi.fetch1('px_depth') # same as original rois
                tuple_['um_height'] = roi.height * one_roi.microns_per_pixel[0]
                tuple_['um_width'] = roi.width * one_roi.microns_per_pixel[1]
                tuple_['um_depth'] = one_roi.fetch1('um_depth') # same as original rois
                Corrections.Stitched().insert1(tuple_)

                # Insert coordinates of each ROI forming this volume
                for roi_coord in roi.roi_coordinates:
                    tuple_ = {**key, 'roi_id': roi_coord.id, 'volume_id': volume_id,
                              'x': roi_coord.x, 'y': roi_coord.y, 'z': roi_coord.z}
                    Corrections.ROICoordinates().insert1(tuple_)


    class ROICoordinates(dj.Part):
        definition = """ # coordinates for each ROI in the stitched volume

        -> Corrections
        -> StackInfo.ROI
        ---
        -> Corrections.Stitched
        x               : float             # (pixels) center of ROI in a volume-wise coordinate system
        y               : float             # (pixels) center of ROI in a volume-wise coordinate system
        z               : float             # (pixels) initial depth in a volume-wise coordinate system
        """


    def _make_tuples(self, key):
        # Insert in Corrections
        self.insert1(key)

        # Get some params
        correction_channel = (CorrectionChannel() & key).fetch1('channel') - 1
        num_slices = (StackInfo.ROI() & key).fetch('px_depth')
        skip_fields = int(round(min(num_slices) * 0.10)) # to avoid artifacts near the top and bottom

        # Compute raster and motion correction per ROI
        rois = []  # stores ROI objects used below for stitching
        for roi_tuple in (StackInfo.ROI() & key).fetch():
            roi_key = {**key, 'roi_id': roi_tuple['roi_id']}

            print('Computing corrections for ROI', roi_tuple['roi_id'])

            # Load ROI
            roi_filename = (experiment.Stack.Filename() & roi_tuple).local_filenames_as_wildcard
            roi = scanreader.read_stack(roi_filename)
            corrected_roi = roi[roi_tuple['field_ids'], :, :, correction_channel, :]
            corrected_roi = corrected_roi.astype(np.float32, copy=False)

            # Raster correction
            Corrections.Raster()._make_tuples(roi_key, corrected_roi)
            corrected_roi = (Corrections.Raster() & roi_key).correct(corrected_roi)

            # Motion correction
            sps = roi.fps * roi.num_scanning_depths # slices per second
            Corrections.Motion()._make_tuples(roi_key, corrected_roi, sps)
            corrected_roi = (Corrections.Motion() & roi_key).correct(corrected_roi)

            # Discard some fields at the top and bottom and some pixels to avoid artifacts
            skip_rows = max(1, int(round(0.005 * corrected_roi.shape[1])))  # 0.5 %
            skip_columns = max(1, int(round(0.005 * corrected_roi.shape[2])))  # 0.5 %
            corrected_roi = corrected_roi[skip_fields: -skip_fields, skip_rows: -skip_rows,
                                          skip_columns: -skip_columns]

            # Create ROI object
            rois.append(stitching.StitchedROI(corrected_roi.mean(axis=-1), x=roi_tuple['x'],
                                              y=roi_tuple['y'], z=roi_tuple['z'],
                                              id_=roi_tuple['roi_id']))

        # Stitch overlapping fields
        print('Computing stitching parameters')
        Corrections.Stitched()._make_tuples(key, rois)

        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        import seaborn as sns

        notifier = (notify.SlackUser() & (experiment.Session() & key))
        notifier.notify('Corrections for stack {} have been populated'.format(key))

        raster_phases = (self.Raster() & key).fetch('raster_phase')
        notifier.notify('Raster phases: {}'.format(raster_phases))

        num_rois = (StackInfo() & key).fetch1('nrois')
        with sns.axes_style('white'):
                fig, axes = plt.subplots(num_rois, 1, figsize=(13, 5 * num_rois), sharex=True, sharey=True)
        fig.suptitle('Shifts in y (blue) and x (red). Scatter dots for different timepoints.')
        axes = [axes] if num_rois == 1 else axes  # make list if single axis object
        for ax in axes.ravel():
            ax.set_ylabel('Pixels')
        axes[-1].set_xlabel('Depth')
        for i, roi_key in enumerate((StackInfo.ROI() & key).fetch.keys()):
            y_shifts, x_shifts = (self.Motion() & roi_key).fetch1('y_shifts', 'x_shifts')

            initial_z = (StackInfo.ROI() & roi_key).fetch1('z')
            z_step = (StackInfo() & key).fetch1('z_step')
            depths = initial_z + z_step * np.arange(len(y_shifts))

            axes[i].set_title('ROI {}'.format(roi_key['roi_id']))
            axes[i].plot(depths, y_shifts, '.b')
            axes[i].plot(depths, x_shifts, '.r')
            fig.tight_layout()
            img_filename = '/tmp/' + key_hash(key) + '.png'
            fig.savefig(img_filename)
            plt.close(fig)

        notifier.notify(file=img_filename, file_title='motion shifts')
        sns.reset_orig()

        for volume_key in (self.Stitched() & key).fetch.keys():
            msg = 'Volume {}:'.format(volume_key['volume_id'])
            for roi_coord in (self.ROICoordinates() & volume_key).fetch():
                    msg += ' ROI 1 at {}, {} (x, y);'.format(roi_coord['x'], roi_coord['y'])
            notifier.notify(msg)


@schema
class CorrectedStack(dj.Computed):
    definition = """ # all slices of each stack after corrections.

    -> Corrections.Stitched
    """
    @property
    def key_source(self):
        return Corrections.Stitched() & {'pipe_version': CURRENT_VERSION}


    class Slice(dj.Part):
        definition = """ # single slice of one stack
        -> CorrectedStack
        -> shared.Channel
        islice              : smallint          # index of slice in volume
        ---
        slice               : longblob          # image (height x width)
        z                   : float             # slice depth in volume-wise coordinate system
        """

    def _make_tuples(self, key):
        print('Correcting stack', key)

        # Insert in CorrectedStack
        self.insert1(key)

        for channel in range((StackInfo() & key).fetch1('nchannels')):
            # Correct each ROI
            rois = []
            for roi_coord in (Corrections.ROICoordinates() & key).fetch():
                roi_info = StackInfo.ROI() & (Corrections.ROICoordinates() & roi_coord).proj()

                # Load ROI
                roi_filename = (experiment.Stack.Filename() & roi_info).local_filenames_as_wildcard
                roi = scanreader.read_stack(roi_filename)
                corrected_roi = roi[roi_info.fetch1('field_ids'), :, :, channel, :]
                corrected_roi = corrected_roi.astype(np.float32, copy=False)

                # Raster correction
                corrected_roi = (Corrections.Raster() & roi_info).correct(corrected_roi)

                # Motion correction
                corrected_roi = (Corrections.Motion() & roi_info).correct(corrected_roi)

                # Create ROI object
                rois.append(stitching.StitchedROI(corrected_roi.mean(axis=-1), x=roi_coord['x'],
                                              y=roi_coord['y'], z=roi_coord['z'],
                                              id_=roi_coord['roi_id']))

#             # Stitch them
#            volume = rois[0]
#            for next_roi in rois[1:]:
#                volume.join_with(next_roi, next_roi.x, next_roi.y, smooth_blend=False)

            # Stitch all rois together (this is convoluted because smooth blending in
            # join_with assumes the second argument is to the right of the first)
            two_rois_joined = True
            while two_rois_joined:
                two_rois_joined = False

                # Join rows
                for roi1, roi2 in itertools.combinations(rois, 2):
                    if roi1.is_aside_to(roi2):
                        if roi1.left_or_right(roi2) == stitching.Position.LEFT: # 2|1
                            roi2.join_with(roi1, roi1.x, roi1.y)
                            rois.remove(roi1)
                        else: # 1|2
                            roi1.join_with(roi2, roi2.x, roi2.y)
                            rois.remove(roi2)

                        two_rois_joined=True
                        break # restart joining

                # Join columns
                [roi.rot90() for roi in rois]
                for roi1, roi2 in itertools.combinations(rois, 2):
                    if roi1.is_aside_to(roi2):
                        if roi1.left_or_right(roi2) == stitching.Position.LEFT: # 2|1
                            roi2.join_with(roi1, roi1.x, roi1.y)
                            rois.remove(roi1)
                        else: # 1|2
                            roi1.join_with(roi2, roi2.x, roi2.y)
                            rois.remove(roi2)

                        two_rois_joined=True
                        break # restart joining
                [roi.rot270() for roi in rois]

            if len(rois) != 1:
                msg = 'ROIs for volume {} could not be stitched properly'.format(key)
                raise PipelineException(msg)
            volume = rois[0].volume
            # volume.trim() # delete black spaces in edges

            # Insert each slice
            initial_z = (Corrections.Stitched() & key).fetch1('z')
            z_step = (StackInfo() & key).fetch1('z_step')
            for i, slice_ in enumerate(volume):
                self.Slice().insert1({**key, 'channel': channel + 1, 'islice': i,
                                      'slice': slice_, 'z': initial_z + i * z_step})

            self.notify({**key, 'channel': channel + 1}, volume)

    def notify(self, key, volume):
        import imageio

        video_filename = '/tmp/' + key_hash(key) + '.gif'
        stitched_roi = volume.volume[:: round(volume.depth / 5)] # volume at 5 diff depths
        imageio.mimsave(video_filename, float2uint8(stitched_roi), duration=1)

        msg = 'CorrectedStack for {} has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg, file=video_filename,
                                                                   file_title='stitched ROI')


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