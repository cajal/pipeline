""" Schemas specific to platinum mice. Not the cleanest design/code"""
import datajoint as dj
import numpy as np
from scipy import ndimage

from . import reso, meso, stack, notify, shared, experiment
from .utils import enhancement, registration
from datajoint.jobs import key_hash


schema = dj.schema('pipeline_fastmeso', locals(), create_tables=True)
dj.config['external-fastmeso'] = {'protocol': 'file',
                                  'location': '/mnt/scratch07/pipeline-externals'}
dj.config['cache'] = '/tmp/dj-cache'

@schema
class PreprocessedStack(dj.Computed):
    definition = """ # stack after preprocessing (and cached in memory) for RegistrationOverTime
    (stack_session) -> stack.CorrectedStack(session)  # animal_id, stack_session, stack_idx, pipe_version, volume_id
    (stack_channel) -> shared.Channel(channel)
    ---
    stack :     external-fastmeso   # motion corrected uint16
    """
    @property
    def key_source(self):
        keys = stack.CorrectedStack.proj(stack_session='session') * shared.Channel().proj(stack_channel='channel')
        return keys & {'stack_channel': 1}

    def _make_tuples(self, key):
        # Get stack
        stack_rel = (stack.CorrectedStack() & key & {'session': key['stack_session']})
        stack_ = stack_rel.get_stack(key['stack_channel'])

        dims = stack_rel.fetch1('um_depth', 'px_depth', 'um_height', 'px_height',
                                'um_width', 'px_width')
        stack_res = np.array([dims[0] / dims[1], dims[2] / dims[3], dims[4] / dims[5]])

        skip_dims = [max(1, int(round(s * 0.025))) for s in stack_.shape]
        stack_ = stack_[:, skip_dims[1] : -skip_dims[1], skip_dims[2]: -skip_dims[2]]

        stack_ = enhancement.lcn(stack_, np.array([3, 25, 25]) / stack_res)

        self.insert1({**key, 'stack': stack_})


#TODO: Drop, replacement in stack.RegistrationOverTime
@schema
class RegistrationOverTime(dj.Computed):
    """ Simplified version of stack.FieldRegistration, see original for details"""
    definition = """ # align a 2-d scan field to a stack
    (stack_session) -> stack.CorrectedStack(session)  # animal_id, stack_session, stack_idx, pipe_version, volume_id
    (scan_session) -> experiment.Scan(session)  # animal_id, scan_session, scan_idx
    -> shared.Field
    (stack_channel) -> shared.Channel(channel)
    (scan_channel) -> shared.Channel(channel)
    frame_id    : int           # number of the frame (1-16)
    ---
    reg_x       : float         # (px) center of scan in stack coordinates
    reg_y       : float         # (px) center of scan in stack coordinates
    reg_z       : float         # (um) depth of scan in stack coordinates
    score       : float         # cross-correlation score (-1 to 1)
    common_res  : float         # (um/px) common resolution used for registration
    """
    @property
    def key_source(self):
        keys = (stack.CorrectedStack.proj(stack_session='session') *
                experiment.Scan().proj(scan_session='session') * shared.Field().proj() *
                shared.Channel().proj(stack_channel='channel') *
                shared.Channel().proj(scan_channel='channel'))
        return keys & stack.RegistrationTask() & meso.Quality.SummaryFrames().proj(scan_session='session')

    def _make_tuples(self, key):
        print('Registering', key)

        # Get stack
        stack_rel = (stack.CorrectedStack() & key & {'session': key['stack_session']})
        stack_ = (PreprocessedStack & key).fetch1('stack')

        # Get average field
        field_key = {'animal_id': key['animal_id'], 'session': key['scan_session'],
                     'scan_idx': key['scan_idx'], 'field': key['field'],
                     'channel': key['scan_channel']} #no pipe_version
        frames = (meso.Quality().SummaryFrames() & field_key).fetch1('summary')
        frames = np.moveaxis(frames, -1, 0)

        # Get field and stack resolution
        field_res = (meso.ScanInfo.Field() & field_key).microns_per_pixel
        dims = stack_rel.fetch1('um_depth', 'px_depth', 'um_height', 'px_height',
                                'um_width', 'px_width')
        stack_res = np.array([dims[0] / dims[1], dims[2] / dims[3], dims[4] / dims[5]])

        # Drop some edges (only y and x) to avoid artifacts
        skip_dims = [max(1, int(round(s * 0.025))) for s in frames.shape]
        frames = frames[:, skip_dims[1] : -skip_dims[1], skip_dims[2]: -skip_dims[2]]

        # Apply local contrast normalization (improves contrast and gets rid of big vessels)
        frames = np.stack(enhancement.lcn(f, 20 / field_res) for f in frames)

        # Rescale to match lowest resolution  (isotropic pixels/voxels)
        common_res = max(*field_res, *stack_res) # minimum available resolution
        stack_ = ndimage.zoom(stack_, stack_res / common_res, order=1)
        frames = np.stack(ndimage.zoom(f, field_res / common_res, order=1) for f in frames)

        # Get estimated depth of the field (from experimenters)
        stack_x, stack_y, stack_z = stack_rel.fetch1('x', 'y', 'z') # z of the first slice (zero is at surface depth)
        field_z = (meso.ScanInfo.Field() & field_key).fetch1('z') # measured in microns (zero is at surface depth)
        if field_z < stack_z or field_z > stack_z + dims[0]:
            msg_template = 'Warning: Estimated depth ({}) outside stack range ({}-{}).'
            print(msg_template.format(field_z, stack_z , stack_z + dims[0]))
        estimated_px_z = (field_z - stack_z + 0.5) / common_res # in pixels

        # Register
        px_estimate = (0, 0, estimated_px_z - stack_.shape[0] / 2) # (0, 0, 0) in center of stack
        px_range = (0.45 * stack_.shape[2], 0.45 * stack_.shape[1], 100 / common_res)
        for i, field in enumerate(frames):
            # Run rigid registration with no rotations
            score, (x, y, z), _ = registration.register_rigid(stack_, field, px_estimate, px_range)

            # Map back to stack coordinates
            final_x = stack_x + x * (common_res / stack_res[2]) # in stack pixels
            final_y = stack_y + y * (common_res / stack_res[1]) # in stack pixels
            final_z = stack_z + (z + stack_.shape[0] / 2) * common_res # in microns*
            #* Best match in slice 0 will not result in z = 0 but 0.5 * z_step.

            # Insert
            self.insert1({**key, 'frame_id': i + 1, 'common_res': common_res, 'reg_x': final_x,
                          'reg_y': final_y, 'reg_z': final_z, 'score': score})

#TODO: Drop, replaced as a method of stack.RegistrationOverTime
# Call once after all sessions are done
@schema
class RegistrationPlot(dj.Computed):
    definition = """ # align a 2-d scan field to a stack
    (stack_session) -> stack.CorrectedStack(session)  # animal_id, stack_session, stack_idx, pipe_version, volume_id
    (scan_session) -> experiment.Session(session)  # animal_id, scan_session, scan_idx
    (stack_channel) -> shared.Channel(channel)
    (scan_channel) -> shared.Channel(channel)
    """
    @property
    def key_source(self):
        keys = (stack.CorrectedStack.proj(stack_session='session') *
                experiment.Session().proj(scan_session='session') *
                shared.Channel().proj(stack_channel='channel') *
                shared.Channel().proj(scan_channel='channel'))
        return keys & RegistrationOverTime()

    def _make_tuples(self, key):
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        # Get all field keys and timestamps
        field_key, field_ts, field_nframes, field_fps = ((experiment.Scan() * meso.ScanInfo() *
            meso.ScanInfo.Field().proj()).proj('nframes', 'fps', 'scan_ts', scan_session='session')
            & key & 'field < 10').fetch('KEY', 'scan_ts', 'nframes', 'fps', order_by='scan_ts')
        if len(field_key) == 0:
            print('Warning: No fields selected for', key)
            return
        initial_time = str(field_ts[0])
        field_ts = [(ts - field_ts[0]).seconds for ts in field_ts] # in hours
        field_duration = field_nframes / field_fps

        # Plot
        fig = plt.figure(figsize=(20, 8))
        for fk, ft, fd in zip(field_key, field_ts, field_duration):
            zs = (RegistrationOverTime() & key & fk).fetch('reg_z', order_by='frame_id')
            ts = ft + np.linspace(0, 1, len(zs) + 2)[1:-1] * fd
            plt.plot(ts / 3600, zs)
        plt.title('Registered zs for {animal_id}-{scan_session} starting {t}'.format(t=initial_time, **key))
        plt.ylabel('Registered zs')
        plt.xlabel('Hours')

        # Plot formatting
        plt.gca().invert_yaxis()
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))
        plt.grid(linestyle='--', alpha=0.8)

        # Notify
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename, bbox_inches='tight')
        plt.close(fig)

        msg = 'registration over time for {animal_id}-{scan_session} into {animal_id}-{stack_session}-{stack_idx}'.format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key & {'session': key['stack_session']})
        slack_user.notify(file=img_filename, file_title=msg, channel='#pipeline_quality')
        self.insert1(key, ignore_extra_fields=True)


@schema
class SegmentationFromStack(dj.Computed):
    definition = """ # find the respective segmentation from the stack for a registered field
    -> stack.RegistrationOverTime.Chunk
    ---
    common_res              : float                     # common resolution stack and field were downsampled to
    stack_field             : blob                      # field (image x height) of cell ids at common_res resolution
    caiman_field            : blob                      # field created from caiman maks at common_res resolution
    """
    @property
    def key_source(self):
        return (stack.RegistrationOverTime.Chunk() &
                stack.Segmentation().proj(stack_session='session'))

    class StackUnit(dj.Part):
        definition = """
        -> master
        sunit_id                : int                  # id in the stack segmentation
        ---
        depth                   : int                  # (um) size in z   
        height                  : int                  # (um) size in y
        width                   : int                  # (um) size in x
        volume                  : float                # (um) volume of the 3-d unit
        area                    : float                # (um) area of the 2-d mask  
        sunit_z                 : float                # (um) centroid in z for the 3d unit
        sunit_y                 : float                # (um) centroid in y for the 3d unit
        sunit_x                 : float                # (um) centroid in x for the 3d unit
        mask_z                  : float                # (um) centroid in z for the 2d mask
        mask_y                  : float                # (um) centroid in y for the 2d mask
        mask_x                  : float                # (um) centroid in x for the 2d mask
        distance                : float                # (um) euclidean distance between centroid of 2-d mask and 3-d unit
        """

    class CaimanMask(dj.Part):
        definition = """ # CNMF mask corresponding to sunit_id (if any overlap)
        -> SegmentationFromStack.StackUnit
        ---
        caiman_id               : int                  # mask id from 2-d caiman segmentation
        caiman_iou              : float                # iou between the 2-d stack and caiman mask
        caiman_z                : float                # (um) centroid in z
        caiman_y                : float                # (um) centroid in y
        caiman_x                : float                # (um) centroid in x
        distance                : float                # (um) distance in the 2-d plane between caiman and 2-d mask
        """

    def _make_tuples(self, key):
        from skimage import measure

        print('Field segmentation: ', key)

        # Get instance segmentation
        instance = (stack.Segmentation() & key & {'session': key['stack_session']}).get_stack()

        # Get masks and binarize them (same way as used for plotting)
        masks = (meso.Segmentation() & key & {'session': key['scan_session']}).get_all_masks()
        masks = np.moveaxis(masks, -1, 0) # num_masks x height x width

        # Get field and stack resolution
        field_res = (meso.ScanInfo.Field() & key & {'session': key['scan_session']}).microns_per_pixel
        dims = (stack.CorrectedStack() & key & {'session': key['stack_session']}).fetch1(
            'um_depth', 'px_depth', 'um_height', 'px_height', 'um_width', 'px_width')
        stack_res = np.array([dims[0] / dims[1], dims[2] / dims[3], dims[4] / dims[5]])

        # Rescale to match lowest resolution  (isotropic pixels/voxels)
        common_res = min(*field_res, *stack_res[1:])  # maximum available resolution ignoring stack z resolution
        instance = ndimage.zoom(instance, stack_res / common_res, order=0)
        masks = np.stack(ndimage.zoom(f, field_res / common_res, order=1) for f in masks)
        # TODO: I could dilate em here to make em more round

        # Binarize masks
        binary_masks = np.zeros(masks.shape, dtype=bool)
        for i, mask in enumerate(masks):
            ## Compute cumulative mass (similar to caiman)
            indices = np.unravel_index(np.flip(np.argsort(mask, axis=None), axis=0),
                                       mask.shape)  # max to min value in mask
            cumsum_mask = np.cumsum(mask[indices] ** 2) / np.sum(mask ** 2)
            binary_masks[i][indices] = cumsum_mask < 0.9

        # Compute z, y, x of field as distances to the center of the stack
        reg_x, reg_y, reg_z = (stack.RegistrationOverTime.Chunk() & key).fetch1('regot_x',
                                                                                'regot_y',
                                                                                'regot_z')
        orig_x, orig_y, orig_z = (stack.CorrectedStack() & key &
                                  {'session': key['stack_session']}).fetch1('x', 'y', 'z')
        z = (reg_z - orig_z) / common_res - instance.shape[0] / 2
        y = (reg_y - orig_y) * stack_res[1] / common_res
        x = (reg_x - orig_x) * stack_res[2] / common_res

        # Get field segmentation from the stack segmentation
        field_height, field_width = binary_masks.shape[1:]
        reg_segmentation = registration.find_field_in_stack(instance, field_height,
                                                            field_width, x, y, z,
                                                            order=0).astype(np.int32)

        # Insert
        caiman_field = np.argmax(binary_masks, axis=0)
        self.insert1({**key, 'common_res': common_res, 'stack_field': reg_segmentation,
                      'caiman_field': caiman_field})

        instance_props =  measure.regionprops(instance)
        instance_labels = np.array([p.label for p in instance_props])
        for prop in measure.regionprops(reg_segmentation):
            sunit_id = prop.label
            instance_prop = instance_props[np.argmax(instance_labels == sunit_id)]

            depth = (instance_prop.bbox[3] - instance_prop.bbox[0]) * common_res
            height = (instance_prop.bbox[4] - instance_prop.bbox[1]) * common_res
            width = (instance_prop.bbox[5] - instance_prop.bbox[2]) * common_res
            volume = instance_prop.area * common_res ** 3
            sunit_z, sunit_y, sunit_x = [x * common_res for x in instance_prop.centroid]

            binary_sunit = reg_segmentation == sunit_id
            area = np.count_nonzero(binary_sunit) * common_res ** 2
            mask_z = (z + instance.shape[0] / 2) * common_res
            mask_y, mask_x = ndimage.measurements.center_of_mass(binary_sunit)
            mask_y = (mask_y + y + instance.shape[1] / 2 - field_height/2) * common_res
            mask_x = (mask_x + x + instance.shape[2] / 2 - field_width/2) * common_res
            #TODO: This won't hold when there's yaw pitch roll in the registration.

            distance = np.sqrt((sunit_z - mask_z) ** 2 + (sunit_y - mask_y) ** 2 +
                               (sunit_x - mask_x) ** 2)

            # Insert in StackUnit
            self.StackUnit().insert1({**key, 'sunit_id': sunit_id, 'depth': depth,
                                      'height': height, 'width': width, 'volume': volume,
                                      'area': area, 'sunit_z': sunit_z,
                                      'sunit_y': sunit_y, 'sunit_x': sunit_x,
                                      'mask_z': mask_z, 'mask_y': mask_y,
                                      'mask_x': mask_x, 'distance': distance})

            # Find closest caiman mask
            intersection = np.logical_and(binary_masks, binary_sunit).sum(axis=(1, 2)) # num_masks
            union = np.logical_or(binary_masks, binary_sunit).sum(axis=(1, 2)) # num_masks
            ious = intersection / union
            if np.any(ious > 0.2):
                caiman_id = np.argmax(ious) + 1
                caiman_iou = ious[caiman_id - 1]
                caiman_z = (z + instance.shape[0] / 2) * common_res
                caiman_y, caiman_x = ndimage.measurements.center_of_mass(binary_masks[caiman_id -1])
                caiman_y = (caiman_y + y + instance.shape[1] / 2 - field_height / 2) * common_res
                caiman_x = (caiman_x + x + instance.shape[2] / 2 - field_width / 2) * common_res

                distance = np.sqrt((caiman_y - mask_y) ** 2 + (caiman_x - mask_x) ** 2)

                self.CaimanMask().insert1({**key, 'sunit_id': sunit_id,
                                           'caiman_id': caiman_id,
                                           'caiman_iou': caiman_iou, 'caiman_z': caiman_z,
                                           'caiman_y': caiman_y, 'caiman_x': caiman_x,
                                           'distance': distance})


@schema
class ChunkWiseMethod(dj.Lookup):
    definition = """ # params for the chunkwise method
    chunk_method        :int
    ---
    duration            :int
    pad                 :int
    """
    contents = [[1, 10, 2], [2, 20, 4]]


@schema
class ChunkWiseSegmentation(dj.Computed):
    definition = """ # segment each 10 minute chunk of a scan by itself (using CNMF)
    -> meso.MotionCorrection
    -> meso.SegmentationTask
    -> ChunkWiseMethod
    ---
    chunk_ts=CURRENT_TIMESTAMP          :timestamp                 # timestamp 
    """

    class Chunk(dj.Part):
        definition = """ # results for one 10 min chunk
        -> master
        chunk_id            :int            # number of chunk
        ---
        initial_frame       :int            # initial frame in this chunk (1-based)
        final_frame         :int            # final frame in this chunk (1-based)
        background_mask     :longblob          # mask of the background component
        background_trace    :longblob          # trace of the background component
        avg_chunk           :longblob          # average of all frames in this chunk      
        """

        def get_all_masks(self):
            chunk_rel = (ChunkWiseSegmentation.Mask() & self)
            res = chunk_rel.fetch('indices_y', 'indices_x', 'weights', order_by='mask_id')

            height, width = (meso.ScanInfo.Field() & self).fetch1('px_height', 'px_width')
            masks = np.zeros((len(res[0]), height, width), dtype=np.float32)
            for i, (indices_y, indices_x, weights) in enumerate(zip(*res)):
                masks[i][indices_y, indices_x] = weights

            return masks


    class Mask(dj.Part):
        definition = """ # a single mask in one chunk
        -> ChunkWiseSegmentation.Chunk
        mask_id             :int            # mask id
        ---
        indices_y       : longblob      # indices in y for the mask
        indices_x       : longblob      # indices in x for the mask
        weights         : longblob      # weights of the mask at the indices above
        trace           : longblob      # raw fluorescence trace for this mask
        """

    def make(self, key):
        from .utils import caiman_interface as cmn
        import uuid
        import os

        print('')
        print('*' * 85)
        print('Processing', key, 'in chunks')

        # Get some parameters
        image_height, image_width = (meso.ScanInfo.Field() & key).fetch1('px_height',
                                                                         'px_width')
        fps, num_frames = (meso.ScanInfo() & key).fetch1('fps', 'nframes')
        um_per_px = np.array((meso.ScanInfo.Field() & key).microns_per_pixel)

        # Read corrected scan
        print('Reading scan...')
        scan = stack.RegistrationOverTime._get_corrected_scan(key)

        # Find best chunk size (~ 10 minutes, last chunk may be slightly smaller than rest)
        p, d = (ChunkWiseMethod() & key).fetch1('pad', 'duration')
        overlap = int(round(p * 60 * fps)) # ~ 2 minutes
        num_chunks = int(np.ceil((num_frames - overlap) / (d * 60 * fps - overlap)))
        chunk_size = int(np.ceil((num_frames - overlap) / num_chunks + overlap)) # *
        # * distributes frames in the last (incomplete) chunk to the other chunks

        # Segment with CNMF: iterate over chunks
        print('Registering', num_chunks, 'chunk(s)')
        caiman_kwargs = {'init_on_patches': True, 'proportion_patch_overlap': 0.2,
                         'num_components_per_patch': 6, 'init_method': 'greedy_roi',
                         'patch_size': tuple(50 / um_per_px) ,
                         'soma_diameter': tuple(8 / um_per_px),
                         'num_processes': 8, 'num_pixels_per_process': 10000}
        self.insert1(key)
        for i, initial_frame in enumerate(range(0, num_frames, chunk_size - overlap)):
            # Get next chunk
            final_frame = min(initial_frame + chunk_size, num_frames)
            chunk = scan[..., initial_frame: final_frame]

            # Create memory mapped file (as expected by CaImAn)
            print('Creating memory mapped file...')
            filename = '/tmp/caiman-{}_d1_{}_d2_{}_d3_1_order_C_frames_{}_.mmap'.format(
                uuid.uuid4(), image_height, image_width, chunk.shape[-1])
            mmap_shape = (image_height * image_width, chunk.shape[-1])
            mmap_scan = np.memmap(filename, mode='w+', shape=mmap_shape, dtype=np.float32)
            mmap_scan[:] = chunk.reshape(mmap_shape, order='F') - scan.min()
            mmap_scan.flush()

            # Extract traces
            print('Extracting masks and traces (cnmf)...')
            cnmf_result = cmn.extract_masks(chunk, mmap_scan, **caiman_kwargs)
            (masks, traces, background_masks, background_traces, raw_traces) = cnmf_result

            # Delete memory mapped scan
            print('Deleting memory mapped scan...')
            os.remove(mmap_scan.filename)

            # Insert results
            self.Chunk.insert1({**key, 'chunk_id': i + 1,
                                'initial_frame': initial_frame + 1,
                                'final_frame': final_frame,
                                'background_mask': background_masks[..., 0],
                                'background_trace': background_traces[0],
                                'avg_chunk': chunk.mean(-1)})

            ## Insert masks and traces
            raw_traces = raw_traces.astype(np.float32, copy=False)
            masks = np.moveaxis(masks.astype(np.float32, copy=False), -1, 0)
            for mask_id, (mask, trace) in enumerate(zip(masks, raw_traces), start=1):
                mask_indices = np.where(mask)
                self.Mask().insert1({**key, 'chunk_id': i + 1, 'mask_id': mask_id,
                                     'indices_y': mask_indices[0],
                                     'indices_x': mask_indices[1],
                                     'weights': mask[mask_indices], 'trace': trace})

# Utility class to join masks
class JointMask():
    def __init__(self, chunk_id, mask_id, mask, trace, initial_frame, final_frame):
        self.chunk_ids = [chunk_id]
        self.mask_ids = [mask_id]
        self._masks = [mask]
        self._traces = [trace]
        self._slices = [slice(initial_frame - 1, final_frame)]
        self._binary_mask = binarize_mask(mask) # current mask in binary form

    @property
    def initial_frame(self):
        return min([sl.start for sl in self._slices])

    @property
    def final_frame(self):
        return max([sl.stop for sl in self._slices])

    @property
    def avg_mask(self):
        return np.mean(self._masks, 0)

    @property
    def trace(self):
        from scipy import signal

        final_trace = np.zeros(self.final_frame - self.initial_frame, dtype=np.float32)
        slices, traces = tuple(zip(*sorted(zip(self._slices, self._traces))))
        slices = [slice(sl.start - self.initial_frame, sl.stop - self.initial_frame) for
                  sl in slices] # shift left if mask starts at initial_frame > 0

        # Fill trace
        final_trace[slices[0]] = traces[0]
        for prev_slice, new_slice, prev_trace, new_trace in zip(slices[:-1], slices[1:],
                                                                traces[:-1], traces[1:]):
            final_trace[new_slice] = new_trace

            # Blend overlap
            overlap = prev_slice.stop - new_slice.start
            taper = signal.hann(2 * overlap)[:overlap]
            final_trace[new_slice.start: prev_slice.stop] = (taper * new_trace[:overlap] +
                                                             (1 - taper) * prev_trace[-overlap:])

        return final_trace

    def iou_with(self, other):
        intersection = np.count_nonzero(np.logical_and(self._binary_mask,
                                                       other._binary_mask))
        if intersection == 0: # for efficiency
            return 0
        union = np.count_nonzero(np.logical_or(self._binary_mask, other._binary_mask))
        return intersection / union

    def join_with(self, other):
        """ The second trace is oldest one"""
        self.chunk_ids.extend(other.chunk_ids)
        self.mask_ids.extend(other.mask_ids)
        self._masks.extend(other._masks)
        self._traces.extend(other._traces)
        self._slices.extend(other._slices)

        if other.final_frame > self.final_frame:
            self._binary_mask = other._binary_mask


@schema
class JointChunks(dj.Computed):
    definition = """ # join segmentations generated in chunks
    -> ChunkWiseSegmentation
    ---
    background_mask         : longblob    # average background mask 
    background_trace        : longblob    # background trace through entire scan
    """

    class Mask(dj.Part):
        definition = """ # average mask and joined trace (nans
        -> master
        joint_id        :int            # joint mask id
        ---
        initial_frame   : int           # frame where this mask starts (1-based)
        final_frame     : int           # frame where this mask ends (1-based)
        indices_y       : longblob      # indices in y for the mask
        indices_x       : longblob      # indices in x for the mask
        weights         : longblob      # weights of the mask at the indices above
        trace           : longblob      # raw fluorescence trace for this mask
        """


    class MaskParts(dj.Part):
        definition = """ # parts forming this joint mask
        -> JointChunks.Mask
        -> ChunkWiseSegmentation.Mask
        ---
        iou=NULL                     : float             # IOU between joint mask and chunkwise mask
        """

    def _make_tuples(self, key):
        print('Joining', key)

        chunk_ids = sorted((ChunkWiseSegmentation.Chunk & key).fetch('chunk_id'))
        final_masks = [] # masks that cannot be extended anymore will go here
        edge_masks = [] # masks that can still be extended

        # Initialize edge masks with masks in the first chunk
        chunk_id = chunk_ids[0]
        masks = (ChunkWiseSegmentation.Chunk & key & {'chunk_id': chunk_id}).get_all_masks()
        initial_frame, final_frame = (ChunkWiseSegmentation.Chunk & key &
                                      {'chunk_id': chunk_id}).fetch1('initial_frame',
                                                                     'final_frame')
        traces = (ChunkWiseSegmentation.Mask & key & {'chunk_id': chunk_id}).fetch(
            'trace', order_by='mask_id')
        for mask_id, (mask, trace) in enumerate(zip(masks, traces), start=1):
            new_mask = JointMask(chunk_id, mask_id, mask, trace, initial_frame,
                                 final_frame)
            edge_masks.append(new_mask)

        for chunk_id in chunk_ids[1:]:
            masks = (ChunkWiseSegmentation.Chunk & key & {'chunk_id': chunk_id}).get_all_masks()
            initial_frame, final_frame = (ChunkWiseSegmentation.Chunk & key &
                                          {'chunk_id': chunk_id}).fetch1('initial_frame',
                                                                         'final_frame')
            traces = (ChunkWiseSegmentation.Mask & key & {'chunk_id': chunk_id}).fetch(
                'trace', order_by='mask_id')

            new_edges = []
            for mask_id, (mask, trace) in enumerate(zip(masks, traces), start=1):
                new_mask = JointMask(chunk_id, mask_id, mask, trace, initial_frame,
                                     final_frame)

                ious = np.array([m.iou_with(new_mask) for m in edge_masks])
                if np.any(ious > 0.4): # TODO: Select this threshold better
                    best_match = np.argmax(ious)

                    # Join with highest IOU
                    new_mask.join_with(edge_masks[best_match])

                    # Delete from edge (so it doesn't match with again)
                    del edge_masks[best_match]

                # Add to the new edge list
                new_edges.append(new_mask)

            # Add any previous edge that was not used to final masks and update
            final_masks.extend(edge_masks)
            edge_masks = new_edges
        final_masks.extend(edge_masks) # when there is no more chunks to add

        # Get background
        background_masks, background_traces, initial_frames, final_frames = \
            (ChunkWiseSegmentation.Chunk() & key).fetch('background_mask',
                                                        'background_trace',
                                                        'initial_frame', 'final_frame')
        background_mask = JointMask(0, 0, background_masks[0], background_traces[0],
                                    initial_frames[0], final_frames[0])
        for bm, bt, if_, ff in zip(background_masks[1:], background_traces[1:],
                                   initial_frames[1:], final_frames[1:]):
            new_mask = JointMask(0, 0, bm, bt, if_, ff)
            background_mask.join_with(new_mask)

        # Insert
        self.insert1({**key, 'background_trace': background_mask.trace,
                      'background_mask': background_mask.avg_mask})

        ## Insert masks and traces
        print('Inserting')
        for joint_id, joint_mask in enumerate(final_masks, start=1):
            mask_indices = np.where(joint_mask.avg_mask)
            self.Mask().insert1({**key, 'joint_id': joint_id,
                                 'initial_frame': joint_mask.initial_frame,
                                 'final_frame': joint_mask.final_frame,
                                 'indices_y': mask_indices[0],
                                 'indices_x': mask_indices[1],
                                 'weights': joint_mask.avg_mask[mask_indices],
                                 'trace': joint_mask.trace})
            for chunk_id, mask_id in zip(joint_mask.chunk_ids, joint_mask.mask_ids):
                self.MaskParts().insert1({**key, 'joint_id': joint_id,
                                          'chunk_id': chunk_id, 'mask_id': mask_id})

    def get_all_masks(self):
        chunk_rel = (JointChunks.Mask() & self.proj())
        res = chunk_rel.fetch('indices_y', 'indices_x', 'weights', order_by='joint_id')

        height, width = (meso.ScanInfo.Field() & self).fetch1('px_height', 'px_width')
        masks = np.zeros((len(res[0]), height, width), dtype=np.float32)
        for i, (indices_y, indices_x, weights) in enumerate(zip(*res)):
            masks[i][indices_y, indices_x] = weights

        return masks


def binarize_mask(mask, thresh=0.85):
    binary_mask = np.zeros(mask.shape, dtype=bool)

    # Compute cumulative mass (similar to caiman)
    indices = np.unravel_index(np.flip(np.argsort(mask, axis=None), axis=0),
                               mask.shape)  # max to min value in mask
    cumsum_mask = np.cumsum(mask[indices] ** 2) / np.sum(mask ** 2)
    binary_mask[indices] = cumsum_mask < thresh

    return binary_mask