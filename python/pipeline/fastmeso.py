""" Schemas specific to platinum mice. Not the cleanest design/code"""
import datajoint as dj
import numpy as np
from scipy import ndimage

from . import meso, stack


schema = dj.schema('pipeline_fastmeso', locals(), create_tables=True)


@schema
class SegmentationFromStack(dj.Computed):
    definition = """ # find the respective segmentation from the stack for a registered field
    
    -> stack.RegistrationOverTime.Chunk
    ---
    segm_field         : longblob      # field (image x height) of cell ids at 1 um/px
    caiman_field       : longblob      # field created from caiman maks at 1 um/px
    """

    @property
    def key_source(self):
        return stack.RegistrationOverTime.Chunk & stack.Segmentation.proj(
            stack_session='session')

    class StackUnit(dj.Part):
        definition = """

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

    class CaimanMask(dj.Part):
        definition = """ # CNMF mask corresponding to sunit_id (if any overlap)
        -> SegmentationFromStack.StackUnit
        ---
        caiman_id       : int           # mask id from 2-d caiman segmentation
        caiman_iou      : float         # iou between the 2-d stack mask and caiman mask
        caiman_area     : float         # (um) area of the 2-d caiman mask 
        caiman_z        : float         # (um) centroid in motor coordinate system
        caiman_y        : float         # (um) centroid in motor coordinate system
        caiman_x        : float         # (um) centroid in motor coordinate system
        distance        : float         # (um) distance in the 2-d plane between caiman and 2-d mask
        """

    def _make_tuples(self, key):
        from skimage import measure

        print('Finding segmentation for:', key)

        # Get instance segmentation
        stack_key = {'animal_id': key['animal_id'], 'session': key['stack_session'],
                     'stack_idx': key['stack_idx'], 'volume_id': key['volume_id']}
        instance = (stack.Segmentation & stack_key).fetch1('segmentation') # fails if more than one channel is segmented

        # Get segmented field
        grid = (stack.RegistrationOverTime.Chunk & key).get_grid(desired_res=1)
        stack_z, stack_y, stack_x = (stack.CorrectedStack & stack_key).fetch1('z', 'y', 'x')
        px_grid = (grid[..., ::-1] - np.array([stack_z, stack_y, stack_x]) - 0.5 +
                   np.array(instance.shape) / 2)
        reg_segmentation = ndimage.map_coordinates(instance, np.moveaxis(px_grid, -1, 0),
                                                   order=0) # nearest neighbor sampling

        # Get caiman masks and resize them
        field_res = (meso.ScanInfo.Field & key & {'session': key['scan_session']}).microns_per_pixel
        masks = (meso.Segmentation & key & {'session': key['scan_session']} &
                 {'segmentation_method': 6}).get_all_masks()
        masks = np.moveaxis(masks, -1, 0) # num_masks x height x width
        masks = np.stack([ndimage.zoom(f, field_res, order=1) for f in masks])

        # Binarize masks
        binary_masks = np.zeros(masks.shape, dtype=bool)
        for i, mask in enumerate(masks):
            ## Compute cumulative mass (similar to caiman)
            indices = np.unravel_index(np.flip(np.argsort(mask, axis=None), axis=0),
                                       mask.shape)  # max to min value in mask
            cumsum_mask = np.cumsum(mask[indices] ** 2) / np.sum(mask ** 2)
            binary_masks[i][indices] = cumsum_mask < 0.9

        # Insert
        caiman_field = np.argmax(binary_masks, axis=0).astype(np.int32)
        self.insert1({**key, 'segm_field': reg_segmentation,
                      'caiman_field': caiman_field})

        instance_props =  measure.regionprops(instance)
        instance_labels = np.array([p.label for p in instance_props])
        for prop in measure.regionprops(reg_segmentation):
            sunit_id = prop.label
            instance_prop = instance_props[np.argmax(instance_labels == sunit_id)]

            depth = (instance_prop.bbox[3] - instance_prop.bbox[0])
            height = (instance_prop.bbox[4] - instance_prop.bbox[1])
            width = (instance_prop.bbox[5] - instance_prop.bbox[2])
            volume = instance_prop.area
            sunit_z, sunit_y, sunit_x = (np.array([stack_z, stack_y, stack_x]) + np.array(
                instance_prop.centroid) - np.array(instance.shape) / 2 + 0.5)

            binary_sunit = reg_segmentation == sunit_id
            area = np.count_nonzero(binary_sunit)
            px_y, px_x = ndimage.measurements.center_of_mass(binary_sunit)
            px_coords = np.array([[px_y], [px_x]])
            mask_x, mask_y, mask_z = [ndimage.map_coordinates(grid[..., i], px_coords,
                                                              order=1)[0] for i in range(3)]
            distance = np.sqrt((sunit_z - mask_z) ** 2 + (sunit_y - mask_y) ** 2 +
                               (sunit_x - mask_x) ** 2)

            # Insert in StackUnit
            self.StackUnit.insert1({**key, 'sunit_id': sunit_id, 'depth': depth,
                                    'height': height, 'width': width, 'volume': volume,
                                    'area': area, 'sunit_z': sunit_z, 'sunit_y': sunit_y,
                                    'sunit_x': sunit_x, 'mask_z': mask_z,
                                    'mask_y': mask_y, 'mask_x': mask_x,
                                    'distance': distance})

            # Find closest caiman mask
            intersection = np.logical_and(binary_masks, binary_sunit).sum(axis=(1, 2)) # num_masks
            union = np.logical_or(binary_masks, binary_sunit).sum(axis=(1, 2)) # num_masks
            ious = intersection / union
            if np.any(ious > 0.1):
                caiman_id = np.argmax(ious) + 1
                caiman_iou = ious[caiman_id - 1]
                area = np.count_nonzero(binary_sunit)
                px_y, px_x = ndimage.measurements.center_of_mass(binary_masks[caiman_id -1])
                px_coords = np.array([[px_y], [px_x]])
                caiman_x, caiman_y, caiman_z = [ndimage.map_coordinates(grid[..., i],
                                                                        px_coords,
                                                                        order=1)[0] for i
                                                in range(3)]
                distance = np.sqrt((caiman_y - mask_y) ** 2 + (caiman_x - mask_x) ** 2)

                self.CaimanMask.insert1({**key, 'sunit_id': sunit_id,
                                         'caiman_id': caiman_id, 'caiman_iou': caiman_iou,
                                         'caiman_area': area, 'caiman_z': caiman_z,
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