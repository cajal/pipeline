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
class FieldSegmInStack(dj.Computed):
    definition = """ # find the respective segmentation from the stack for a registered field
    -> RegistrationOverTime
    ---
    common_res              : float                     # common resolution stack and field were downsampled to
    stack_field             : blob                      # field (image x height) of cell ids
    caiman_field            : blob                      # field created from caiman maks 
    """
    @property
    def key_source(self):
        return RegistrationOverTime() & stack.Segmentation().proj(stack_session='session')

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
        -> FieldSegmInStack.StackUnit
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
        common_res = max(*field_res, *stack_res) # minimum available resolution
        instance = ndimage.zoom(instance, stack_res / common_res, order=0)
        masks = np.stack(ndimage.zoom(f, field_res / common_res, order=1) for f in masks)

        # TODO: I could dilate em herre to make em more round

        # Binarize masks
        binary_masks = np.zeros(masks.shape, dtype=bool)
        for i, mask in enumerate(masks):
            ## Compute cumulative mass (similar to caiman)
            indices = np.unravel_index(np.flip(np.argsort(mask, axis=None), axis=0),
                                       mask.shape)  # max to min value in mask
            cumsum_mask = np.cumsum(mask[indices] ** 2) / np.sum(mask ** 2)
            binary_masks[i][indices] = cumsum_mask < 0.9

        # Compute z, y, x of field as distances to the center of the stack
        reg_x, reg_y, reg_z = (RegistrationOverTime() & key).fetch1('reg_x', 'reg_y',
                                                                    'reg_z')
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
            if np.any(ious > 0):
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