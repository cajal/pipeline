""" Schemas specific to platinum mice"""
import datajoint as dj
import numpy as np

from . import meso, stack, notify, shared, experiment


schema = dj.schema('pipeline_fastmeso', locals(), create_tables=True)


@schema
class FastRegistration(dj.Computed):
    """ Simplified version of stack.FieldRegistration, see original for details"""
    definition = """ # align a 2-d scan field to a stack
    (stack_session) -> stack.CorrectedStack(session)  # animal_id, stack_session, stack_idx, pipe_version, volume_id
    (scan_session) -> experiment.Scan(session)  # animal_id, scan_session, scan_idx
    -> shared.Field
    (stack_channel) -> shared.Channel(channel)
    (scan_channel) -> shared.Channel(channel)
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
        return keys & stack.RegistrationTask() & meso.Quality().proj(scan_session='session')

    def _make_tuples(self, key):
        from scipy import ndimage
        from .utils import registration

        print('Registering', key)

        # Get stack
        stack_rel = (stack.CorrectedStack() & key & {'session': key['stack_session']})
        stack_ = stack_rel.get_stack(key['stack_channel'])

        # Get average field
        field_key = {'animal_id': key['animal_id'], 'session': key['scan_session'],
                     'scan_idx': key['scan_idx'], 'field': key['field'],
                     'channel': key['scan_channel']} #no pipe_version
        frames = (meso.Quality().SummaryFrames() & field_key).fetch1('summary')
        field = frames[:, :, int(frames.shape[-1] / 2)]

        # Drop some edges (only y and x) to avoid artifacts (and black edges in stacks)
        skip_dims = np.clip(np.round(np.array(stack_.shape) * 0.025), 1, None).astype(int)
        stack_ = stack_[:, skip_dims[1] : -skip_dims[1], skip_dims[2]: -skip_dims[2]]
        skip_dims = np.clip(np.round(np.array(field.shape) * 0.025), 1, None).astype(int)
        field = field[skip_dims[0] : -skip_dims[0], skip_dims[1]: -skip_dims[1]]

        # Rescale to match lowest resolution  (isotropic pixels/voxels)
        field_res = (meso.ScanInfo.Field() & field_key).microns_per_pixel
        dims = stack_rel.fetch1('um_depth', 'px_depth', 'um_height', 'px_height',
                                'um_width', 'px_width')
        stack_res = np.array([dims[0] / dims[1], dims[2] / dims[3], dims[4] / dims[5]])
        common_res = max(*field_res, *stack_res) # minimum available resolution
        stack_ = ndimage.zoom(stack_, stack_res / common_res, order=1)
        field = ndimage.zoom(field, field_res / common_res, order=1)

        # Get estimated depth of the field (from experimenters)
        stack_x, stack_y, stack_z = stack_rel.fetch1('x', 'y', 'z') # z of the first slice (zero is at surface depth)
        field_z = (meso.ScanInfo.Field() & field_key).fetch1('z') # measured in microns (zero is at surface depth)
        if field_z < stack_z or field_z > stack_z + dims[0]:
            msg_template = 'Warning: Estimated depth ({}) outside stack range ({}-{}).'
            print(msg_template.format(field_z, stack_z , stack_z + dims[0]))
        estimated_px_z = (field_z - stack_z + 0.5) / common_res # in pixels

        # Register
        z_range = 40 / common_res # search 40 microns up and down

        # Run rigid registration with no rotations
        result = registration.register_rigid(stack_, field, estimated_px_z, z_range)
        score, (x, y, z), (yaw, pitch, roll) = result

        # Map back to stack coordinates
        final_x = stack_x + x * (common_res / stack_res[2]) # in stack pixels
        final_y = stack_y + y * (common_res / stack_res[1]) # in stack pixels
        final_z = stack_z + (z + stack_.shape[0] / 2) * common_res # in microns*
        #* Best match in slice 0 will not result in z = 0 but 0.5 * z_step.

        # Insert
        self.insert1({**key, 'common_res': common_res, 'reg_x': final_x, 'reg_y': final_y,
                      'reg_z': final_z, 'score': score})
        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        x, y, z = (FastRegistration() &  key).fetch('reg_x', 'reg_y', 'reg_z')
        score = (FastRegistration() &  key).fetch('score')
        msg = ('FastRegistration for {} has been populated. Field registered at {}, {}, '
               '{} (z, y, x) with a score of {}').format(key, z, y, x, score)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg)