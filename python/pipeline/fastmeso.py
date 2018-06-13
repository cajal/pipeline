""" Schemas specific to platinum mice. Not the cleanest design/code"""
import datajoint as dj
import numpy as np

from . import reso, meso, stack, notify, shared, experiment
from .utils import enhancement
from stimulus import stimulus
from datajoint.jobs import key_hash


dj.config['external-analysis'] = dict(
    protocol='file',
    location='/mnt/lab/users/ecobost/fastmeso')
dj.config['cache'] = '/tmp/dj-cache'

schema = dj.schema('pipeline_fastmeso', locals(), create_tables=True)


@schema
class PreprocessedStack(dj.Computed):
    definition = """ # stack after preprocessing (and cached in memory) for RegistrationOverTime
    (stack_session) -> stack.CorrectedStack(session)  # animal_id, stack_session, stack_idx, pipe_version, volume_id
    (stack_channel) -> shared.Channel(channel)
    ---
    stack :     external-analysis   # motion corrected uint16
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
        from scipy import ndimage
        from .utils import registration

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
class ConditionTraces(dj.Computed):
    definition=""" # 10-secs traces for each unit in stack during some condition (used for visualization)
    -> stack.StackSet
    -> stimulus.Condition
    ---
    common_fps              :float          #
    """
    @property
    def key_source(self):
        conditions = (stimulus.Clip().aggr(stimulus.Trial() & animal, 'movie_name',
                                           nscans='count(DISTINCT scan_idx)')
                      & 'nscans>=10' & {'movie_name': 'matrixrl'}).proj().fetch(as_dict=True)
                      # clips from Matrix Reloaded (choosen because of good oracle performance) presented in more than 10 scans

        return stack.StackSet() * (stimulus.Condition() & conditions)

    class Trace(dj.Part):
        definition = """
        -> master
        -> stack.StackSet.Unit
        ---
        num_masks           : int        # num_masks averaged to get this trace
        trace               : longblob   # 10-secs trace
        corr=NULL           : float      # Mean correlation across traces for masks forming this unit
        """

    def make(self, key):
        import itertools

        # Compute common fps to which traces will be downsampled
        reg_fields = (stack.FieldRegistration() & key).proj(session='scan_session')
        common_fps = min([*(reso.ScanInfo() & reg_fields).fetch('fps'), *(meso.ScanInfo() & reg_fields).fetch('fps')])

        # Create list of scan units
        traces = {} # dictionary with unit-name -> trace pairs
        for field_key in reg_fields.fetch('KEY'):
            # Get some field info
            scan_name = '{animal_id}-{session}-{scan_idx}'.format(**field_key)
            pipe = reso if reso.ScanInfo() & field_key else meso if meso.ScanInfo() & field_key else None

            # Get flip_times at common_fps resolution
            if len(stimulus.Trial() & key & field_key) == 0:
                print('No trials. Skipping', field_key)
                continue
            trial_times = (stimulus.Trial() & key & field_key).fetch('flip_times')
            trial_times = [np.linspace(ft.min(), ft.max(), int(round((ft.max() - ft.min()) * common_fps))) for ft in trial_times]

            # Get scan frame times
            if len(stimulus.Sync() & field_key) == 0:
                print('No Sync. Skipping', field_key)
                continue
            num_frames = (pipe.ScanInfo() & field_key).fetch1('nframes')
            num_slices = len(np.unique((pipe.ScanInfo.Field().proj('z', nomatch='field') & field_key).fetch('z')))
            frame_times = (stimulus.Sync() & field_key).fetch1('frame_times', squeeze=True) # one per depth
            frame_times = frame_times[:num_slices * num_frames:num_slices] # one per volume

            # Get unit traces
            somas = (pipe.MaskClassification.Type() & {'type': 'soma'})
            units = pipe.ScanSet.Unit() & field_key & somas
            spikes = pipe.Activity.Trace() * pipe.ScanSet.UnitInfo() & units.proj()
            for unit_id, ms_delay, trace in zip(*spikes.fetch('unit_id', 'ms_delay', 'trace')):
                interp_traces =  np.interp(trial_times, frame_times + ms_delay / 1000, trace) # num_trials x trial_duration
                traces['{}-{}'.format(scan_name, unit_id)] = np.mean(interp_traces, axis=0)
            #TODO: Deal with frame_numbers wrapping around after 2**32
            #TODO: Deal with trial times being outside range of frame times
        print('Traces created')

        # Create the mean traces per munit_id
        self.insert1({**key, 'common_fps': common_fps})
        concat_query = 'CONCAT_WS("-", animal_id, scan_session, scan_idx, unit_id)'
        match_rel = (stack.StackSet.Match() & key).proj('munit_id', unit_name=concat_query)
        munit_ids, unit_names = match_rel.fetch('munit_id', 'unit_name', order_by='munit_id')
        for munit_id, unit_group in itertools.groupby(zip(munit_ids, unit_names), lambda x: x[0]):
            unit_traces = list(filter(lambda x: x is not None, [traces.get(un, None) for _, un in unit_group]))
            num_traces = len(unit_traces) # traces coming from diff cells+
            if num_traces > 0:
                munit_trace = np.mean(unit_traces, axis=0)
                corr = np.mean(np.corrcoef(unit_traces)[np.triu_indices(num_traces, k=1)]) if num_traces > 1 else None
                self.Trace.insert1({**key, 'munit_id': munit_id, 'num_masks': num_traces,
                                    'trace': munit_trace, 'corr': corr})
