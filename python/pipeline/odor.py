import os
import h5py
import itertools

import numpy as np
import datajoint as dj

from commons import lab
from scipy import ndimage
from pipeline.utils import h5
from datajoint.hash import key_hash
from .exceptions import PipelineException
from pipeline import meso, stack, mice, experiment, shared


#dj.config['external-odor'] = {'protocol': 'file',
#                              'location': '/mnt/dj-stor01/pipeline-externals'}

schema = dj.schema('pipeline_odor', locals(), create_tables=False)


@schema
class Odorant(dj.Lookup):
    definition = """ # Odorants used in solutions.
    odorant                     : varchar(32)          # name of odorant
    ---
    molar_mass                  : float                # odorant molar mass (g/mole)
    density                     : float                # odorant density (g/mL)
    vapor_pressure              : float                # vapor pressure at 25C (mmHg)
    odorant_description = ''    : varchar(2048)
    """
    contents = [
        ['Empty', '0', '0', '0', '']
    ]


@schema
class OdorSolution(dj.Manual):
    definition = """ # Solutions made for olfactory experiments.
    -> Odorant
    concentration               : decimal(2,2)         # Dilution fraction
    solution_date               : date                 # date solution was created (YYYY-MM-DD)
    ---
    solution_notes = ''         : varchar(2048)
    """


@schema
class OdorSession(dj.Manual):
    definition = """ # Single session of olfactory experiment.
    -> mice.Mice
    odor_session                : smallint unsigned    # session index for the mouse
    ---
    odor_path                   : varchar(2048)        # folder where h5 files for olfaction are stored
    odor_session_date           : date                 # date session was recorded (YYYY-MM-DD)
    time_loaded = null          : time
    """


@schema
class OdorConfig(dj.Manual):
    definition = """ # Configuration of solutions for each olfactory experiment. One entry for each channel.
    -> OdorSession
    channel                     : tinyint unsigned     # channel number of assigned solution
    ---
    -> OdorSolution
    """


@schema
class OdorRecording(dj.Manual):
    definition = """ # Single scan/recording within an olfactory session.
    -> OdorSession
    recording_idx               : smallint unsigned    # recording index for session
    ---
    filename                    : varchar(255)
    """


@schema
class OdorTrials(dj.Imported):
    definition = """ # Single presentation of olfactory stimulus. One entry for each ON channel.
    -> OdorRecording
    trial_idx                   : smallint unsigned    # trial index for recording
    channel                     : tinyint unsigned     # channel number used
    ---
    trial_start_time            : float                # start of trial on olfactory clock (seconds)
    trial_end_time              : float                # end of trial on olfactory clock (seconds)
    """

    def convert_valves(num):
        """ Converts decimal number encoding valve states into boolean array
        :param num: int. Decimal number encoding valve state in binary
        :returns: numpy array of booleans. Array encodes valve state (True=OPEN) starting with the smallest valve number
            ex. decimal(10) -> [False, True, False, True] -> 2nd and 4th valve open.
        """

        # Format integer into binary, then convert each 0/1 into boolean
        # Finally, reverse list to make first index encode first valve
        valve_states = [bool(int(d)) for d in format(int(num), 'b')][::-1]
        return np.array(valve_states)

    def make(self, key):

        print(f'Populating trials for {key}')

        # Get olfactory h5 path and filename
        olfactory_path = (OdorSession & key).fetch1('odor_path')
        local_path = lab.Paths().get_local_path(olfactory_path)
        filename_base = (OdorRecording & key).fetch1('filename')
        digital_filename = os.path.join(local_path, filename_base + '_D_%d.h5')

        # Load olfactory data
        digital_data = h5.read_digital_olfaction_file(digital_filename)

        # Check valve data ends with all valves closed
        if digital_data['valves'][-1] != 0:
            msg = f'Error: Final valve state is open! Ending time cannot be calculated for {key}.'
            raise PipelineException(msg)

        valve_open_idx = np.where(digital_data['valves'] > 0)[0]
        trial_valve_states = digital_data['valves'][valve_open_idx]
        trial_start_times = h5.ts2sec(digital_data['ts'][valve_open_idx])

        # Shift start indices by one to get end indices
        trial_end_times = h5.ts2sec(digital_data['ts'][valve_open_idx + 1])

        # All keys are appended to a list and inserted at the end to prevent errors from halting mid-calculation
        all_trial_keys = []

        # Find all trials and insert a key for each channel open during each trial
        for trial_num, (state, start, stop) in enumerate(zip(trial_valve_states, trial_start_times, trial_end_times)):

            valve_array = OdorTrials.convert_valves(state)

            for valve_num in np.where(valve_array)[0]:  # Valve array is already a boolean, look for all true values

                # We start counting valves at 1, not 0 like python indices
                valve_num = valve_num + 1
                trial_key = [key['animal_id'], key['odor_session'], key['recording_idx'],
                             trial_num, valve_num, start, stop]
                all_trial_keys.append(trial_key)

        self.insert(all_trial_keys)

        print(f'{valve_open_idx.shape[0]} odor trials found and inserted for {key}.\n')


@schema
class OdorSync(dj.Imported):
    definition = """ # ScanImage frame times on Olfactory Master Clock.
    -> OdorRecording
    ---
    signal_start_time           : float                # start of analog signal recording on olfactory clock (seconds)
    signal_duration             : float                # duration of analog signal (seconds)
    frame_times                 : longblob             # start of each scanimage frame on olfactory clock (seconds)
    sync_ts=CURRENT_TIMESTAMP   : timestamp
    """

    def make(self, key):

        print(f'Populating Sync for {key}')

        # Get olfactory h5 path and filename
        olfactory_path = (OdorSession & key).fetch1('odor_path')
        local_path = lab.Paths().get_local_path(olfactory_path)
        filename_base = (OdorRecording & key).fetch1('filename')
        analog_filename = os.path.join(local_path, filename_base + '_%d.h5')

        # Load olfactory data
        analog_data = h5.read_analog_olfaction_file(analog_filename)

        scan_times = h5.ts2sec(analog_data['ts'], is_packeted=True)
        binarized_signal = analog_data['scanImage'] > 2.7  # TTL voltage low/high threshold
        rising_edges = np.where(np.diff(binarized_signal.astype(int)) > 0)[0]
        frame_times = scan_times[rising_edges]

        # Correct NaN gaps in timestamps (mistimed or dropped packets during recording)
        if np.any(np.isnan(frame_times)):
            # Raise exception if first or last frame pulse was recorded in mistimed packet
            if np.isnan(frame_times[0]) or np.isnan(frame_times[-1]):
                msg = ('First or last frame happened during misstamped packets. Pulses '
                       'could have been missed: start/end of scanning is unknown.')
                raise PipelineException(msg)

            # Fill each gap of nan values with correct number of timepoints
            frame_period = np.nanmedian(np.diff(frame_times))  # approx
            nan_limits = np.where(np.diff(np.isnan(frame_times)))[0]
            nan_limits[1::2] += 1  # limits are indices of the last valid point before the nan gap and first after it
            correct_fts = []
            for i, (start, stop) in enumerate(zip(nan_limits[::2], nan_limits[1::2])):
                correct_fts.extend(frame_times[0 if i == 0 else nan_limits[2 * i - 1]: start + 1])
                num_missing_points = int(round((frame_times[stop] - frame_times[start]) /
                                               frame_period - 1))
                correct_fts.extend(np.linspace(frame_times[start], frame_times[stop],
                                               num_missing_points + 2)[1:-1])
            correct_fts.extend(frame_times[nan_limits[-1]:])
            frame_times = np.array(correct_fts)

        # Check that frame times occur at the same period
        frame_intervals = np.diff(frame_times)
        frame_period = np.median(frame_intervals)
        if np.any(abs(frame_intervals - frame_period) > 0.15 * frame_period):
            raise PipelineException('Frame time period is irregular')

        self.insert1({**key, 'signal_start_time': frame_times[0],
                      'signal_duration': frame_times[-1] - frame_times[0],
                      'frame_times': frame_times})

        print(f'ScanImage sync added for animal {key["animal_id"]}, '
              f'olfactory session {key["odor_session"]}, '
              f'recording {key["recording_idx"]}\n')


@schema
class Respiration(dj.Imported):
    definition = """ # Analog recording of mouse respiration
    -> OdorRecording
    ---
    trace                       : longblob             # mouse respiration (arbitrary units)
    times                       : longblob             # trace times on olfactory clock (seconds)
    """

    def make(self, key):

        print(f'Populating Respiration for {key}')

        # Get olfactory h5 path and filename
        olfactory_path = (OdorSession & key).fetch1('odor_path')
        local_path = lab.Paths().get_local_path(olfactory_path)
        filename_base = (OdorRecording & key).fetch1('filename')
        analog_filename = os.path.join(local_path, filename_base + '_%d.h5')

        # Load olfactory data
        analog_data = h5.read_analog_olfaction_file(analog_filename)
        breath_times = h5.ts2sec(analog_data['ts'], is_packeted=True)
        breath_trace = analog_data['breath']

        # Correct NaN gaps in timestamps (mistimed or dropped packets during recording)
        if np.any(np.isnan(breath_times)):
            # Raise exception if first or last frame pulse was recorded in mistimed packet
            if np.isnan(breath_times[0]) or np.isnan(breath_times[-1]):
                msg = ('First or last breath happened during misstamped packets. Pulses '
                       'could have been missed: start/end of collection is unknown.')
                raise PipelineException(msg)

            # Linear interpolate between nans
            nans_idx = np.where(np.isnan(breath_times))[0]
            non_nans_idx = np.where(~np.isnan(breath_times))[0]
            breath_times[nans_idx] = np.interp(nans_idx, non_nans_idx, breath_times[non_nans_idx])
            print(f'Largest NaN gap found: {np.max(np.abs(np.diff(breath_times[non_nans_idx])))} seconds')

        # Check that frame times occur at the same period
        breath_intervals = np.diff(breath_times)
        breath_period = np.median(breath_intervals)
        if np.any(abs(breath_intervals - breath_period) > 0.15 * breath_period):
            raise PipelineException('Breath time period is irregular')

        # Error check tracing and timing match
        if breath_trace.shape[0] != breath_times.shape[0]:
            raise PipelineException('Breath timing and trace mismatch!')

        breath_key = {**key, 'trace': breath_trace, 'times': breath_times}

        self.insert1(breath_key)
        print(f'Respiration data for {key} successfully inserted.\n')


@schema
class MesoMatch(dj.Manual):
    definition = """ # Match between Odor Recording and Scan Session
    -> OdorRecording
    ---
    -> meso.ScanInfo
    """


@schema
class StackMatching(dj.Computed):
    definition = """ # match segmented masks by proximity in the stack
    
    -> stack.CorrectedStack.proj(stack_session='session')
    -> shared.RegistrationMethod
    -> shared.SegmentationMethod
    ---
    min_distance            :tinyint                   # distance used as threshold to accept two masks as the same
    max_height              :tinyint                   # maximum allowed height of a joint mask
    """

    @property
    def key_source(self):
        return (stack.CorrectedStack.proj(stack_session='session') *
                shared.RegistrationMethod.proj() * shared.SegmentationMethod.proj() &
                stack.Registration & {'segmentation_method': 1}) & MesoMatch.proj(stack_session='session')

    class Unit(dj.Part):
        definition = """ # a unit in the stack
        
        -> master
        munit_id            :int                       # unique id in the stack
        ---
        munit_x             :float                     # (um) position of centroid in motor coordinate system
        munit_y             :float                     # (um) position of centroid in motor coordinate system
        munit_z             :float                     # (um) position of centroid in motor coordinate system
        """

    class Match(dj.Part):
        definition = """ # Scan unit to stack unit match (n:1 relation)
        
        -> master
        -> experiment.Scan.proj(scan_session='session')  # animal_id, scan_session, scan_idx
        unit_id             :int                       # unit id from ScanSet.Unit
        ---
        -> StackMatching.Unit
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
        min_distance = 40
        max_height = 100

        # Create list of units
        units = []  # stands for matched units
        for field in stack.Registration & key:
            # Edge case: when two channels are registered, we don't know which to use
            if len(stack.Registration.proj(ignore='scan_channel') & field) > 1:
                msg = ('More than one channel was registered for {animal_id}-'
                       '{scan_session}-{scan_idx} field {field}'.format(**field))
                raise PipelineException(msg)

            # Get registered grid
            field_key = {'animal_id': field['animal_id'],
                         'session': field['scan_session'], 'scan_idx': field['scan_idx'],
                         'field': field['field']}
            um_per_px = (meso.ScanInfo.Field & field_key).microns_per_pixel
            grid = (stack.Registration & field).get_grid(type='affine', desired_res=um_per_px)

            # Create cell objects
            for channel_key in (meso.ScanSet & field_key &
                                {'segmentation_method': key['segmentation_method']}):  # *
                field_masks = meso.ScanSet.Unit & channel_key
                unit_keys, xs, ys = (meso.ScanSet.UnitInfo & field_masks).fetch('KEY',
                        'px_x', 'px_y')
                px_coords = np.stack([ys, xs])
                xs, ys, zs = [ndimage.map_coordinates(grid[..., i], px_coords, order=1)
                              for i in range(3)]
                units += [StackMatching.MatchedUnit(*args, key_hash(channel_key)) for args in
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


    def plot_centroids3d(self):
        """ Plots the centroids of all units in the motor coordinate system (in microns)

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Get centroids
        xs, ys, zs = (StackMatching.Unit).fetch('munit_x', 'munit_y', 'munit_z')

        # Plot
        fig = plt.figure(figsize=(30, 10))
        views = ((30,-60), (1,0), (90,0))  # (elev, azim)
        for n,view in enumerate(views):

            ax = fig.add_subplot(1, 3, n+1, projection='3d')
            ax.view_init(elev=view[0], azim=view[1])
            ax.scatter(xs, ys, zs, alpha=0.5)
            ax.invert_zaxis()
            ax.set_xlabel('x (um)')
            ax.set_ylabel('y (um)')
            ax.set_zlabel('z (um)')
            
        plt.tight_layout()

        return fig
