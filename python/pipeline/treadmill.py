import datajoint as dj
from datajoint.jobs import key_hash
import numpy as np
from commons import lab
import os

from . import experiment, notify
from .utils import h5
from .exceptions import PipelineException


schema = dj.schema('pipeline_treadmill', locals())


@schema
class Sync(dj.Computed):
    definition = """ # syncing scanimage frame times to behavior clock

    -> experiment.Scan
    ---
    frame_times                         : longblob      # times of each slice in behavior clock
    behavior_sync_ts=CURRENT_TIMESTAMP  : timestamp
    """
    @property
    def key_source(self):
        return experiment.Scan() & experiment.Scan.BehaviorFile().proj()

    def _make_tuples(self, key):
        # Get behavior filename
        behavior_path = (experiment.Session() & key).fetch1('behavior_path')
        local_path = lab.Paths().get_local_path(behavior_path)
        filename = (experiment.Scan.BehaviorFile() & key).fetch1('filename')
        full_filename = os.path.join(local_path, filename)

        # Read file
        data = h5.read_behavior_file(full_filename)

        # Read counter timestamps and convert to seconds
        timestamps_in_secs = h5.ts2sec(data['ts'], is_packeted=True)

        # Detect rising edges in scanimage clock signal (start of each frame)
        binarized_signal = data['scanImage'] > 2.7 # TTL voltage low/high threshold
        rising_edges = np.where(np.diff(binarized_signal.astype(int)) > 0)[0]
        frame_times = timestamps_in_secs[rising_edges]

        # Correct NaN gaps in timestamps (mistimed or dropped packets during recording)
        if np.any(np.isnan(frame_times)):
            # Raise exception if first or last frame pulse was recorded in mistimed packet
            if np.isnan(frame_times[0]) or np.isnan(frame_times[-1]):
                msg = ('First or last frame happened during misstamped packets. Pulses '
                       'could have been missed: start/end of scanning is unknown.')
                raise PipelineException(msg)

            # Fill each gap of nan values with correct number of timepoints
            frame_period = np.nanmedian(np.diff(frame_times)) # approx
            nan_limits = np.where(np.diff(np.isnan(frame_times)))[0]
            nan_limits[1::2] += 1 # limits are indices of the last valid point before the nan gap and first after it
            correct_fts = []
            for i, (start, stop) in enumerate(zip(nan_limits[::2], nan_limits[1::2])):
                correct_fts.extend(frame_times[0 if i == 0 else nan_limits[2 * i - 1]: start + 1])
                num_missing_points = int(round((frame_times[stop] - frame_times[start]) /
                                                frame_period - 1))
                correct_fts.extend(np.linspace(frame_times[start], frame_times[stop],
                                               num_missing_points + 2)[1:-1])
            correct_fts.extend(frame_times[nan_limits[-1]:])
            frame_times = correct_fts

            # Record the NaN fix
            num_gaps = int(len(nan_limits) / 2)
            nan_length = sum(nan_limits[1::2] - nan_limits[::2]) * frame_period # secs
            experiment.Fixes.insert1(key, skip_duplicates=True)
            experiment.Fixes.IrregularTimestamps.insert1({**key, 'num_gaps': num_gaps,
                                                          'num_secs': nan_length})

        # Check that frame times occur at the same period
        frame_intervals = np.diff(frame_times)
        frame_period = np.median(frame_intervals)
        if np.any(abs(frame_intervals - frame_period) > 0.15 * frame_period):
            raise PipelineException('Frame time period is irregular')

        # Drop last frame time if scan crashed or was stopped before completion
        valid_times = ~np.isnan(timestamps_in_secs[rising_edges[0]: rising_edges[-1]]) # restricted to scan period
        binarized_valid = binarized_signal[rising_edges[0]: rising_edges[-1]][valid_times]
        frame_duration = np.mean(binarized_valid) * frame_period
        falling_edges = np.where(np.diff(binarized_signal.astype(int)) < 0)[0]
        last_frame_duration = timestamps_in_secs[falling_edges[-1]] - frame_times[-1]
        if (np.isnan(last_frame_duration) or last_frame_duration < 0 or
            abs(last_frame_duration - frame_duration) > 0.15 * frame_duration):
            frame_times = frame_times[:-1]

        self.insert1({**key, 'frame_times': frame_times})
        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        msg = 'treadmill.Sync for {animal_id}-{session}-{scan_idx} has been populated.'
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg.format(**key))


@schema
class Treadmill(dj.Computed):
    definition = """ # treadmill velocity synchronized to behavior clock

    -> experiment.Scan
    ---
    treadmill_raw                       :longblob       # raw treadmill counts
    treadmill_time                      :longblob       # (secs) velocity timestamps in behavior clock
    treadmill_vel                       :longblob       # (cm/sec) wheel velocity
    treadmill_ts=CURRENT_TIMESTAMP      :timestamp
    """
    @property
    def key_source(self):
        return experiment.Scan() & experiment.Scan.BehaviorFile().proj()

    def _make_tuples(self, key):
        # Get behavior filename
        behavior_path = (experiment.Session() & key).fetch1('behavior_path')
        local_path = lab.Paths().get_local_path(behavior_path)
        filename = (experiment.Scan.BehaviorFile() & key).fetch1('filename')
        full_filename = os.path.join(local_path, filename)

        # Read file
        data = h5.read_behavior_file(full_filename)

        # Read counter timestamps and convert to seconds
        timestamps_in_secs = h5.ts2sec(data['wheel'][1])
        ts = h5.ts2sec(data['ts'], is_packeted=True)
        # edge case when ts and wheel ts start in different sides of the master clock max value 2 **32
        if abs(ts[0] - timestamps_in_secs[0]) > 2 ** 31:
            timestamps_in_secs += (2 ** 32 if ts[0] > timestamps_in_secs[0] else -2 ** 32)

        # Read wheel position counter and fix wrap around at 2 ** 32
        wheel_position = data['wheel'][0]
        wheel_diffs = np.diff(wheel_position)
        for wrap_idx in np.where(abs(wheel_diffs) > 2 ** 31)[0]:
            wheel_position[wrap_idx + 1:] += (2 ** 32 if wheel_diffs[wrap_idx] < 0 else -2 ** 32)
        wheel_position -= wheel_position[0] # start counts at zero

        # Compute wheel velocity
        num_samples = int(round((timestamps_in_secs[-1] - timestamps_in_secs[0]) * 10)) # every 100 msecs
        sample_times = np.linspace(timestamps_in_secs[0], timestamps_in_secs[-1], num_samples)
        sample_position = np.interp(sample_times, timestamps_in_secs, wheel_position)
        counter_velocity = np.gradient(sample_position) * 10 # counts / sec

        # Transform velocity from counts/sec to cm/sec
        wheel_specs = experiment.TreadmillSpecs() * experiment.Session() & key
        diameter, counts_per_rev = wheel_specs.fetch1('diameter', 'counts_per_revolution')
        wheel_perimeter = np.pi * diameter # 1 rev = xx cms
        velocity = (counter_velocity / counts_per_rev) * wheel_perimeter # cm /sec

        # Resample at initial timestamps
        velocity = np.interp(timestamps_in_secs, sample_times, velocity)

        # Fill with NaNs for out-of-range data or mistimed packets
        velocity[timestamps_in_secs < ts[0]] = float('nan')
        velocity[timestamps_in_secs > ts[-1]] = float('nan')
        nan_limits = np.where(np.diff([0, *np.isnan(ts), 0]))[0]
        for start, stop in zip(nan_limits[::2], nan_limits[1::2]):
            lower_ts = float('-inf') if start == 0 else ts[start - 1]
            upper_ts = float('inf') if stop == len(ts) else ts[stop]
            velocity[np.logical_and(timestamps_in_secs > lower_ts,
                                    timestamps_in_secs < upper_ts)] = float('nan')
        timestamps_in_secs[np.isnan(velocity)] = float('nan')

        # Insert
        self.insert1({**key, 'treadmill_time': timestamps_in_secs,
                      'treadmill_raw': data['wheel'][0], 'treadmill_vel': velocity})
        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        import matplotlib.pyplot as plt
        time, velocity = (self & key).fetch1('treadmill_time', 'treadmill_vel')
        fig = plt.figure()
        plt.plot(time, velocity)
        plt.ylabel('Treadmill velocity (cm/sec)')
        plt.xlabel('Seconds')
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)

        msg = 'treadmill velocity for {animal_id}-{session}-{scan_idx}'.format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=img_filename, file_title=msg)