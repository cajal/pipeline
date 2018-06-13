import datajoint as dj
from datajoint.jobs import key_hash
import numpy as np
from commons import lab
import os

from . import experiment, notify
from .utils import h5


schema = dj.schema('pipeline_treadmill', locals())


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

        # Get counter timestamps and convert to seconds
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