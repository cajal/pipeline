import datajoint as dj
import numpy as np
from commons import lab
from scipy import interpolate as interp
import os

from . import experiment, notify
from .utils import h5, signal
from .exceptions import PipelineException


schema = dj.schema('pipeline_treadmill', locals())


@schema
class Sync(dj.Computed):
    definition = """ # syncing scanimage frame times to behavior clock

    -> experiment.Scan
    ---
    frame_times=null                    : longblob      # times of each slice in behavior clock
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
        data = h5.read_video_hdf5(full_filename)

        # Read counter timestamps and convert to seconds
        timestamps_in_secs = h5.ts2sec(data['ts'], is_packeted=True)

        # Remove NaNs from timestamps
        nans = np.isnan(timestamps_in_secs)
        xs = np.arange(len(timestamps_in_secs))
        timestamps_in_secs = np.interp(xs, xs[~nans], timestamps_in_secs[~nans])

        if np.any(nans):
            raise PipelineException('Temporary exception until experiment.Fixes is set up: Gaps in the ts signal')

        # Detect peaks in scanimage clock signal
        fps = 1 / np.median(np.diff(timestamps_in_secs))
        n = int(np.ceil(0.0002 * fps))
        k = np.hamming(2 * n)
        k /= -k.sum()
        k[:n] = -k[:n]

        pulses = np.convolve(data['scanImage'], k, mode='full')[n:-n + 1]  # mode='same' with MATLAB compatibility
        peaks = signal.spaced_max(pulses, 0.005 * fps)
        peaks = peaks[pulses[peaks] > 0.1 * np.percentile(pulses[peaks], 90)]
        peaks = signal.longest_contiguous_block(peaks)

        self.insert1({**key, 'frame_times': timestamps_in_secs[peaks]})
        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        msg = 'treadmill.Sync for {animal_id}-{session}-{scan_idx} has been populated.'
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg.format(**key))



@schema
class Treadmill(dj.Computed):
    definition = """ # behavior times

    -> experiment.Scan
    ---
    treadmill_raw                       :longblob           #raw treadmill counts
    treadmill_vel                       :longblob           #ball velocity integrated over 100ms bins in cm/sec
    treadmill_time                      :longblob           #timestamps of each sample in seconds on behavior clock
    treadmill_ts = CURRENT_TIMESTAMP    :timestamp          #automatic
    """

    def _make_tuples(self, key):
        # Get behavior filename
        behavior_path = (experiment.Session() & key).fetch1('behavior_path')
        local_path = lab.Paths().get_local_path(behavior_path)
        filename = (experiment.Scan.BehaviorFile() & key).fetch1('filename')
        full_filename = os.path.join(local_path, filename)

        # read hdf file for ball data
        data = h5.read_video_hdf5(full_filename)

        # read out counter time stamp and convert to seconds
        ball_time = h5.ts2sec(data['wheel'][1])

        # read out raw ball counts and integrate by 100ms intervals
        ball_raw = data['wheel'][0]
        ball_time_to_raw = interp.interp1d(ball_time, ball_raw - ball_raw[0])
        bin_times = np.arange(ball_time[0], ball_time[-1], .1)
        bin_times[-1] = ball_time[-1]
        ball_counts = np.append([0], np.diff(ball_time_to_raw(bin_times)))

        # pull Treadmill specs, warn if more than one Treadmill fits session key
        diam, counts_per_revolution = (
                experiment.TreadmillSpecs() * experiment.Session() & key & 'treadmill_start_date <= session_date').fetch(
            'diameter', 'counts_per_revolution')
        if len(diam) != 1:
            raise PipelineException('Unclear which treadmill fits session key')

        # convert ball counts to cm/s for each ball time point
        cmPerCount = np.pi * diam[-1] / counts_per_revolution[-1]
        ball_time_to_vel = interp.interp1d(bin_times, ball_counts * cmPerCount * 10)
        ball_vel = ball_time_to_vel(ball_time)

        # assign calculated properties to key
        key['treadmill_time'] = ball_time
        key['treadmill_raw'] = ball_raw
        key['treadmill_vel'] = ball_vel

        # insert and notify user
        self.insert1(key)
        self.notify({k: key[k] for k in self.heading.primary_key})

    @notify.ignore_exceptions
    def notify(self, key):
        msg = 'treadmill.Treadmill for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg)
