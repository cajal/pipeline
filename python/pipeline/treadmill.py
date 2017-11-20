import datajoint as dj
import numpy as np
from commons import lab

from .utils.signal import spaced_max, longest_contiguous_block
from .utils.h5 import read_video_hdf5, ts2sec
from . import experiment, notify
from .exceptions import PipelineException

import datajoint as dj

from scipy.interpolate import interp1d

schema = dj.schema('pipeline_treadmill', locals())


@schema
class Sync(dj.Computed):
    definition = """
    -> experiment.Scan
    ---
    frame_times=null                    : longblob                      # times of frames and slices on behavior clock
    behavior_sync_ts=CURRENT_TIMESTAMP  : timestamp                     # automatic
    """

    def _make_tuples(self, key):


        rel = experiment.Session() * experiment.Scan.BehaviorFile().proj(
            hdf_file='filename')

        info = (rel & key).fetch1()

        # replace number by %d for hdf-file reader
        tmp = info['hdf_file'].split('.')
        if not '%d' in tmp[0]:
            info['hdf_file'] = tmp[0][:-1] + '%d.' + tmp[-1]

        hdf_path = lab.Paths().get_local_path("{behavior_path}/{hdf_file}".format(**info))

        data = read_video_hdf5(hdf_path)
        packet_length = data['analogPacketLen']
        dat_time, _ = ts2sec(data['ts'], packet_length)


        dat_fs = 1./np.median(np.diff(dat_time))



        n = int(np.ceil(0.0002 * dat_fs))
        k = np.hamming(2 * n)
        k /= -k.sum()
        k[:n] = -k[:n]


        pulses = np.convolve(data['scanImage'], k, mode='full')[n:-n+1] # mode='same' with MATLAB compatibility


        peaks = spaced_max(pulses, 0.005 * dat_fs)
        peaks = peaks[pulses[peaks] > 0.1 * np.percentile(pulses[peaks], 90)]
        peaks = longest_contiguous_block(peaks)


        self.insert1(dict(key, frame_times = dat_time[peaks]))
        self.notify(key)

    def notify(self, key):
        msg = 'behavior.Sync for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg)
        
@schema
class Treadmill(dj.Computed):
    definition = """
    -> experiment.Scan
    ---
    treadmill_raw                       :longblob           #raw treadmill counts
    treadmill_vel                       :longblob           #ball velocity integrated over 100ms bins in cm/sec 
    treadmill_time                      :longblob           #timestamps of each sample in seconds on behavior clock
    treadmill_ts = CURRENT_TIMESTAMP    :timestamp          #automatic
    """
    # adapted from Treadmill.m by Paul Fahey, 2017-10-13
    def _make_tuples(self, key):
        #pull filename for key
        rel = experiment.Session() * experiment.Scan.BehaviorFile().proj(
            hdf_file='filename')
        info = (rel & key).fetch1()

        # replace number by %d for hdf-file reader
        tmp = info['hdf_file'].split('.')
        if not '%d' in tmp[0]:
            info['hdf_file'] = tmp[0][:-1] + '%d.' + tmp[-1]

        #read hdf file for ball data
        hdf_path = lab.Paths().get_local_path("{behavior_path}/{hdf_file}".format(**info))
        data = read_video_hdf5(hdf_path)

        #read out counter time stamp and convert to seconds
        packet_length = data['analogPacketLen']
        ball_time,_ = ts2sec(data['ball'].transpose()[1],packet_length)

        #read out raw ball counts and integrate by 100ms intervals
        ball_raw = data['ball'].transpose()[0]
        ball_time_to_raw = interp1d(ball_time,ball_raw-ball_raw[0])
        bin_times = np.arange(ball_time[0],ball_time[-1],.1)
        bin_times[-1] = ball_time[-1]
        ball_counts = np.append([0],np.diff(ball_time_to_raw(bin_times)))

        #pull Treadmill specs, warn if more than one Treadmill fits session key
        diam, counts_per_revolution = (
            experiment.TreadmillSpecs() * experiment.Session() & key & 'treadmill_start_date <= session_date').fetch('diameter', 'counts_per_revolution')
        if len(diam) != 1:
            raise PipelineException('Unclear which treadmill fits session key')

        #convert ball counts to cm/s for each ball time point
        cmPerCount = np.pi*diam[-1]/counts_per_revolution[-1]
        ball_time_to_vel = interp1d(bin_times, ball_counts*cmPerCount*10)
        ball_vel = ball_time_to_vel(ball_time)

        #assign calculated properties to key
        key['treadmill_time'] = ball_time
        key['treadmill_raw'] = ball_raw
        key['treadmill_vel'] = ball_vel

        #insert and notify user
        self.insert1(key)
        self.notify({k:key[k] for k in self.heading.primary_key})

    def notify(self, key):
        msg = 'behavior.Treadmill for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg)


