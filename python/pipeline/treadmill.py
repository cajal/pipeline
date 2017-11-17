import datajoint as dj
import numpy as np
from commons import lab

from .utils.signal import spaced_max, longest_contiguous_block
from .utils.h5 import read_video_hdf5, ts2sec
from . import experiment, notify


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

