import h5py
import numpy as np
from pipeline.exceptions import PipelineException
from pipeline.utils.eye_tracking import ANALOG_PACKET_LEN
from scipy.interpolate import InterpolatedUnivariateSpline as iu_spline

def ts2sec(ts, packet_length=0, samplingrate=1e7):
    """
    Convert 10MHz timestamps from Saumil's patching program (ts) to seconds (s)

    :param ts: timestamps
    :param packet_length: length of timestamped packets
    :returns:
        timestamps converted to seconds
        system time (in seconds) of t=0
        bad camera indices from 2^31:2^32 in camera timestamps prior to 4/10/13
    """
    ts = ts.astype(float)

    # find bad indices in camera timestamps and replace with linear est
    bad_idx = ts == 2 ** 31 - 1
    if bad_idx.sum() > 10:
        raise PipelineException('Bad camera ts...')
        x = np.where(~bad_idx)[0]
        x_bad = np.where(bad_idx)[0]
        f = iu_spline(x, ts[~bad_idx], k=1)
        ts[bad_idx] = f(x_bad)

    # remove wraparound
    wrap_idx = np.where(np.diff(ts) < 0)[0]
    while not len(wrap_idx) == 0:
        ts[wrap_idx[0] + 1:] += 2 ** 32
        wrap_idx = np.where(np.diff(ts) < 0)[0]

    s = ts / samplingrate

    # Remove offset, and if not monotonically increasing (i.e. for packeted ts), interpolate
    if np.any(np.diff(s) <= 0):
        # Check to make sure it's packets
        diffs = np.where(np.diff(s) > 0)[0]
        assert packet_length == diffs[0] + 1

        # Interpolate
        not_zero = np.hstack((0, diffs + 1))
        f = iu_spline(not_zero, s[not_zero], k=1)
        s = f(np.arange(len(s)))

    return s, bad_idx


def read_video_hdf5(hdf_path):
    """
    Reads hdf5 file for eye tracking

    :param hdf_path: path of the file. Needs a %d where multiple files differ.
    :return: dictionary with the data
    """
    data = {}
    with h5py.File(hdf_path, 'r', driver='family', memb_size=0) as fid:
        data['version'] = fid.attrs['Version']
        if float(fid.attrs['Version']) == 2.:
            data['ball'] = np.asarray(fid['Wheel']).T
            wf = np.asarray(np.asarray(fid['Analog Signals'])).T
            data['framenum_ts'] = np.asarray(fid['framenum_ts']).squeeze()
            data['trialnum_ts'] = np.asarray(fid['trialnum_ts']).squeeze()
            data['eyecam_ts'] = np.asarray(fid['videotimestamps']).squeeze()
            data['syncPd'] = wf[:, 0]  # flip photo diode
            data['scanImage'] = wf[:, 1]
            data['ts'] = wf[:, 2]
            data['analogPacketLen'] = float(fid.attrs['AS_samples_per_channel'])

        elif float(fid.attrs['Version']) == 1.:
            data['ball'] = np.asarray(fid['ball']).T
            wf = np.asarray(np.asarray(fid['waveform'])).T
            data['cam1ts'] = np.asarray(fid['behaviorvideotimestamp']).squeeze()
            data['cam2ts'] = np.asarray(fid['eyetrackingvideotimestamp']).squeeze()
            data['syncPd'] = wf[:, 2]  # flip photo diode
            data['scanImage'] = wf[:, 9]
            data['ts'] = wf[:, 10]
            data['analogPacketLen'] = ANALOG_PACKET_LEN
        else:
            print('File version not known')

    return data


