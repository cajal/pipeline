import h5py
import numpy as np
from ..exceptions import PipelineException
from .eye_tracking import ANALOG_PACKET_LEN

def ts2sec(ts, sampling_rate=1e7, is_packeted=False):
    """ Convert timestamps from 2pMaster (ts) to seconds (s)

    :param np.array ts: Timestamps from behavior files.
    :param float sampling_rate: Sampling rate of the master clock counter in Hz.
    :param bool is_packeted: Whether data was recorded in packets of samples with the
        same timestamp.

    :returns: Timestamps converted to seconds.
    """
    # Remove wrap around
    ts = np.array(ts, np.float64) # copy to avoid overwriting input
    for wrap_idx in np.where(np.diff(ts) < 0)[0]:
        ts[wrap_idx + 1: ] += 2 ** 32

    # Convert counter timestamps to secs
    ts_secs = ts / sampling_rate

    if is_packeted:
        # Check that recorded packet sizes have equal length
        packet_limits = np.where(np.diff([-float('inf'), *ts, float('inf')]))[0]
        recorded_packet_sizes = np.diff(packet_limits)
        if not np.all(recorded_packet_sizes == recorded_packet_sizes[0]):
            raise PipelineException('Unequal packet sizes in signal.')
        packet_size = recorded_packet_sizes[0] # number of continuous samples with the same timestamp in ts.

        # Resample timepoints between packets
        expected_length = np.median(np.diff(ts_secs[packet_limits[:-1]])) # secs between packets
        xs = np.array([*range(0, len(ts_secs), packet_size), len(ts_secs)])
        ys = np.array([*ts_secs[xs[:-1]], ts_secs[-1] + expected_length])
        if np.any(abs(np.diff(ys) - expected_length) > expected_length * 0.15):
            abnormal_tss = sum(abs(np.diff(ys) - expected_length) > expected_length * 0.15)
            msg = 'Unequal spacing between {} continuous packets'.format(abnormal_tss)
            raise PipelineException(msg)
        ts_secs = np.interp(range(len(ts_secs)), xs, ys)

    return ts_secs


def read_video_hdf5(hdf5_path):
    """ Reads hdf5 file with timestamps and analog signals.

    :param hdf5_path: path of the file. Needs a %d where multiple files differ.

    :returns: A dictionary with these fields:
        version: string. File version
        analogPacketLen: int. Number of samples received in a single packet coming from
            analog channels. See ts for why this is relevant.
        ts: 1-d array. Timestamps for analog sync signals (photodiode and ScanImage) in
            master clock time. All samples in the same packet are given the same
            timestamp: the clock time when they arrived.
        syncPd: 1-d array. Photodiode current in volts at 10 KHz.
        scanImage: 1-d array. ScanImage frame clock signal in volts at 10 KHz; high value
            (~5 V) during aqcquisiton.
        wheel: np.array (2 x num_timepoints) .
            wheel[0] Wheel position (as a counter value that wraps around 2 **32 -1)
                recorded at 100 Hz.
            wheel[1] Timestamps in master clock time of each sample point.
            New to '2.x':
            wheel[1] Timestamps in  seconds since some reference time.

        ## Specific to version '1.0'
        cam1_ts: 1-d vector. Timestamps for each video frame of camera 1 in master clock time.
        cam2_ts: 1-d vector. Timestamps for each video frame of camera 2 in master clock time.

        ## Specific to versions '2.x'
        framenum_ts: np.array (2 x num_scanimage_slices / 30)
            framenum_ts[0]: Counter starting at 1 and records every 30.
            framenum_ts[1]: Timestamps in master clock time.
        trialnum_ts: np.array (2 x num_stimulus_trials)
            trialnum_ts[0]: Counter starting at 1 for first run of session.
            trialnum_ts[1]: Timestamps in master clock time.
        eyecam_ts: np.array (2 x num_video_frames)
            eyecam_ts[0]: Timestamps in master clock time.
            eyecam_ts[1]: Timestamps in seconds since some reference time.

    ..note:: Master clock time is an integer counter that increases every 0.1 usecs and
        wraps around at 2 ** 32 - 1.
    """
    with h5py.File(hdf5_path, 'r', driver='family', memb_size=0) as f:
        file_version = str(f.attrs['Version'][0])

        data = {'version': file_version}
        if file_version == '1.0':
            data['analogPacketLen'] = ANALOG_PACKET_LEN
            data['wheel'] = np.array(f['ball'])
            data['cam1_ts'] = np.array(f['behaviorvideotimestamp']).squeeze()
            data['cam2_ts'] = np.array(f['eyetrackingvideotimestamp']).squeeze()

            analog_signals = np.array(f['waveform'])
            data['syncPd'] = analog_signals[2]
            data['scanImage'] = analog_signals[9]
            data['ts'] = analog_signals[10]

        elif file_version in ['2.0', '2.1']:
            data['analogPacketLen'] = int(f.attrs['AS_samples_per_channel'][0])
            data['wheel'] = np.array(f['Wheel'])
            data['framenum_ts'] = np.array(f['framenum_ts'])
            data['trialnum_ts'] = np.array(f['trialnum_ts'])
            data['eyecam_ts'] = np.array(f['videotimestamps'])

            analog_signals = np.array(f['Analog Signals'])
            channel_names = f.attrs['AS_channelNames'].decode('ascii').split(',')
            channel_names = [cn.strip() for cn in channel_names]
            data['syncPd'] = analog_signals[channel_names.index('Photodiode')]
            data['scanImage'] = analog_signals[channel_names.index('ScanImageFrameSync')]
            data['ts'] = analog_signals[channel_names.index('Time')]

        else:
            msg = 'Wrong file version {} in file {}'.format(file_version, hdf5_path)
            raise PipelineException(msg)

    return data