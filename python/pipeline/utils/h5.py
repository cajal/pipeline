import h5py
import numpy as np
from ..exceptions import PipelineException
from .eye_tracking import ANALOG_PACKET_LEN
from .signal import mirrconv, spaced_max


def read_behavior_file(filename):
    """ Reads hdf5 file with timestamps and analog signals.

    :param filename: path of the file. Needs a %d where multiple files differ.

    :returns: A dictionary with these fields:
        version: string. File version
        analogPacketLen: int. Number of samples received in a single packet coming from
            analog channels. See ts for why this is relevant.
        ts: 1-d array (num_samples). Timestamps for analog sync signals (photodiode,
            scanimage and temperature) in master clock time. All samples in the same
            packet are given the same timestamp: the clock time when they arrived.
        syncPd: 1-d array (num_samples). Photodiode current in volts at 10 KHz.
        scanImage: 1-d array (num_samples). ScanImage frame clock signal in volts at
            10 KHz; high value (~5 V) during aqcquisiton.
        wheel: np.array (2 x num_timepoints) .
            wheel[0] Wheel position (as a counter value that wraps around 2 **32 -1)
                recorded at 100 Hz.
            wheel[1] Timestamps in master clock time of each sample point.
            New to '2.x':
            wheel[2] Timestamps in  seconds since some reference time.

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
        ### Optional:
        posture_ts: np.array (2 x num_video_frames)
            posture_ts[0]: Timestamps in master clock time.
            posture_ts[1]: Timestamps in seconds since some reference time.
        temperature: np.array (num_samples). Temperature in Fahrenheit degrees / 100 at 10
            KHz.

    ..note:: Master clock time is an integer counter that increases every 0.1 usecs and
        wraps around at 2 ** 32 - 1.
    """
    with h5py.File(filename, 'r', driver='family', memb_size=0) as f:
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
            if 'videotimestamps_posture' in f:
                data['posture_ts'] = np.array(f['videotimestamps_posture'])

            analog_signals = np.array(f['Analog Signals'])
            channel_names = f.attrs['AS_channelNames'].decode('ascii').split(',')
            channel_names = [cn.strip() for cn in channel_names]
            data['syncPd'] = analog_signals[channel_names.index('Photodiode')]
            data['ts'] = analog_signals[channel_names.index('Time')]
            if 'Temperature' in channel_names:
                data['temperature'] = analog_signals[channel_names.index('Temperature')]
            
            if 'AudioStimWindow' in channel_names:
                data['auditory'] = analog_signals[channel_names.index('AudioStimWindow')]

            if str(f.attrs['AS_Version'][0]) == '2.1':
                data['scanImage'] = analog_signals[channel_names.index('ScanImageFrameSync')]
            else:
                data['scanImage'] = analog_signals[channel_names.index('FrameSync')]

        else:
            msg = 'Unknown file version {} in file {}'.format(file_version, filename)
            raise PipelineException(msg)

    return data


def ts2sec(ts, sampling_rate=1e7, is_packeted=False):
    """ Convert timestamps from master clock (ts) to seconds (s)

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
        ys = np.array([ts_secs[0] - expected_length, *ts_secs[xs[:-1]]])
        sample_xs = np.arange(len(ts_secs))
        ts_secs = np.interp(sample_xs, xs, ys)

        # Invalidate timepoints with unequal spacing between packets
        if np.any(abs(np.diff(ys) - expected_length) > 0.1 * expected_length):
            abnormal_diffs = abs(np.diff(ys) - expected_length) > 0.1 * expected_length
            abnormal_limits = np.where(np.diff([0, *abnormal_diffs, 0]))[0]
            for start, stop in zip(abnormal_limits[::2], abnormal_limits[1::2]):
                abnormal_indices = np.logical_and(sample_xs > xs[start], sample_xs < xs[stop])
                ts_secs[abnormal_indices] = float('nan')

            print('Warning: Unequal spacing between continuos packets: {} abnormal gap(s)'
                  ' detected. Signal will have NaNs.'.format(len(abnormal_limits) // 2))

    return ts_secs


def read_imager_file(filename):
    """ Reads hdf5 file with Intrinsic Imager data saved by Labview.

    :param filename: path of the file.

    :returns: A dictionary with these fields:
        photodiode: 1-d array. Photodiode current in volts at 10 KHz.
        photodiode_fps: Photodiode sampling rate in Hz.
        scan: 3-d array. Scan (height x width x num_frames).
        scan_fps: Framerate of scan in Hz.
    """
    with h5py.File(filename, 'r') as f:
        data = {}
        for group in f.values():
            if group['id'].attrs['name'].decode() == 'ephys':
                for trace in group['traces'].values():
                    if trace.attrs['name'].decode() == 'photodiode':
                        data['photodiode'] = np.array(trace['y-axis/data_vector/data'])
                    if trace.attrs['name'].decode() == 'hz':
                        data['photodiode_fps'] = trace['y-axis/data_vector/data'][0]

            if group['id'].attrs['name'].decode() == 'imaging':
                for trace in group['traces'].values():
                    if trace.attrs['name'].decode() == 'movie':
                        raw_scan = np.array(trace['y-axis/data_vector/data'], dtype=np.float32)
                    if trace.attrs['name'].decode() == 'hz':
                        data['scan_fps'] = trace['y-axis/data_vector/data'][0]
                    if trace.attrs['name'].decode() == 'x':
                        width = int(trace['y-axis/data_vector/data'][0])
                    if trace.attrs['name'].decode() == 'y':
                        height = int(trace['y-axis/data_vector/data'][0])
                data['scan'] = raw_scan.reshape([height, -1, width]).swapaxes(1, 2)

    return data


def find_flips(signal, fps, monitor_fps):
    """ Detect 'flips' in the photodiode signal and decode encoded numbers.

    Photodiode is pointed to a square in the monitor that changes each frame. Valid
    intensities are black (baseline), gray (middle amplitude representing a 1) and white
    (high amplitude representing a 0); e.g., bgbwbgbgbwb = 10110. This binary string
    encodes consecutive 16-bit integers (0-65535). Each integer is encoded in reverse
    order so, 001 would be 4 (100).

    Each decoded number represents 32 flips (16 encoding flips and 16 black flips) which
    are assigned numbers between 32 * num + 1 to 32 * (num + 1).

    :param np.array signal: Photodiode signal.
    :param float fps: Sampling rate of photodiode signal (Hz).
    :param float monitor_fps: Framerate of the monitor (Hz).

    :returns indices of each detected flip (0-based) and their decoded flip number.
    """
    # Find flips (works very well even if there is more than one (monitor) frame per flip)
    samples_per_frame = fps / monitor_fps
    flip_detector = np.sin(np.linspace(0, 2 * np.pi, 2 * int(round(samples_per_frame)) + 1))
    filtered = mirrconv(signal, flip_detector)
    flip_indices = spaced_max(abs(filtered), 0.5 * samples_per_frame)

    # Remove flips with filtered amplitudes close to zero
    bad_flips = abs(filtered[flip_indices]) < 0.15 * np.percentile(abs(filtered[flip_indices]), 99.9)
    flip_indices = flip_indices[~bad_flips]
    flip_amps = filtered[flip_indices]

    # Compute gray:white (0:1) threshold
    thresh = np.percentile(flip_amps[flip_amps > 0], (10, 90)).mean()

    # Decode numbers encoded in the binary sequence of flips
    flip_nums = np.full_like(flip_amps, np.nan)
    num_consecutive_bins = 5 # number of consecutive bins to check to assert data is correct
    i = 0
    while i < len(flip_amps) - 32 * num_consecutive_bins:
        next_amps = flip_amps[i: i + 32 * num_consecutive_bins: 2] # skip black frames
        if np.all(next_amps > 0): # black flips will have negative amplitude
            # Decode numbers
            bits = np.reshape(next_amps < thresh, [num_consecutive_bins, 16])
            nums = [int(''.join(map(str, reversed(b.astype(int)))), 2) for b in bits]

            # If numbers are sequential fill flip_nums
            if np.all(np.diff(nums) == 1):
                flip_nums[i: i + 32 * num_consecutive_bins] = (32 * nums[0] +
                    np.arange(1, 32 * num_consecutive_bins + 1))

                i = i + 32 * num_consecutive_bins # skip to next set of 5
                continue
        i += 1

    # Remove flips with no decoded number
    flip_indices = flip_indices[~np.isnan(flip_nums)]
    flip_nums = flip_nums[~np.isnan(flip_nums)].astype(int)

    return flip_indices, flip_nums


def read_digital_olfaction_file(filename):
    """ Reads hdf5 files with olfaction digital signals.
    :param filename: path of the file. Needs a %d where 2GB file split counter is located.
    :returns: A dictionary with these fields:
        version: string. File version
        delay: float. Time in milliseconds between puffs of odor
        duration: float. Duration of odor presentation in milliseconds.
        valves: 1-d array (num_samples). State of valve stored as decimal of binary
            representation. 0 = Closed, 1 = Open.
            ex. 2028 = 0b100000100000 = 12th and 6th valve are open.
        ts: 1-d array (num_samples). Timestamps for valve state changes in master clock
            time.
    ..note:: Master clock time is an integer counter that increases every 0.1 usecs.
    """
    with h5py.File(filename, 'r', driver='family', memb_size=0) as f:

        file_version = str(f.attrs['Version'][0])
        data = {'version': file_version}

        if file_version in ['1.0']:
            data['delay'] = f.attrs['Delay'][0]
            data['duration'] = f.attrs['Puff Duration'][0]
            data['valves'] = f['Digital Pattern'][0, :]
            data['ts'] = f['Digital Pattern'][1, :]

        else:
            msg = 'Unknown file version {} in file {}'.format(file_version, filename)
            raise PipelineException(msg)

        return data


def read_analog_olfaction_file(filename):
    """ Reads hdf5 files with olfaction analog signals.
    :param filename: path of the file. Needs a %d where 2GB file split counter is located.
    :returns: A dictionary with these fields:
        version: string. File version
        analogPacketLen: int. Number of samples received in a single packet coming from
            analog channels. See ts for why this is relevant.
        fs: sampling rate  in Hz
        breath: 1-d array (num_samples). Respiration recording.
        ts: 1-d array (num_samples). Timestamps for analog signals in master clock time.
            All samples in the same packet are given the same timestamp: the clock time
            when they arrived.
        scanImage: 1-d array (num_samples). ScanImage frame clock signal in volts at
            10 KHz; high value (~5 V) during aqcquisiton.
    ..note:: Master clock time is an integer counter that increases every 0.1 usecs.
    """

    with h5py.File(filename, 'r', driver='family', memb_size=0) as f:

        file_version = str(f.attrs['Version'][0])
        data = {'version': file_version}

        if file_version in ['1.0']:
            data['analogPacketLen'] = f.attrs['waveform Frame Size'][0]
            data['fs'] = f.attrs['waveform Fs'][0]
            data['breath'] = f['waveform'][1, :]
            data['ts'] = f['waveform'][2, :]
            data['scanImage'] = f['waveform'][0, :]

        else:
            msg = 'Unknown file version {} in file {}'.format(file_version, filename)
            raise PipelineException(msg)

        return data