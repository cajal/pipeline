import os
import scanreader
import numpy as np
import datajoint as dj
from commons import lab
from pipeline.exceptions import PipelineException
from pipeline import reso, experiment, mice, notify, shared
from pipeline.utils import galvo_corrections, signal, quality, mask_classification, performance, h5

# external storage place
dj.config['external'] = {'protocol': 'file',
                         'location': '/mnt/dj-stor01/pipeline-externals'}

schema = dj.schema("pipeline_vreso", locals(), create_tables=False)
CURRENT_VERSION = 1


@schema
class Amplifier(dj.Lookup):
    definition = """ # Electrophysiology rig setup
    amplifier                     : varchar(32)         # name and model of patch rig
    ---
    description                   : varchar(256)        # description of patch rig
    """
    contents = [
        ['Axoclamp 2B', ''],
        ['NPI ELC-03XS', '']
    ]


@schema
class PatchSession(dj.Manual):
    definition = """ # Single session of patch clamp experiment
    -> mice.Mice
    patch_session                 : smallint unsigned   # session index for the mouse
    ---
    -> Amplifier
    -> experiment.Rig
    -> experiment.Person
    recording_path = ''           : varchar(256)        # path to folder containing all h5 files for the given session
    session_ts=CURRENT_TIMESTAMP  : timestamp           # time of session
    """


@schema
class Cell(dj.Manual):
    definition = """ # Single cell patched during ephys experiment
    -> PatchSession
    cell_id                       : smallint unsigned   # cell index for patch session
    ---
    cell_notes = ''               : varchar(256)
    """


@schema
class Recording(dj.Manual):
    definition = """ # Single recording of a cell
    -> Cell
    recording_id                  : smallint unsigned   # recording index for patch session
    ---
    clamp_type                    : enum('voltage','current')
    patch_type                    : enum('whole', 'loose')
    file_name = ''                : varchar(128)        # filename of h5 file containing recording data
    patch_ts=CURRENT_TIMESTAMP    : timestamp           # time of recording
    """


@schema
class RecordingInfo(dj.Imported):
    definition = """ # Imported information of a single recording of a cell
    -> Recording
    ---
    voltage                       : external            # (V_amplifier) recorded voltage
    current                       : external            # (V_amplifier) recorded current
    command                       : external            # (V_amplifier) output voltage/current into cell
    patch_times                   : external            # (seconds) ephys recording times on timestamp/master clock
    frame_times                   : external            # (seconds) scanimage times on timestamp/master clock
    vgain                         : float               # (mV_cell/V_amplifier) gain applied to recorded voltage
    igain                         : float               # (nA_cell/V_amplifier) gain applied to recorded current
    command_gain                  : float               # ([cell_nA or cell_mV]/V_computer) gain applied to command output
    voltage_lowpass=null          : float               # (Hz) lowpass filter applied to voltage
    voltage_highpass=null         : float               # (Hz) highpass filter applied to voltage
    current_lowpass=null          : float               # (Hz) lowpass filter applied to current
    """

    def make(self, key):
        """ Read ephys data and insert into table """
        import h5py

        # Read the scan
        print('Reading file...')
        vreso_path, filename_base = (PatchSession * (Recording() & key)).fetch1('recording_path', 'file_name')
        local_path = lab.Paths().get_local_path(vreso_path)
        filename = os.path.join(local_path, filename_base + '_%d.h5')
        with h5py.File(filename, 'r', driver='family', memb_size=0) as f:

            # Load timing info
            ANALOG_PACKET_LEN = f.attrs['waveform Frame Size'][0]

            # Get counter timestamps and convert to seconds
            patch_times = h5.ts2sec(f['waveform'][10, :], is_packeted=True)

            # Detect rising edges in scanimage clock signal (start of each frame)
            binarized_signal = f['waveform'][9, :] > 2.7  # TTL voltage low/high threshold
            rising_edges = np.where(np.diff(binarized_signal.astype(int)) > 0)[0]
            frame_times = patch_times[rising_edges]

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
                    num_missing_points = int(round((frame_times[stop] - frame_times[start]) / frame_period - 1))
                    correct_fts.extend(np.linspace(frame_times[start], frame_times[stop], num_missing_points + 2)[1:-1])
                correct_fts.extend(frame_times[nan_limits[-1]:])
                frame_times = np.array(correct_fts)

                # Record the NaN fix
                num_gaps = int(len(nan_limits) / 2)
                nan_length = sum(nan_limits[1::2] - nan_limits[::2]) * frame_period  # secs


            ####### WARNING: FRAME INTERVALS NOT ERROR CHECKED - TEMP CODE #######
            # Check that frame times occur at the same period
            frame_intervals = np.diff(frame_times)
            frame_period = np.median(frame_intervals)
            #if np.any(abs(frame_intervals - frame_period) > 0.15 * frame_period):
            #    raise PipelineException('Frame time period is irregular')


            # Drop last frame time if scan crashed or was stopped before completion
            valid_times = ~np.isnan(patch_times[rising_edges[0]: rising_edges[-1]])  # restricted to scan period
            binarized_valid = binarized_signal[rising_edges[0]: rising_edges[-1]][valid_times]
            frame_duration = np.mean(binarized_valid) * frame_period
            falling_edges = np.where(np.diff(binarized_signal.astype(int)) < 0)[0]
            last_frame_duration = patch_times[falling_edges[-1]] - frame_times[-1]
            if (np.isnan(last_frame_duration) or last_frame_duration < 0 or
                    abs(last_frame_duration - frame_duration) > 0.15 * frame_duration):
                frame_times = frame_times[:-1]

            ####### WARNING: NO CORRECTION APPLIED - TEMP CODE #######
            voltage = np.array(f['waveform'][1, :], dtype='float32')
            current = np.array(f['waveform'][0, :], dtype='float32')
            command = np.array(f['waveform'][5, :], dtype='float32')

            ####### WARNING: DUMMY VARIABLES - TEMP CODE #######
            vgain = 0
            igain = 0
            command_gain = 0

            self.insert1({**key, 'voltage': voltage, 'current': current, 'command': command, 'patch_times': patch_times,
                          'frame_times': frame_times, 'vgain': vgain, 'igain': igain, 'command_gain': command_gain})


@schema
class PatchSpikes(dj.Computed):
    definition = """
        -> RecordingInfo
        ---
        spike_ts                  : external            # (seconds) array of times spikes occurred on timestamp/master clock
        """


@schema
class ManualPatchSpikes(dj.Manual):
    definition = """
        -> RecordingInfo
        ---
        spike_ts                  : external            # (seconds) array of times spikes occurred on timestamp/master clock
        method_notes = ''         : varchar(256)
        """


@schema
class ResoMatch(dj.Manual):
    definition = """ # Match between Patch Recording and Segmented Cell
    -> Recording
    segmentation_method             : smallint          # segmentation method
    ---
    -> reso.Segmentation.Mask
    """
