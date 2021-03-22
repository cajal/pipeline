import numbers
import numpy as np
import datajoint as dj
from scipy import interpolate
from stimulus import stimulus
from pipeline.exceptions import PipelineException
from pipeline import treadmill, fuse, shared, odor

pupil = dj.create_virtual_module("pipeline_eye", "pipeline_eye")


def find_idx_boundaries(indices, drop_single_idx=False):
    """
    Given a flatten list/array of indices, break list into a list of lists of indices incrementing by 1

        Example:
            >>>find_idx_boundaries([1,2,3,4,501,502,503,504])
               return value: [[1,2,3,4],[501,502,503,504]]

        Parameters:
            indices: Flattened list or numpy array of indices to break apart into sublists
            drop_single_idx: Boolean which sets if single indices not part of any sublist
                             should be dropped or raise an error upon detection.

        Returns:
            events: List of lists of indices that are incrementing by 1

    """

    ## Since list slicing counts up to but not including ends, we need to add 1 to all detected end locations
    ends = np.where(np.diff(indices) > 1)[0] + 1

    ## Starts and ends will equal each other since list slicing includes start values, but start needs 0 appended
    starts = np.copy(ends)
    if len(starts) == 0 or starts[0] != 0:
        starts = np.insert(starts, 0, 0)

    ## np.diff returns an array one smaller than the indices list, so we need to add the last idx to the ends
    if len(ends) == 0 or ends[-1] != len(indices):
        ends = np.insert(ends, len(ends), len(indices))

    ## Loop through all continuous idx start & end to see if any are too small (length = 1)
    events = []
    for start, end in zip(starts, ends):
        if end - start < 2:
            if not drop_single_idx:
                raise PipelineException(f"Disconnected index found at index {start}")
        else:
            events.append(indices[start:end])

    return events


def find_time_boundaries(indices, times, drop_single_idx=False):
    """
    Given a flatten list/array of indices and timings, return the start & stop times for all idx clusters
    incrementing by 1. Start/stop times are taken to be non-NaN min/max times during the idx fragment.

        Example:
            >>>find_time_boundaries(indices=[1,2,3,501,502,503],times=[0.1,0.2,0.3,0.4,0.5,0.6])
               return value: [[0.1,0.3], [0.4,0.6]]

        Parameters:
            indices: Flattened list or numpy array of indices to break apart into incrementing fragments
            times: Flattened list or numpy array of times of recording. Can either be the same size as
                   the list of indices (only the times corresponding to those indices) or can be the
                   full list of all times recordings happened. If full list, indices are assumed to index
                   their own recording time in the times array/list.
            drop_single_idx: Boolean which sets if single indices not part of any sublist
                             should be dropped or raise an error upon detection.

        Returns:
            events: List of lists of start & stop times for each incrementing by 1 idx fragments

    """

    ## If times are not the same size as indices, assume these times are for all recordings
    ## and the recording time for IDX NUM is times[NUM] (ie. idx 5 was recorded at times[5])
    if len(times) != len(indices):
        times = np.array(times)[np.array(indices)]

    ## Since list slicing counts up to but not including ends, we need to add 1 to all detected end locations
    ends = np.where(np.diff(indices) > 1)[0] + 1

    ## Starts and ends will equal each other since list slicing includes start values, but start needs 0 appended
    starts = np.copy(ends)
    if len(starts) == 0 or starts[0] != 0:
        starts = np.insert(starts, 0, 0)

    ## np.diff returns an array one smaller than the indices list, so we need to add the last idx to the ends
    if len(ends) == 0 or ends[-1] != len(indices):
        ends = np.insert(ends, len(ends), len(indices))

    ## Loop through all continuous idx start & end to see if any are too small (length = 1)
    time_boundaries = []
    for start, end in zip(starts, ends):
        if end - start < 2:
            if not drop_single_idx:
                raise PipelineException(f"Disconnected index found at index {start}")
        else:
            bounds = [np.nanmin(times[start:end]), np.nanmax(times[start:end])]
            time_boundaries.append(bounds)

    return time_boundaries


def fetch_timing_data(
    scan_key: dict,
    source_type: str,
    target_type: str,
    debug: bool = True,
):

    ##
    ## Set pipe, error check scan_key, and fetch field offset
    ##

    ## Define the pipe (meso/reso) to use
    if len(fuse.MotionCorrection & scan_key) == 0:
        msg = f"scan_key {scan_key} not found in fuse.MotionCorrection."
        raise PipelineException(msg)
    pipe = (fuse.MotionCorrection & scan_key).module

    ## Make strings lowercase and process indices
    source_type = source_type.lower()
    target_type = target_type.lower()

    ## Set default values for later processing
    field_offset = 0
    slice_num = 1
    ms_delay = 0

    ## Determine if source or target type requires extra scan info
    scan_types = (
        "fluorescence-stimulus",
        "fluorescence-behavior",
        "deconvolution-stimulus",
        "deconvolution-behavior",
    )
    if source_type in scan_types or target_type in scan_types:

        ## Check scan_key defines a unique scan
        if len(pipe.ScanInfo & scan_key) != 1:
            msg = (
                f"scan_key {scan_key} does not define a unique scan. "
                f"Matching scans found: {len(fuse.MotionCorrection & scan_key)}"
            )
            raise PipelineException(msg)

        ## Check a single field is defined by scan_key
        if len(pipe.ScanInfo.Field & scan_key) != 1:
            msg = (
                f"scan_key {scan_key} must specify a single field when source or target type is set "
                f"to 'scan'. Matching fields found: {len(pipe.ScanInfo.Field & scan_key)}"
            )
            raise PipelineException(msg)

        ## Determine field offset to slice times later on and set ms_delay to field average
        scan_restriction = (pipe.ScanInfo & scan_key).fetch("KEY")
        all_z = np.unique(
            (pipe.ScanInfo.Field & scan_restriction).fetch("z", order_by="field ASC")
        )
        slice_num = len(all_z)
        field_z = (pipe.ScanInfo.Field & scan_key).fetch1("z")
        field_offset = np.where(all_z == field_z)[0][0]
        if debug:
            print(f"Field offset found as {field_offset} for depths 0-{len(all_z)}")

        field_delay_im = (pipe.ScanInfo.Field & scan_key).fetch1("delay_image")
        average_field_delay = np.mean(field_delay_im)
        ms_delay = average_field_delay
        if debug:
            print(
                f"Average field delay found to be {round(ms_delay,4)}ms. This will be used unless a unit is specified in the key."
            )

        ## If included, add unit offset
        if "unit_id" in scan_key or "mask_id" in scan_key:
            if len(pipe.ScanSet.Unit & scan_key) > 0:
                unit_key = (pipe.ScanSet.Unit & scan_key).fetch1()
                ms_delay = (pipe.ScanSet.UnitInfo & unit_key).fetch1("ms_delay")
                if debug:
                    print(
                        f"Unit found with delay of {round(ms_delay,4)}ms. Delay added to relevant times."
                    )
            else:
                if debug:
                    print(
                        f"Warning: ScanSet.Unit is not populated for the given key! Using field offset minimum instead."
                    )

    ##
    ## Fetch source and target sync data
    ##

    ## Define a lookup for data sources. Key values are in (data_table, column_name) tuples.
    data_source_lookup = {
        "fluorescence-stimulus": (stimulus.Sync, "frame_times"),
        "deconvolution-stimulus": (stimulus.Sync, "frame_times"),
        "fluorescence-behavior": (stimulus.BehaviorSync, "frame_times"),
        "deconvolution-behavior": (stimulus.BehaviorSync, "frame_times"),
        "treadmill": (treadmill.Treadmill, "treadmill_time"),
        "pupil": (pupil.Eye, "eye_time"),
        "respiration": (odor.Respiration * odor.MesoMatch, "times"),
    }

    ## Error check inputs
    if source_type not in data_source_lookup or target_type not in data_source_lookup:
        msg = (
            f"source and target type combination '{source_type}' and '{target_type}' not supported. "
            f"Valid values are 'scan-behavior', 'scan-stimulus', 'treadmill', 'respiration' or 'pupil'."
        )
        raise PipelineException(msg)

    ## Fetch source and target times using lookup dictionary
    source_table, source_column = data_source_lookup[source_type]
    source_times = (source_table & scan_key).fetch1(source_column).squeeze()

    target_table, target_column = data_source_lookup[target_type]
    target_times = (target_table & scan_key).fetch1(target_column).squeeze()

    ##
    ## Timing corrections
    ##

    ## Slice times if on ScanImage clock and add delay (scan_types defined near top)
    if source_type in scan_types:
        source_times = source_times[field_offset::slice_num] + ms_delay
    if target_type in scan_types:
        target_times = target_times[field_offset::slice_num] + ms_delay

    ##
    ## Interpolate into different clock if necessary
    ##

    clock_type_lookup = {
        "fluorescence-stimulus": "stimulus",
        "deconvolution-stimulus": "stimulus",
        "fluorescence-behavior": "behavior",
        "deconvolution-behavior": "behavior",
        "pupil": "behavior",
        "processed-pupil": "behavior",
        "treadmill": "behavior",
        "processed-treadmill": "behavior",
        "respiration": "odor",
    }

    sync_conversion_lookup = {
        "stimulus": stimulus.Sync,
        "behavior": stimulus.BehaviorSync,
        "odor": odor.OdorSync * odor.MesoMatch,
    }

    source_clock_type = clock_type_lookup[source_type]
    target_clock_type = clock_type_lookup[target_type]

    if source_clock_type != target_clock_type:

        interp_source_table = sync_conversion_lookup[source_clock_type]
        interp_target_table = sync_conversion_lookup[target_clock_type]

        interp_source = (interp_source_table & scan_key).fetch1("frame_times").squeeze()
        interp_target = (interp_target_table & scan_key).fetch1("frame_times").squeeze()

        source2target_interp = interpolate.interp1d(
            interp_source, interp_target, fill_value="extrapolate"
        )
        source_times = source2target_interp(source_times)

    return source_times, target_times


def interpolate_signal_data(
    scan_key: dict,
    source_type: str,
    target_type: str,
    source_times,
    target_times,
    debug: bool = True,
):

    ## Pre-format source_type
    source_type = source_type.lower()

    ## Define the pipe (meso/reso) to use
    if len(fuse.MotionCorrection & scan_key) == 0:
        msg = f"scan_key {scan_key} not found in fuse.MotionCorrection."
        raise PipelineException(msg)
    pipe = (fuse.MotionCorrection & scan_key).module

    ## Run helpful error checking
    if source_type == "pupil":
        tracking_method_num = len(
            dj.U("tracking_method") & (pupil.FittedPupil & scan_key)
        )
        if tracking_method_num > 1:
            msg = (
                "More than one pupil tracking method found for entered scan. "
                "Specify tracking_method in scan key (tracking_method=2 for DeepLabCut)."
            )
            raise PipelineException(msg)

    ## Fetch required signal
    ## Note: Pupil requires .fetch() while other signals require .fetch1().
    ##       It is easier to make an if-elif-else structure than a lookup dictionary in this case.
    if source_type in ("fluorescence-stimulus", "fluorescence-behavior"):
        source_signal = (pipe.Fluorescence.Trace & scan_key).fetch1("trace")
    if source_type in ("deconvolution-stimulus", "deconvolution-behavior"):
        unit_key = (pipe.ScanSet.Unit & scan_key).fetch1()
        source_signal = (pipe.Activity.Trace & unit_key).fetch1("trace")
    if source_type == "pupil":
        source_signal = (pupil.FittedPupil.Circle & scan_key).fetch("radius")
    if source_type == "treadmill":
        source_signal = (treadmill.Treadmill & scan_key).fetch1("treadmill_vel")

    ## Calculate FPS to determine if lowpass filtering is needed
    source_fps = 1 / np.nanmedian(np.diff(source_times))
    target_fps = 1 / np.nanmedian(np.diff(target_times))

    ## Fill NaNs to prevent interpolation errors, but store NaNs for later to add back in after interpolating
    target_replace_nans = None  # Use this as a switch to refill things later
    if sum(np.isnan(source_signal)) > 0:
        source_nan_indices = np.isnan(source_signal)
        time_nan_indices = np.isnan(source_times)
        source_replace_nans = np.logical_and(source_nan_indices, ~time_nan_indices)
        if sum(source_replace_nans) > 0:
            target_replace_nans = convert_clocks_idx_to_idx(
                scan_key,
                np.where(source_replace_nans)[0],
                source_type,
                target_type,
                debug=False,
            )
        nan_filler_func = (
            shared.FilterMethod & {"filter_method": "NaN Filler"}
        ).run_filter
        source_signal = nan_filler_func(source_signal)
        if debug:
            biggest_time_gap = np.nanmax(
                np.diff(source_times[np.where(~source_replace_nans)[0]])
            )
            msg = (
                f"Found NaNs in {sum(source_nan_indices)} locations, which corresponds to "
                f"{round(100*sum(source_nan_indices)/len(source_signal),2)}% of total signal. "
                f"Largest NaN gap found: {round(biggest_time_gap, 2)} seconds."
            )
            print(msg)

    ## Lowpass signal if needed
    if target_fps < source_fps:
        if debug:
            msg = (
                f"Source FPS of {round(source_fps,2)} is greater than target FPS {round(target_fps,2)}. "
                f"Hamming lowpass filtering source signal before interpolation"
            )
            print(msg)
        source_signal = shared.FilterMethod._lowpass_hamming(
            signal=source_signal, signal_freq=source_fps, lowpass_freq=target_fps
        )

    ## Timing and recording array lengths can differ slightly if recording was stopped mid-scan. Timings for
    ## the next X depths would be recorded, but fluorescence values would be dropped if all depths were not
    ## recorded. This would mean timings difference shouldn't be more than the number of depths of the scan.
    if len(source_times) < len(source_signal):
        msg = (
            f"More recording values than source time values exist! This should not be possible.\n"
            f"Source time length: {len(source_times)}\n"
            f"Source signal length: {len(source_signal)}"
        )
        raise PipelineException(msg)

    elif len(source_times) > len(source_signal):

        scan_res = pipe.ScanInfo.proj() & scan_key  ## To make sure we select all fields
        z_plane_num = len(dj.U("z") & (pipe.ScanInfo.Field & scan_res))
        if (len(source_times) - len(source_signal)) > z_plane_num:
            msg = (
                f"Extra timing values exceeds reasonable error bounds. "
                f"Error length of {len(source_times) - len(source_signal)} with only {z_plane_num} z-planes."
            )
            raise PipelineException(msg)

        else:
            
            shorter_length = np.min((len(source_times), len(source_signal)))
            source_times = source_times[:shorter_length]
            source_signal = source_signal[:shorter_length]
            if debug:
                length_diff = np.abs(len(source_times) - len(source_signal))
                msg = (
                    f"Source times and source signal show length mismatch within acceptable error."
                    f"Difference of {length_diff} within acceptable bounds of {z_plane_num} z-planes."
                )
                print(msg)

    ## Interpolating source signal into target timings
    signal_interp = interpolate.interp1d(
        source_times, source_signal, bounds_error=False
    )
    interpolated_signal = signal_interp(target_times)
    if target_replace_nans is not None:
        for target_nan_idx in target_replace_nans:
            interpolated_signal[target_nan_idx] = np.nan

    return interpolated_signal


def convert_clocks_idx_to_idx(
    scan_key: dict,
    indices,
    source_type: str,
    target_type: str,
    return_interpolate: bool = False,
    drop_single_idx: bool = True,
    debug: bool = True,
):
    """
    Converts indices of interest on one type of clock/recording to a different one to be used for slicing or processing.

        Parameters:

                scan_key: A dictionary specifying a single scan and/or field. A single field must be defined if requesting
                          a source or target from ScanImage. If key specifies a single unit, unit delay will be added to
                          all timepoints recorded. Single units can be specified via unique mask_id + field or via unit_id.
                          If only field is specified, average field delay will be added.

                indices: List/array of indices to convert or a boolean array with True values at indices of interest.
                         NOTE: Set to None to use the entire source signal length.
                         Indices can be discontinuous fragments (something like 20 indices around a several spikes)
                         and the function will return of list of lists, each containings the corresponding target_idx
                         fragments.
                         ex. [25,26,27..,29,503,504...] OR [False False ... True True True False...] OR None

                source_type: A string specifying what indices you want to convert from. Fluorescence and deconvolution
                             have a dash followed by "behavior" or "stimulus" to refer to which clock you are using.
                             Supported options: 
                                 'fluorescence-stimulus', 'deconvolution-stimulus', ,'fluorescence-behavior',
                                 'deconvolution-behavior', 'pupil', 'treadmill', 'respiration'

                target_type: A string specifying what indices to convert into. Fluorescence and deconvolution have a 
                             dash followed by "behavior" or "stimulus" to refer to which clock you are using.
                             Supported options: 
                                 'fluorescence-stimulus', 'deconvolution-stimulus', ,'fluorescence-behavior',
                                 'deconvolution-behavior', 'pupil', 'treadmill', 'respiration'

                return_interpolate: Boolean with the following behavior
                                     - True: Interpolate source signal onto target clock, return interpolated signal
                                             at indices instead of target indices
                                             NOTE: NaNs in source signal are linearly interpolated over and then
                                                   filled back in after interpolation onto target indices.
                                     - False: Return corresponding indices for target

                drop_single_idx: Boolean with the following behavior
                                     - True: Drop any signal fragments which would create a signal of length 1.
                                     - False: Raise an error and stop if any list of indices leads to a signal
                                              of length 1.
                                    ex. Source IDX [1,2,...,300] on a 500HZ recording will only correspond to
                                        target IDX [1] if target is recorded at 1Hz.

                debug: Set function to print helpful debug text while running


        Returns:

                requested_array: Numpy array of corresponding indices in target type or interpolated source signal 
                                 during target indices.
                                 
        Warnings:
        
                * NaN refilling for source signal will only refill values if NaNs stretch for multiple indices on 
                  target clock
                  
                * Recording points where the time value is NaN are dropped from analysis/processing
    """

    ##
    ## Fetch source and target times, along with converting between Stimulus or Behavior clock if needed
    ##

    source_times, target_times = fetch_timing_data(
        scan_key, source_type, target_type, debug
    )

    ##
    ## Convert indices to a list of numbers if argument equals None or a Boolean mask
    ##

    if indices is None:
        indices = np.arange(len(source_times))
    elif type(indices[0]) == bool:
        indices = np.where(indices)[0]
    else:
        ## Check for duplicates if manually entered
        if len(np.unique(indices)) != len(indices):
            msg = (
                f"Duplicate entries found for provided indice array! "
                f"Try to fix the error or use np.unique() on indices array."
            )
            raise PipelineException(msg)

    ##
    ## Convert source indices to time boundaries, then convert time boundaries into target indices
    ##

    ## Convert indices into start/end times for each continuous fragment (incrementing by 1)
    time_boundaries = find_time_boundaries(indices, source_times, drop_single_idx)
    target_indices = []
    single_idx_count = 0

    ## Loop through start & end times and create list of indices corresponding to that block of time
    for [start, end] in time_boundaries:
        target_idx = np.where(
            np.logical_and(target_times >= start, target_times <= end)
        )[0]
        if len(target_idx) < 2:
            if drop_single_idx:
                single_idx_count += 1
            else:
                msg = (
                    f"Event of length {len(target_idx)} found. "
                    f"Set drop_single_idx to True to suppress these errors."
                )
                raise PipelineException(msg)
        else:
            target_indices.append(target_idx)

    if debug:
        print(f"Indices converted. {single_idx_count} events of length 0 or 1 dropped.")

    ##
    ## Interpolate related signal if requested, else just return the target_indices found.
    ##

    if return_interpolate:

        ## Create full interpolated signal
        interpolated_signal = interpolate_signal_data(
            scan_key, source_type, target_type, source_times, target_times, debug=debug
        )

        ## Split indices given into fragments based on which ones are continuous (incrementing by 1)
        source_signal_fragments = []
        for idx_fragment in target_indices:
            source_signal_fragments.append(interpolated_signal[idx_fragment])

        ## If full signal is converted, remove wrapping list
        if len(source_signal_fragments) == 1:
            source_signal_fragments = source_signal_fragments[0]

        return source_signal_fragments

    else:

        return target_indices
    

def convert_clocks_idx_to_time(
    scan_key: dict,
    indices,
    source_type: str,
    target_type: str,
    return_interpolate: bool = False,
    drop_single_idx: bool = True,
    debug: bool = True,
):
    """
    Converts indices of interest on one type of clock/recording to times on a target clock/recording

        Parameters:

                scan_key: A dictionary specifying a single scan and/or field. A single field must be defined if requesting
                          a source or target from ScanImage. If key specifies a single unit, unit delay will be added to
                          all timepoints recorded. Single units can be specified via unique mask_id + field or via unit_id.
                          If only field is specified, average field delay will be added.

                indices: List/array of indices to convert or a boolean array with True values at indices of interest.
                         NOTE: Set to None to use the entire source signal length.
                         Indices can be discontinuous fragments (something like 20 indices around a several spikes)
                         and the function will return of list of lists, each containings the corresponding target_idx
                         fragments.
                         ex. [25,26,27..,29,503,504...] OR [False False ... True True True False...] OR None

                source_type: A string specifying what indices you want to convert from. Fluorescence and deconvolution
                             have a dash followed by "behavior" or "stimulus" to refer to which clock you are using.
                             Supported options:
                                 'fluorescence-stimulus', 'deconvolution-stimulus', ,'fluorescence-behavior',
                                 'deconvolution-behavior', 'pupil', 'treadmill', 'respiration'

                target_type: A string specifying what times to convert into. Fluorescence and deconvolution have a
                             dash followed by "behavior" or "stimulus" to refer to which clock you are using.
                             Supported options:
                                 'fluorescence-stimulus', 'deconvolution-stimulus', ,'fluorescence-behavior',
                                 'deconvolution-behavior', 'pupil', 'treadmill', 'respiration'

                return_interpolate: Boolean with the following behavior
                                     - True: Interpolate source signal onto target clock, return interpolated signal
                                             at indices instead of target indices
                                             NOTE: NaNs in source signal are linearly interpolated over and then
                                                   filled back in after interpolation onto target times.
                                     - False: Return corresponding times for target

                drop_single_idx: Boolean with the following behavior
                                     - True: Drop any signal fragments which would create a signal of length 1.
                                     - False: Raise an error and stop if any list of indices leads to a signal
                                              of length 1.
                                    ex. Source IDX [1,2,...,300] on a 500HZ recording will only correspond to
                                        target IDX [1] if target is recorded at 1Hz.

                debug: Set function to print helpful debug text while running


        Returns:

                requested_array: Numpy array of corresponding times in target type recording or interpolated
                                 source signal during target times.

        Warnings:

                * NaN refilling for source signal will only refill values if NaNs stretch for multiple indices on
                  target clock

                * Recording points where the time value is NaN are dropped from analysis/processing
    """

    ##
    ## Fetch source and target times, along with converting between Stimulus or Behavior clock if needed
    ##

    source_times, target_times = fetch_timing_data(
        scan_key, source_type, target_type, debug
    )

    ##
    ## Convert indices to a list of numbers if argument equals None or a Boolean mask
    ##

    if indices is None:
        indices = np.arange(len(source_times))
    elif type(indices[0]) == bool:
        indices = np.where(indices)[0]
    else:
        ## Check for duplicates if manually entered
        if len(np.unique(indices)) != len(indices):
            msg = (
                f"Duplicate entries found for provided indice array! "
                f"Try to fix the error or use np.unique() on indices array."
            )
            raise PipelineException(msg)

    ##
    ## Convert source indices to time boundaries, then convert time boundaries into target indices
    ##

    ## Convert indices into start/end times for each continuous fragment (incrementing by 1)
    time_boundaries = find_time_boundaries(indices, source_times, drop_single_idx)
    target_indices = []
    single_idx_count = 0

    ## Loop through start & end times and create list of indices corresponding to that block of time
    for [start, end] in time_boundaries:
        target_idx = np.where(
            np.logical_and(target_times >= start, target_times <= end)
        )[0]
        if len(target_idx) < 2:
            if drop_single_idx:
                single_idx_count += 1
            else:
                msg = (
                    f"Event of length {len(target_idx)} found. "
                    f"Set drop_single_idx to True to suppress these errors."
                )
                raise PipelineException(msg)
        else:
            target_indices.append(target_idx)

    if debug:
        print(f"Indices converted. {single_idx_count} events of length 0 or 1 dropped.")

    ##
    ## Interpolate related signal if requested, else return target times.
    ##

    if return_interpolate:

        ## Create full interpolated signal
        interpolated_signal = interpolate_signal_data(
            scan_key, source_type, target_type, source_times, target_times, debug=debug
        )

        ## Split indices given into fragments based on which ones are continuous (incrementing by 1)
        source_signal_fragments = []
        for idx_fragment in target_indices:
            source_signal_fragments.append(interpolated_signal[idx_fragment])

        ## If full signal is converted, remove wrapping list
        if len(source_signal_fragments) == 1:
            source_signal_fragments = source_signal_fragments[0]

        return source_signal_fragments

    else:

        ## Convert indices to times and return
        source_idx_to_target_times = []

        for target_idx_list in target_indices:
            source_idx_to_target_times.append(target_times[target_idx_list])

        return source_idx_to_target_times


def convert_clocks_time_to_idx(
    scan_key: dict,
    time_boundaries,
    source_type: str,
    target_type: str,
    return_interpolate: bool = False,
    drop_single_idx: bool = True,
    debug: bool = True,
):
    """
    Converts time boundaries of interest on one type of clock/recording to indices on a target recording

        Parameters:

                scan_key: A dictionary specifying a single scan and/or field. A single field must be defined if requesting
                          a source or target from ScanImage. If key specifies a single unit, unit delay will be added to
                          all timepoints recorded. Single units can be specified via unique mask_id + field or via unit_id.
                          If only field is specified, average field delay will be added.

                time_boundaries: List of lists containing [start,stop] boundaries for times of note on source clock.
                                 Start/Stop times are included (>=,<=). These boundaries are converted to all indices
                                 equal to or between recording times on target recording.
                                 NOTE: Set to None to use the entire source signal length.
                                 ex. [[271, 314], [690.321, 800.1]] OR None

                source_type: A string specifying what times you want to convert from. Fluorescence and deconvolution
                             have a dash followed by "behavior" or "stimulus" to refer to which clock you are using.
                             Supported options:
                                 'fluorescence-stimulus', 'deconvolution-stimulus', ,'fluorescence-behavior',
                                 'deconvolution-behavior', 'pupil', 'treadmill', 'respiration'

                target_type: A string specifying what indices to convert into. Fluorescence and deconvolution have a
                             dash followed by "behavior" or "stimulus" to refer to which clock you are using.
                             Supported options:
                                 'fluorescence-stimulus', 'deconvolution-stimulus', ,'fluorescence-behavior',
                                 'deconvolution-behavior', 'pupil', 'treadmill', 'respiration'

                return_interpolate: Boolean with the following behavior
                                     - True: Interpolate source signal onto target clock, return interpolated signal
                                             at indices instead of target indices
                                             NOTE: NaNs in source signal are linearly interpolated over and then
                                                   filled back in after interpolation onto target times.
                                     - False: Return corresponding indices between boundaries for target

                drop_single_idx: Boolean with the following behavior
                                     - True: Drop any signal fragments which would create a signal of length 1.
                                     - False: Raise an error and stop if any list of indices leads to a signal
                                              of length 1.
                                    ex. Source IDX [1,2,...,300] on a 500HZ recording will only correspond to
                                        target IDX [1] if target is recorded at 1Hz.

                debug: Set function to print helpful debug text while running


        Returns:

                requested_array: Numpy array of corresponding times in target type recording or interpolated
                                 source signal during target times.

        Warnings:

                * NaN refilling for source signal will only refill values if NaNs stretch for multiple indices on
                  target clock

                * Recording points where the time value is NaN are dropped from analysis/processing
    """

    ##
    ## Fetch source and target times, along with converting between Stimulus or Behavior clock if needed
    ##

    source_times, target_times = fetch_timing_data(
        scan_key, source_type, target_type, debug
    )

    ##
    ## Check if None is used to set to full length of signal or fix common error of not having a list of lists
    ##

    if time_boundaries is None:
        time_start = np.nanmin(source_times)
        time_stop = np.nanmax(source_times)
        time_boundaries = [[time_start, time_stop]]
    elif isinstance(time_boundaries[0], numbers.Number):
        time_boundaries = [time_boundaries]

    ##
    ## Convert source indices to time boundaries, then convert time boundaries into target indices
    ##

    target_indices = []
    single_idx_count = 0

    ## Loop through start & end times and create list of indices corresponding to that block of time
    for [start, end] in time_boundaries:
        target_idx = np.where(
            np.logical_and(target_times >= start, target_times <= end)
        )[0]
        if len(target_idx) < 2:
            if drop_single_idx:
                single_idx_count += 1
            else:
                msg = (
                    f"Event of length {len(target_idx)} found. "
                    f"Set drop_single_idx to True to suppress these errors."
                )
                raise PipelineException(msg)
        else:
            target_indices.append(target_idx)

    if debug:
        print(f"Indices converted. {single_idx_count} events of length 0 or 1 dropped.")

    ##
    ## Interpolate related signal if requested, else just return the target_indices found.
    ##

    if return_interpolate:

        ## Create full interpolated signal
        interpolated_signal = interpolate_signal_data(
            scan_key, source_type, target_type, source_times, target_times, debug=debug
        )

        ## Split indices given into fragments based on which ones are continuous (incrementing by 1)
        source_signal_fragments = []
        for idx_fragment in target_indices:
            source_signal_fragments.append(interpolated_signal[idx_fragment])

        ## If full signal is converted, remove wrapping list
        if len(source_signal_fragments) == 1:
            source_signal_fragments = source_signal_fragments[0]

        return source_signal_fragments

    else:

        return target_indices


def convert_clocks_time_to_time(
    scan_key: dict,
    time_boundaries,
    source_type: str,
    target_type: str,
    return_interpolate: bool = False,
    drop_single_idx: bool = True,
    debug: bool = True,
):
    """
    Converts time boundaries of interest on one type of clock/recording to all times between those boundaries
    on the target signal/clock

        Parameters:

                scan_key: A dictionary specifying a single scan and/or field. A single field must be defined if requesting
                          a source or target from ScanImage. If key specifies a single unit, unit delay will be added to
                          all timepoints recorded. Single units can be specified via unique mask_id + field or via unit_id.
                          If only field is specified, average field delay will be added.

                time_boundaries: List of lists containing [start,stop] boundaries for times of note on source clock.
                                 Start/Stop times are included (>=,<=). These boundaries are converted to all indices
                                 equal to or between recording times on target recording.
                                 NOTE: Set to None to use the entire source signal length.
                                 ex. [[271, 314], [690.321, 800.1]] OR None

                source_type: A string specifying what times you want to convert from. Fluorescence and deconvolution
                             have a dash followed by "behavior" or "stimulus" to refer to which clock you are using.
                             Supported options:
                                 'fluorescence-stimulus', 'deconvolution-stimulus', ,'fluorescence-behavior',
                                 'deconvolution-behavior', 'pupil', 'treadmill', 'respiration'

                target_type: A string specifying what indices to convert into. Fluorescence and deconvolution have a
                             dash followed by "behavior" or "stimulus" to refer to which clock you are using.
                             Supported options:
                                 'fluorescence-stimulus', 'deconvolution-stimulus', ,'fluorescence-behavior',
                                 'deconvolution-behavior', 'pupil', 'treadmill', 'respiration'

                return_interpolate: Boolean with the following behavior
                                     - True: Interpolate source signal onto target clock, return interpolated signal
                                             between time boundaries instead of times
                                             NOTE: NaNs in source signal are linearly interpolated over and then
                                                   filled back in after interpolation onto target times.
                                     - False: Return corresponding times between boundaries for target

                drop_single_idx: Boolean with the following behavior
                                     - True: Drop any signal fragments which would create a signal of length 1.
                                     - False: Raise an error and stop if any list of indices leads to a signal
                                              of length 1.
                                    ex. Source IDX [1,2,...,300] on a 500HZ recording will only correspond to
                                        target IDX [1] if target is recorded at 1Hz.

                debug: Set function to print helpful debug text while running


        Returns:

                requested_array: Numpy array of corresponding times in target type recording or interpolated
                                 source signal during target times.

        Warnings:

                * NaN refilling for source signal will only refill values if NaNs stretch for multiple indices on
                  target clock

                * Recording points where the time value is NaN are dropped from analysis/processing
    """

    ##
    ## Fetch source and target times, along with converting between Stimulus or Behavior clock if needed
    ##

    source_times, target_times = fetch_timing_data(
        scan_key, source_type, target_type, debug
    )

    ##
    ## Check if None is used to set to full length of signal or fix common error of not having a list of lists
    ##

    if time_boundaries is None:
        time_start = np.nanmin(source_times)
        time_stop = np.nanmax(source_times)
        time_boundaries = [[time_start, time_stop]]
    elif isinstance(time_boundaries[0], numbers.Number):
        time_boundaries = [time_boundaries]

    ##
    ## Convert source indices to time boundaries, then convert time boundaries into target indices
    ##

    target_indices = []
    single_idx_count = 0

    ## Loop through start & end times and create list of indices corresponding to that block of time
    for [start, end] in time_boundaries:
        target_idx = np.where(
            np.logical_and(target_times >= start, target_times <= end)
        )[0]
        if len(target_idx) < 2:
            if drop_single_idx:
                single_idx_count += 1
            else:
                msg = (
                    f"Event of length {len(target_idx)} found. "
                    f"Set drop_single_idx to True to suppress these errors."
                )
                raise PipelineException(msg)
        else:
            target_indices.append(target_idx)

    if debug:
        print(f"Indices converted. {single_idx_count} events of length 0 or 1 dropped.")

    ##
    ## Interpolate related signal if requested, else return target times.
    ##

    if return_interpolate:

        ## Create full interpolated signal
        interpolated_signal = interpolate_signal_data(
            scan_key, source_type, target_type, source_times, target_times, debug=debug
        )

        ## Split indices given into fragments based on which ones are continuous (incrementing by 1)
        source_signal_fragments = []
        for idx_fragment in target_indices:
            source_signal_fragments.append(interpolated_signal[idx_fragment])

        ## If full signal is converted, remove wrapping list
        if len(source_signal_fragments) == 1:
            source_signal_fragments = source_signal_fragments[0]

        return source_signal_fragments

    else:

        ## Convert indices to times and return
        source_times_to_target_times = []

        for target_idx_list in target_indices:
            source_times_to_target_times.append(target_times[target_idx_list])

        return source_times_to_target_times
