import datajoint as dj

from distutils.version import StrictVersion

import itertools
import os
from pprint import pformat, pprint
from functools import partial
from warnings import warn

import numpy as np
import scanreader
from datajoint.jobs import key_hash
from scipy.interpolate import interp1d
from tqdm import tqdm
from stimulus import stimulus
from scipy.ndimage import filters
from scipy import signal
from scipy.ndimage import convolve1d
from scipy.ndimage import median_filter

from pipeline import fuse, experiment, notify, shared, reso, meso, treadmill
from pipeline.exceptions import PipelineException
import pipeline.utils.clocktools as ct

from stimline import tune
from stimline._utils import corr, fill_nans, SplineCurve
from stimline._trippy import make_trippy

import os
from tifffile import imsave
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
from scipy.spatial.distance import cdist

dj.config["external-analysis"] = dict(
    protocol="file", location="/mnt/dj-stor01/datajoint-store/analysis"
)

dj.config["cache"] = "/tmp/dj-cache"

schema = dj.schema("pipeline_astrocytes", create_tables=True)


@schema
class QualityReference(dj.Lookup):
    definition = """ # Quality scores for 2p scans
    quality_score               : tinyint              # numeric quality score
    ---
    score_meaning               : varchar(256)         # explanation of score and examples
    """
    contents = [
        [5, "Perfect."],
        [
            4,
            "No unusual problems. Compromised in some small way. (ex. small bubble, very rare squinting, etc.)",
        ],
        [
            3,
            "Notable problems but possibly usable. Some parts are bad but less than 5-10min. (ex. eye closing, eye clouding, water running out, bubble formation, or other scan interruptions)",
        ],
        [2, "Terrible but some good parts. Less than 20min of quality data."],
        [1, "Profoundly bad. No usable data."],
    ]


@schema
class FieldPurpose(dj.Lookup):
    definition = """ # Purpose of the recording field in one channel
    field_purpose               : varchar(128)         # label for information stored in one field & channel combination
    ---
    purpose_notes               : varchar(256)         # explanation of listed field & channel purpose
    """
    contents = [
        ["Neurons", "Field contains neural data."],
        ["Astrocytes", "Field contains astrocyte calcium data."],
        ["NE Reporter", "Field contains measurements of Norepinephrine using GRAB_NE"],
        ["ACh Reporter", "Field contains measurements of Acetylcholine using GACh"],
    ]


@schema
class AstrocyteScans(dj.Manual):
    definition = """ # Scan which include astrocyte data
    -> experiment.Scan
    ---
    -> [nullable] QualityReference
    scan_purpose                : varchar(256)         # scan purpose/goal containing keywords used for processing
    scan_notes = ''             : varchar(256)         # relevant scan notes
    """

    class Field(dj.Part):
        definition = """ # Field specific information
        -> AstrocyteScans
        -> shared.Field
        -> shared.Channel
        -> shared.ExpressionConstruct
        ---
        -> [nullable] FieldPurpose
        field_notes = ''        : varchar(256)         # relevant field & channel notes
        """


@schema
class AstroCaMovie(dj.Computed):
    definition = """ # Corrected calcium movies for dual imaging or astrocytes
    -> AstrocyteScans.Field
    -> stimulus.Sync
    -> shared.Channel
    ---
    corrected_scan              : external-analysis   # motion corrected uint16
    scale_factor                : float               # scale back to float
    """

    def make(self, key):
        from pipeline.utils import performance

        # Read scan
        print("Reading scan...")
        scan_filename = (experiment.Scan & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)

        # Get some params
        pipe = (fuse.MotionCorrection() & key).module

        # Map: Correct scan in parallel
        f = performance.parallel_correct_scan  # function to map
        raster_phase = (pipe.RasterCorrection() & key).fetch1("raster_phase")
        fill_fraction = (pipe.ScanInfo() & key).fetch1("fill_fraction")
        y_shifts, x_shifts = (pipe.MotionCorrection() & key).fetch1(
            "y_shifts", "x_shifts"
        )
        kwargs = {
            "raster_phase": raster_phase,
            "fill_fraction": fill_fraction,
            "y_shifts": y_shifts,
            "x_shifts": x_shifts,
        }
        results = performance.map_frames(
            f,
            scan,
            field_id=key["field"] - 1,
            channel=key["channel"] - 1,
            kwargs=kwargs,
        )

        # Reduce: Rescale and save as int16
        height, width, _ = results[0][1].shape
        corrected_scan = np.zeros([height, width, scan.num_frames], dtype=np.int16)
        max_abs_intensity = max(np.abs(c).max() for f, c in results)
        scale = max_abs_intensity / (2 ** 15 - 1)
        for frames, chunk in results:
            corrected_scan[:, :, frames] = (chunk / scale).astype(np.int16)

        # Insert
        self.insert1({**key, "scale_factor": scale, "corrected_scan": corrected_scan})

    def save_tif(self, frame_limits=None, path=None):

        key = self.fetch1("KEY")
        frame_str = ""
        if frame_limits is not None:
            frame_str = f"_frames{frame_limits[0]}-{frame_limits[1]}"
        filename = f'id-{key["animal_id"]}_sess-{key["session"]}_scan-{key["scan_idx"]}_field-{key["field"]}_chan-{key["channel"]}{frame_str}.tif'
        if path is not None:
            filename = os.path.join(path, filename, ".tif")
        print("Loading scan...")
        scan = self.fetch1("corrected_scan")
        if frame_limits is not None:
            scan = scan[:, :, frame_limits[0] : frame_limits[1]]
        scan = np.moveaxis(scan, 2, 0)
        print(f"Saving scan {filename}")
        imsave(filename, scan)
        print("File saved!")
        scan = None


@schema
class AstroOriMap(dj.Computed):
    definition = """ # Pixelwise orientation response map
    -> AstroCaMovie
    -> tune.OriDesign
    ---
    response_map                : longblob            # pixelwise normalized response
    activity_map                : longblob            # root of sum of squares
    """

    def make(self, key):
        # get movie
        scan, scale = (AstroCaMovie & key).fetch1("corrected_scan", "scale_factor")
        scan = scan.astype(np.float32) * scale

        # get regressors
        X = (tune.OriDesign() & key).fetch1("regressors")
        pipe = (fuse.MotionCorrection() & key).module
        nfields_name = (
            "nfields/nrois" if "nrois" in pipe.ScanInfo.heading else "nfields"
        )
        nfields = int((pipe.ScanInfo & key).proj(n=nfields_name).fetch1("n"))
        X = X[key["field"] - 1 :: nfields, :]

        if abs(X.shape[0] - scan.shape[2]) > 1:
            raise PipelineException("The sync frames do not match scan frames.")
        else:
            # truncate scan if X is shorter
            if X.shape[0] < scan.shape[2]:
                warn("Scan is longer than design matrix")
                scan = scan[:, :, : X.shape[0]]
            # truncate design matrix if scan is shorter
            if scan.shape[2] < X.shape[0]:
                warn("Scan is shorter than design matrix")
                X = X[: scan.shape[2], :]

        # limit the analysis to times when X is non-zero
        ix = (X ** 2).sum(axis=1) > 1e-4 * (X ** 2).sum(axis=1).max()
        X = X[ix, :]
        scan = scan[:, :, ix]

        # normalize regressors
        X = X - X.mean(axis=0)
        X /= np.sqrt((X ** 2).sum(axis=0, keepdims=True))

        # normalize movie
        scan -= scan.mean(axis=2, keepdims=True)
        key["activity_map"] = np.sqrt((scan ** 2).sum(axis=2))
        scan /= key["activity_map"][:, :, None] + 1e-6

        # compute response
        key["response_map"] = np.tensordot(scan, np.linalg.pinv(X), axes=(2, 1))
        self.insert1(key)


@schema
class Cos2Map(dj.Computed):
    definition = """
    # Pixelwise orientation tuning map
    -> AstroOriMap
    ---
    direction_map  : longblob
    amplitude_map  : longblob
    """

    def make(self, key):
        u, r = (AstroOriMap() * tune.Drift() & key).fetch1(
            "unique_directions", "response_map"
        )
        assert (
            all(u[1:] - u[:-1] == u[1] - u[0]) and (u[1] - u[0]) * u.size == 360
        ), "nonuniform directions"
        cos2_vec = np.exp(2j * u * np.pi / 180) / np.sqrt(u.size / 2)
        a = np.tensordot(r, cos2_vec, (2, 0))
        self.insert1(dict(key, direction_map=np.angle(a / 2), amplitude_map=np.abs(a)))

    def map(self, key):
        import matplotlib.colors as mcolors

        a, m = (Cos2Map & key).fetch1("direction_map", "amplitude_map")
        h = (a / np.pi / 2) % 1
        astro_boost = 100
        neuron_boost = 10
        print(
            f"RUNNING HACKED AMPS - Boosting astrocyte signal by {astro_boost}. Boosting neuron signal by {neuron_boost}."
        )

        #         v = np.minimum(m ** 2 / 0.05, 1)
        v = np.minimum(m ** 2 / 0.025 * neuron_boost, 1)
        s = np.minimum(m / 0.1 * neuron_boost, 1)
        if key["channel"] == 1:
            v = np.minimum(m ** 2 / 0.025 * astro_boost, 1)
            s = np.minimum(m / 0.1 * astro_boost, 1)
        return mcolors.hsv_to_rgb(np.stack((h, s, v), axis=2))

    def make_figure(self):
        import seaborn as sns
        import matplotlib.pyplot as plt

        key = self.fetch1("KEY")
        im = self.map(key)
        amap = (self & key).fetch1("amplitude_map") ** 2
        pmap = np.interp(
            amap, amap.ravel()[np.argsort(amap.ravel())], np.linspace(0, 100, amap.size)
        )

        with sns.axes_style("white"):
            title = "pixelwise orientation for {animal_id}-{session}-{scan_idx} field {field}".format(
                **key
            )
            fig = plt.figure(figsize=(15, 15))
            if im.shape[0] > im.shape[1]:
                orientation = "horizontal"
                gs = plt.GridSpec(21, 2)
                ax_ori = fig.add_subplot(gs[:-1, 0])
                cax_ori = fig.add_subplot(gs[-1, 0])
                ax_r2 = fig.add_subplot(gs[:-1, 1])
                cax_r2 = fig.add_subplot(gs[-1, 1])
            else:
                orientation = "vertical"
                gs = plt.GridSpec(2, 21)
                ax_ori = fig.add_subplot(gs[0, :-1])
                cax_ori = fig.add_subplot(gs[0, -1])
                ax_r2 = fig.add_subplot(gs[1, :-1])
                cax_r2 = fig.add_subplot(gs[1, -1])

            h = ax_ori.imshow(im, interpolation="nearest")
            fig.colorbar(h, cax=cax_ori, orientation=orientation)
            h = ax_r2.matshow(pmap, cmap="coolwarm")
            fig.colorbar(h, cax=cax_r2, orientation=orientation)
            [a.axis("off") for a in [ax_ori, ax_r2]]
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            ax_ori.set_title("pixelwise orientation map")
            ax_r2.set_title("percentile power (amplitude_map^2)")
            fig.suptitle("{animal_id}-{session}-{scan_idx} field {field}".format(**key))
            plt.show()
            plt.close(fig)


@schema
class OriMapQuality(dj.Computed):
    definition = """
    # Summary quality of pixelwise tuning
    -> Cos2Map
    ---
    r2_9  : float  # 90th percentile of variance explained
    r2_99 : float  # 99th percentile of variance explained
    r2_999: float  # 99.9th percentile of variance explained
    """

    def make(self, key):
        r2 = (Cos2Map & key).fetch1("amplitude_map") ** 2
        self.insert1(
            dict(
                key,
                r2_9=np.nanpercentile(r2, 90),
                r2_99=np.nanpercentile(r2, 99),
                r2_999=np.nanpercentile(r2, 99.9),
            )
        )


@schema
class OracleMap(dj.Computed):
    definition = """
    # Pixelwise oracle map
    -> AstrocyteScans.Field
    ---
    oracle      :  float      # average oracle value
    oracle_map  :  longblob   # pixelwise correlation values
    p_map       :  longblob   # pixel p-value
    """

    def load(self, key):
        from pipeline.utils import galvo_corrections

        # load
        print("Loading scan", flush=True)
        reader = scanreader.read_scan(
            (experiment.Scan() & key).local_filenames_as_wildcard
        )
        scan = reader[key["field"] - 1, :, :, key["channel"] - 1].astype(np.float32)

        # raster correction
        print("Raster correction", flush=True)
        pipe = (fuse.MotionCorrection() & key).module
        raster_phase = (pipe.RasterCorrection() & key).fetch1("raster_phase")
        fill_fraction = (pipe.ScanInfo() & key).fetch1("fill_fraction")
        scan = galvo_corrections.correct_raster(scan, raster_phase, fill_fraction)

        # motion correction
        print("Motion correction", flush=True)
        x_shifts, y_shifts = (pipe.MotionCorrection() & key).fetch1(
            "x_shifts", "y_shifts"
        )
        scan = galvo_corrections.correct_motion(scan, x_shifts, y_shifts)

        return scan, reader.num_scanning_depths

    @property
    def key_source(self):
        rel2 = (
            stimulus.Clip() * fuse.MotionCorrection() * stimulus.Movie()
            & 'movie_class in ("cinema", "youtube", "unreal")'
        ).aggr(stimulus.Trial(), repeats="count(movie_name)")
        return AstrocyteScans.Field & stimulus.Sync() & (rel2 & "repeats > 2").proj()

    def make(self, key):
        print("Populating\n", pformat(key, indent=10))
        repeats = (
            stimulus.Clip().aggr(stimulus.Trial() & key, repeats="count(movie_name)")
            & "repeats > 2"
        )
        scan, ndepths = self.load(key)

        frame_times = (stimulus.Sync() & key).fetch1("frame_times").squeeze()
        frame_times = frame_times[key["field"] - 1 :: ndepths]

        ft_min = frame_times.min()
        frame_times = frame_times - ft_min

        if 0 <= np.abs(frame_times.size - scan.shape[-1]) < 20:
            print("Shortening length of frametimes and scan to the same size")
            ml = min(frame_times.size, scan.shape[-1])
            scan = scan[..., :ml]
            frame_times = frame_times[:ml]
        else:
            raise ValueError(
                "Difference in frametimes and scan length greater 20 frames"
            )

        downsample_to = 0.250  # 4 Hz
        h = np.hamming(
            2 * int(downsample_to // np.median(np.diff(frame_times))) + 1
        ).astype(np.float32)
        h /= h.sum()

        oracles, data, data_shuffle = [], [], []
        *spatial_dim, T = scan.shape
        scan = scan.reshape((-1, T))
        # permute = lambda x: x[np.random.permutation(len(x))]
        for condition in (dj.U("condition_hash") & repeats).fetch(dj.key):
            # --- load fliptimes
            trial_keys, flip_times = (stimulus.Trial() & key & condition).fetch(
                dj.key, "flip_times"
            )
            l = np.min([ft.size for ft in flip_times])
            flip_times = [ft.squeeze()[:l] - ft_min for ft in flip_times]
            flip_times = [
                np.arange(ft.min(), ft.max(), downsample_to) for ft in flip_times
            ]  # downsample to 4 Hz

            # --- smooth trial, subsample, compute mean
            movs = []
            for ft in tqdm(flip_times, desc="Trial: "):
                movs.append(
                    np.vstack(
                        [
                            np.interp(ft, frame_times, np.convolve(px, h, mode="same"))
                            for px in scan
                        ]
                    ).reshape(tuple(spatial_dim) + (len(ft),))
                )
            mov = np.stack(movs, axis=0)
            mu = mov.mean(axis=0, keepdims=True)

            r, *_, t = mov.shape
            oracle = (mu - mov / r) * r / (r - 1)
            spatial_dim = tuple(spatial_dim)
            oracles.append(oracle.transpose([0, 3, 1, 2]).astype(np.float32))
            data.append(mov.transpose([0, 3, 1, 2]).astype(np.float32))
        key["oracle_map"], key["p_map"] = corr(
            np.concatenate(data, axis=0),
            np.concatenate(oracles, axis=0),
            axis=(0, 1),
            return_p=True,
        )
        key["oracle"] = key["oracle_map"].mean()
        self.insert1(key)

    def make_figure(self):
        import seaborn as sns
        import matplotlib.pyplot as plt

        key = self.fetch1()
        m, p = key["oracle_map"], key["p_map"]
        cmap = sns.blend_palette(
            ["dodgerblue", "steelblue", "k", "lime", "yellow"], as_cmap=True
        )
        with sns.axes_style("white"):
            title = "oracle image for {animal_id}-{session}-{scan_idx} field {field}".format(
                **key
            )
            fig = plt.figure(figsize=(15, 15))
            if m.shape[0] > m.shape[1]:
                orientation = "horizontal"
                gs = plt.GridSpec(21, 2)
                ax_corr = fig.add_subplot(gs[:-1, 0])
                cax_corr = fig.add_subplot(gs[-1, 0])
                ax_p = fig.add_subplot(gs[:-1, 1])
                cax_p = fig.add_subplot(gs[-1, 1])
            else:
                orientation = "vertical"
                gs = plt.GridSpec(2, 21)
                ax_corr = fig.add_subplot(gs[0, :-1])
                cax_corr = fig.add_subplot(gs[0, -1])
                ax_p = fig.add_subplot(gs[1, :-1])
                cax_p = fig.add_subplot(gs[1, -1])

            # v = np.abs(m).max()
            h = ax_corr.imshow(m, vmin=-1, vmax=1, cmap=cmap)
            fig.colorbar(h, cax=cax_corr, orientation=orientation)

            h = ax_p.matshow(np.log(p / p.size), cmap="coolwarm_r")
            # fig.colorbar(h, cax=cax_p, orientation=orientation)
            [a.axis("off") for a in [ax_corr, ax_p]]
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            ax_corr.set_title("oracle correlation map")
            ax_p.set_title("log p-value (incorrect DF)")
            fig.suptitle("{animal_id}-{session}-{scan_idx} field {field}".format(**key))
            plt.show()
            plt.close(fig)

            
@schema
class SignalDelayMethod(dj.Lookup):
    definition = """ # Method used to calculate the delay between two signals.
    delay_method_id             : tinyint unsigned     # numeric id for delay method
    ---
    -> shared.FilterMethod
    delay_method_notes          : varchar(256)         # explanation of delay detection method
    """
    contents = [
        [1, '5Hz Hamming Lowpass', 'Compares astrocytes and neurons. Filter and run cross-correlation on 20sec window (in 500ms steps) after trimming 10sec from onset/offset of full trace. Running periods excluded using running_method_id 3.'],
        [2, '5Hz Hamming Lowpass', 'Compares astrocytes and neurons. Filter and run cross-correlation on 6sec window (in 1 frame steps) after trimming 10sec from onset/offset of full trace. Running periods excluded using running_method_id 3.'],
        [3, '5Hz Hamming Lowpass', 'Compares astrocytes and neurons. Filter and run cross-correlation on 6sec window (in 1 frame steps) after trimming 10sec from onset/offset of full trace.'],
    ]


@schema
class KernelEstimationMethod(dj.Lookup):
    definition = """ # Method used to estimate the kernel of a spiking signal.
    kernel_method_id            : tinyint unsigned     # numeric id for kernel estimation method
    ---
    -> shared.FilterMethod
    kernel_method_notes         : varchar(256)         # explanation of delay detection method
    """
    contents = [
        [1, '2Hz Hamming Lowpass', 'Filter and run scipy find_peaks detection. Peak is defined in 20sec window.'],
    ]


@schema
class DelayDetection(dj.Computed):
    definition = """ # Computed delay between two cells 
    -> AstrocyteScans
    -> SignalDelayMethod
    ---
    signal_idx_mask             : mediumblob           # boolean mask used on signal traces
    """
    
    class UnitComparison(dj.Part):
        definition = """ # individual comparison between two units
        -> DelayDetection
        -> fuse.ScanSet.Unit.proj(source_unit_id='unit_id')
        -> fuse.ScanSet.Unit.proj(target_unit_id='unit_id')
        ---
        similarity_metric       : float                # similarity score given metric such as pearsonr correlation, DTW, etc.
        estimated_delay         : float                # (sec) guessed delay between the two signals. Negative means source_unit leads.
        unit_dist = NULL        : float                # (um) distance between units given motor coordinates
        """


        def plot_comparison(self, plot_window=None):
            
            def normalize_signal(signal, offset=0):
                signal = signal - np.nanmean(signal)
                signal = signal / (np.nanmax(signal) - np.nanmin(signal))
                signal = signal - np.nanmin(signal) + offset
                return signal
            
            key = self.fetch1('KEY')
            signal_idx_mask = (DelayDetection & key).fetch1('signal_idx_mask')
            pipe = (fuse.ScanSet & key).module
            fps = (pipe.ScanInfo & key).fetch1('fps')
            
            # We default to field 1 here since we just need general timing data to detrend and figure out if there is an extra frame_time without a corresponding measurement
            plot_frame_times = ct.convert_clocks({**key, 'field': 1}, None, 'times', 'fluorescence-behavior', 'times-source', 'fluorescence-behavior', debug=False)
            plot_frame_times = plot_frame_times[:len(signal_idx_mask)]
            frame_times = plot_frame_times[signal_idx_mask]
            
            source_key = (pipe.ScanSet.Unit & {**key, 'unit_id': key['source_unit_id']}).fetch(as_dict=True)
            target_key = (pipe.ScanSet.Unit & {**key, 'unit_id': key['target_unit_id']}).fetch(as_dict=True)
            
            fig,axes = plt.subplots(2,1,figsize=(14,8))
            
            if key['delay_method_id'] == 1:
                source_table = pipe.Fluorescence.Trace
                target_table = pipe.Fluorescence.Trace
                detrending_func = partial(DelayDetection._RANSAC_detrending, self=None, frame_times=frame_times)
                delay_function = partial(DelayDetection._pearsonr_crosscorr, self=None, fps=fps, window_size=20, step_size=0.25, max_or_min="max")
                title_line_two = '\nAstro leads <> Neuro leads'
                xlabel, ylabel = ('Delay (sec)', 'PearsonR')
            elif key['delay_method_id'] == 2:
                source_table = pipe.Fluorescence.Trace
                target_table = pipe.Fluorescence.Trace
                detrending_func = partial(DelayDetection._rolling_median_detrending, self=None, frame_times=frame_times, cropping_offset=10)
                delay_function = partial(DelayDetection._pearsonr_crosscorr, self=None, fps=fps, window_size=6, step_size=None, max_or_min="max")
                title_line_two = '\nAstro leads <> Neuro leads'
                xlabel, ylabel = ('Delay (sec)', 'PearsonR')
            elif key['delay_method_id'] == 3:
                source_table = pipe.Fluorescence.Trace
                target_table = pipe.Fluorescence.Trace
                detrending_func = partial(DelayDetection._rolling_median_detrending, self=None, frame_times=frame_times, cropping_offset=10)
                delay_function = partial(DelayDetection._pearsonr_crosscorr, self=None, fps=fps, window_size=6, step_size=None, max_or_min="max")
                title_line_two = '\nAstro leads <> Neuro leads'
                xlabel, ylabel = ('Delay (sec)', 'PearsonR')
            else:
                raise Exception(f'Delay method id {key["delay_method_id"]} not supported.')
                
            fps = (pipe.ScanInfo & key).fetch1('fps')
            delay_method_key = (SignalDelayMethod & key).fetch1()
            my_filter = (shared.FilterMethod & delay_method_key).run_filter
            source_plot_signal = (source_table & source_key).fetch1('trace')
            source_signal = source_plot_signal[signal_idx_mask]
            source_signal = detrending_func(trace=source_signal)
            source_signal = my_filter(source_signal, fps)
            target_plot_signal = (target_table & target_key).fetch1('trace')
            target_signal = target_plot_signal[signal_idx_mask]
            target_signal = detrending_func(trace=target_signal)
            target_signal = my_filter(target_signal, fps)
            similarity_vals, lags = delay_function(trace1=source_signal, trace2=target_signal, return_all_values=True)
            _, calculated_delay = delay_function(trace1=source_signal, trace2=target_signal, return_all_values=False)
            
            if plot_window is None:
                mask_fragments = ct.find_idx_boundaries(np.where(signal_idx_mask)[0], drop_single_idx=True)
                source_plot_signal = normalize_signal(source_plot_signal, 1)
                target_plot_signal = normalize_signal(target_plot_signal, 0)
                for mask_fragment in mask_fragments:
                    axes[0].plot(plot_frame_times[mask_fragment], source_plot_signal[mask_fragment], color='green')
                    axes[0].plot(plot_frame_times[mask_fragment], target_plot_signal[mask_fragment], color='red')
            else:
                plot_window_idx = int(plot_window * fps)
                start = np.random.choice(len(source_signal) - plot_window_idx)
                end = start + plot_window_idx
                axes[0].plot(frame_times[start:end], normalize_signal(source_signal[start:end], 1), color='green')
                axes[0].plot(frame_times[start:end], normalize_signal(target_signal[start:end], 0), color='red')
                
            axes[0].get_yaxis().set_visible(False)
            axes[0].spines['left'].set_visible(False)
            axes[0].spines['right'].set_visible(False)
            axes[0].spines['top'].set_visible(False)
            axes[0].set_xlabel('Seconds')
            
            title = f'Unit {key["source_unit_id"]} vs Unit {key["target_unit_id"]}. Offset = {round(calculated_delay,3)} seconds' + title_line_two
            axes[1].plot(lags, similarity_vals)
            axes[1].set(title=title, xlabel=xlabel, ylabel=ylabel)
            plt.subplots_adjust(hspace = 0.6)
            plt.show()
            plt.close()
            
            return
    
    
    def _no_detrending(self, trace, frame_times):
        return trace
    
    
    def _RANSAC_detrending(self, trace, frame_times):
        model = RANSACRegressor(random_state=0, max_trials=1000) # robust regression
        model.fit(frame_times.reshape(-1,1), trace)
        trace_treadline = model.predict(frame_times.reshape(-1,1))
        return trace - trace_treadline
    
    
    def _rolling_mean_detrending(self, trace, frame_times, cropping_offset=10, window_size=300):
        cropped_start_mask = frame_times > (cropping_offset + np.nanmin(frame_times))
        cropped_end_mask = frame_times < (np.nanmax(frame_times) - cropping_offset)
        cropping_mask = np.all((cropped_start_mask, cropped_end_mask), axis=0)
        trace_fps = 1/np.median(np.diff(frame_times))
        rolling_mean_size = int(trace_fps * window_size)
        rolling_mean = convolve1d(trace[cropping_mask], np.ones(rolling_mean_size) / rolling_mean_size, mode="reflect")
        return_trace = trace.copy()
        return_trace[cropping_mask] -= rolling_mean
        return return_trace
    
    
    def _rolling_median_detrending(self, trace, frame_times, cropping_offset=10, window_size=300):
        cropped_start_mask = frame_times > (cropping_offset + np.nanmin(frame_times))
        cropped_end_mask = frame_times < (np.nanmax(frame_times) - cropping_offset)
        cropping_mask = np.all((cropped_start_mask, cropped_end_mask), axis=0)
        trace_fps = 1/np.median(np.diff(frame_times))
        rolling_median_size = int(trace_fps * window_size)
        rolling_median = median_filter(trace[cropping_mask], size=rolling_median_size, mode="reflect")
        return_trace = trace.copy()
        return_trace[cropping_mask] -= rolling_median
        return return_trace
    
    
    def _pearsonr_crosscorr(self, trace1, trace2, fps, window_size, step_size, max_or_min, return_all_values=False):
        
        def crosscorr(datax, datay, lag=0, wrap=False):
            if wrap:
                shiftedy = datay.shift(lag)
                shiftedy.iloc[:lag] = datay.iloc[-lag:].values
                return datax.corr(shiftedy)
            else: 
                return datax.corr(datay.shift(lag))
        
        t1 = pd.Series(trace1)
        t2 = pd.Series(trace2)
        if step_size is None:
            lags = np.arange(0,round(window_size/2*fps),step=1)
        else:
            lags = np.arange(0,round(window_size/2*fps),step=round(step_size*fps))
        lags = np.hstack((-lags[::-1], lags[1:])).astype(int)  # This ensures 0 is at the center of given lags. Mirror and change sign before appending to beginning of lags[1:]
        rs = [crosscorr(t1, t2, lag) for lag in lags]
        if max_or_min.lower() == 'max':
            best_corr = np.max(rs)
            best_corr_idx = np.argmax(rs)
        elif max_or_min.lower() == 'min':
            best_corr = np.min(rs)
            best_corr_idx = np.argmin(rs)
        else:
            raise Exception(f'Argument {max_or_min} not supported. Use "max" or "min".')
        calculated_delay = lags[best_corr_idx]/fps
        if return_all_values:
            return rs, lags/fps
        return best_corr, calculated_delay
        
    def _get_locations(self, pipe, key, unit_id1, unit_id2):
        source_unit_key = {**key, 'unit_id': unit_id1}
        target_unit_key = {**key, 'unit_id': unit_id2}
        source_xyz_um = list((pipe.ScanSet.UnitInfo & source_unit_key).fetch1('um_x', 'um_y', 'um_z'))
        target_xyz_um = list((pipe.ScanSet.UnitInfo & target_unit_key).fetch1('um_x', 'um_y', 'um_z'))
        return source_xyz_um, target_xyz_um
            
    def _point_distance(self, source_loc, target_loc):
        distance = cdist([source_loc], [target_loc])[0][0]
        return distance


    def make(self, key):
        
        print(f'Populating DelayDetection for {key}...')
        
        ## Fetch scan information
        pipe = (fuse.ScanSet & key).module
        fps = (pipe.ScanInfo & key).fetch1('fps')
        # We default to field 1 here since we just need general timing data to detrend and figure out if there is an extra frame_time without a corresponding measurement
        frame_times = ct.convert_clocks({**key, 'field': 1}, None, 'times', 'fluorescence-behavior', 'times-source', 'fluorescence-behavior', debug=False)
        
        ## Define delay method settingsa
        if key['delay_method_id'] == 1:
            source_table = pipe.Fluorescence.Trace
            target_table = pipe.Fluorescence.Trace
            
            source_keys = (AstrocyteScans.Field & key & 'field_purpose="Astrocytes"').fetch('KEY')
            for source_key in source_keys:
                source_key['segmentation_method'] = 6
                source_key['pipe_version'] = 1
            target_keys = (AstrocyteScans.Field & key & 'field_purpose="Neurons"').fetch('KEY')
            for target_key in target_keys:
                target_key['segmentation_method'] = 6
                target_key['pipe_version'] = 1
            
            cropping_offset = 10  # sec
            trace_mask = (treadmill.Running & key & {'running_method_id': 3 }).get_nonrunning_idx({**key, 'field': 1})
            detrending_func = self._RANSAC_detrending
            delay_function = partial(self._pearsonr_crosscorr, fps=fps, window_size=20, step_size=0.25, max_or_min="max")
            distance_function = self._point_distance
            
        elif key['delay_method_id'] == 2:
            source_table = pipe.Fluorescence.Trace
            target_table = pipe.Fluorescence.Trace
            
            source_keys = (AstrocyteScans.Field & key & 'field_purpose="Astrocytes"').fetch('KEY')
            for source_key in source_keys:
                source_key['segmentation_method'] = 6
                source_key['pipe_version'] = 1
            target_keys = (AstrocyteScans.Field & key & 'field_purpose="Neurons"').fetch('KEY')
            for target_key in target_keys:
                target_key['segmentation_method'] = 6
                target_key['pipe_version'] = 1
            
            cropping_offset = 10  # sec
            trace_mask = (treadmill.Running & key & {'running_method_id': 3 }).get_nonrunning_idx({**key, 'field': 1})
            detrending_func = partial(self._rolling_median_detrending, cropping_offset=cropping_offset)
            delay_function = partial(self._pearsonr_crosscorr, fps=fps, window_size=6, step_size=None, max_or_min="max")
            distance_function = self._point_distance
            
        elif key['delay_method_id'] == 3:
            source_table = pipe.Fluorescence.Trace
            target_table = pipe.Fluorescence.Trace
            
            source_keys = (AstrocyteScans.Field & key & 'field_purpose="Astrocytes"').fetch('KEY')
            for source_key in source_keys:
                source_key['segmentation_method'] = 6
                source_key['pipe_version'] = 1
            target_keys = (AstrocyteScans.Field & key & 'field_purpose="Neurons"').fetch('KEY')
            for target_key in target_keys:
                target_key['segmentation_method'] = 6
                target_key['pipe_version'] = 1
            
            cropping_offset = 10  # sec
            trace_mask = np.ones_like(frame_times).astype(bool)  # no trials removed
            detrending_func = partial(self._rolling_median_detrending, cropping_offset=cropping_offset)
            delay_function = partial(self._pearsonr_crosscorr, fps=fps, window_size=6, step_size=None, max_or_min="max")
            distance_function = self._point_distance
            
        else:
            raise Exception(f'Delay method {key["delay_method_id"]} not supported.')
            

        ## Fetch trace data
        all_source_traces, source_unit_ids = ((source_table & source_keys) * pipe.ScanSet.Unit).fetch('trace', 'unit_id')
        all_target_traces, target_unit_ids = ((target_table & target_keys) * pipe.ScanSet.Unit).fetch('trace', 'unit_id')
        
        ## Fix extra timing indices (if an issue)
        shorter_length = np.min((len(frame_times), len(all_source_traces[0])))
        frame_times = frame_times[:shorter_length]
        trace_mask = trace_mask[:shorter_length] 
        
        ## Modifying trace mask based on cropping to exclude times during first/last cropping_offset seconds
        cropped_start_mask = frame_times > (cropping_offset + np.nanmin(frame_times))
        cropped_end_mask = frame_times < (np.nanmax(frame_times) - cropping_offset)
        cropping_mask = np.all((cropped_start_mask, cropped_end_mask), axis=0)
        trace_mask = np.all((trace_mask, cropping_mask), axis=0)
        
        ## Filter, detrend, crop, and mask traces
        print('Filtering, detrending, cropping, and masking source & target traces...')
        delay_method_key = (SignalDelayMethod & key).fetch1()
        my_filter = (shared.FilterMethod & delay_method_key).run_filter
        for trace_collection in (all_source_traces, all_target_traces):
            for i in tqdm(range(len(trace_collection))):
                # Filter
                temp_trace = my_filter(trace_collection[i], fps)
                # Detrend
                temp_trace = detrending_func(trace=temp_trace, frame_times=frame_times)
                # Crop and Mask
                temp_trace = temp_trace[trace_mask]
                trace_collection[i] = temp_trace
        
        ## Define basis for all the pair keys we will be populating
        important_keys = ('pipe_version', 'segmentation_method')
        pair_key_basis = {k: v for k, v in source_keys[0].items() if k in important_keys}
        pair_key_basis = {**key, **pair_key_basis}
        pair_comparison_keys = []
        
        ## Run comparisons for all source-target trace pairs
        print(f'Running all pairwise comparisons for delay_method_id {key["delay_method_id"]}.')
        source_indices = np.arange(len(all_source_traces))
        target_indices = np.arange(len(all_target_traces))
        loop_total = len(all_source_traces) * len(all_target_traces)
        for source_idx,target_idx in tqdm(product(source_indices,target_indices), total=loop_total):
            
            source_id, target_id = (source_unit_ids[source_idx], target_unit_ids[target_idx])
            similarity, delay = delay_function(trace1=all_source_traces[source_idx], trace2=all_target_traces[target_idx])
            source_location, target_location = self._get_locations(pipe, {**key, 'segmentation_method': source_keys[0]['segmentation_method']}, source_id, target_id)
            distance = distance_function(source_location, target_location)
            pair_key = {**pair_key_basis, 'source_unit_id': source_id, 'target_unit_id': target_id, 'similarity_metric': similarity, 'estimated_delay': delay, 'unit_dist': distance}
            pair_comparison_keys.append(pair_key)
            
        self.insert1({**key, 'signal_idx_mask': trace_mask})
        DelayDetection.UnitComparison.insert(pair_comparison_keys)
        print(f'DelayDetection finished populating for {key}!\n')