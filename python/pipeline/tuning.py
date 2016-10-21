"""
Analysis of visual tuning: receptive fields, tuning curves, pixelwise maps
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve
import io
import imageio
import datajoint as dj
from . import preprocess, vis   # needed for foreign keys

from distutils.version import StrictVersion
assert StrictVersion(dj.__version__) >= StrictVersion('0.3.8')


schema = dj.schema('pipeline_tuning', locals())


def hamming(half, dim):
    """ normalized hamming kernel """
    k = np.hamming(np.floor(half) * 2 + 1)
    return k.reshape([1] * dim + [k.size]) / k.sum()


@schema
class CaKernel(dj.Lookup):
    definition = """  # options for calcium response kinetics.
    kernel  : tinyint    # calcium option number
    -----
    transient_shape  : enum('exp','onAlpha')  # calcium transient shape
    latency = 0      : float                  # (s) assumed neural response latency
    tau              : float                  # (s) time constant (used by some integration functions
    explanation      : varchar(255)           # explanation of calcium response kinents
    """

    contents = [
        [0, 'exp', 0.03,  0.5, 'instantaneous rise, exponential delay'],
        [1, 'onAlpha', 0.03, 0.5, 'alpha function response to on onset only'],
        [2, 'exp', 0.03,  1.0, 'instantaneous rise, exponential delay'],
        [3, 'onAlpha', 0.03, 1.0, 'alpha function response to on onset only']
    ]


@schema
class Directional(dj.Computed):
    definition = """  # all directional drift trials for the scan
    -> preprocess.Sync
    ---
    ndirections     : tinyint    # number of directions
    """

    class Trial(dj.Part):
        definition = """ #  directional drift trials
        -> Directional
        drift_trial     : smallint               # trial index
        ---
        -> vis.Trial
        direction                   : float                         # (degrees) direction of drift
        onset                       : double                        # (s) onset time in rf.Sync times
        offset                      : double                        # (s) offset time in rf.Sync times
        """


@schema
class OriDesignMatrix(dj.Computed):
    definition = """  # design matrix for directional response
    -> Directional
    -> CaKernel
    -----
    design_matrix   : longblob   # times x nConds
    regressor_cov   : longblob   # regressor covariance matrix,  nConds x nConds
    """


@schema
class OriMap(dj.Imported):
    definition = """ # pixelwise responses to full-field directional stimuli
    -> OriDesignMatrix
    -> preprocess.Prepare.GalvoMotion
    ---
    regr_coef_maps: longblob  # regression coefficients, widtlh x height x nConds
    r2_map: longblob  # pixelwise r-squared after gaussinization
    dof_map: longblob  # degrees of in original signal, width x height
    """


@schema
class Cos2Map(dj.Computed):
    definition = """  # pixelwise cosine fit to directional response
    -> OriMap
    -----
    cos2_amp   : longblob   # dF/F at preferred direction
    cos2_r2    : longblob   # fraction of variance explained (after gaussinization)
    cos2_fp    : longblob   # p-value of F-test (after gaussinization)
    pref_ori   : longblob   # (radians) preferred direction
    """


@schema
class DirectionalResponse(dj.Computed):
    definition = """  # response to directional stimulus
    -> Directional
    -> preprocess.Spikes
    ---
    latency : float   # latency used (ms)
    """

    class Trial(dj.Part):
        definition = """   # the response for each trial and each trace
        -> DirectionalResponse
        -> preprocess.Spikes.RateTrace
        -> Directional.Trial
        ---
        response : float   #  integrated response
        """

    def _make_tuples(self, key):
        print('Directional response for ', key)
        traces, slices, trace_keys = (
            preprocess.Spikes.RateTrace() * preprocess.Slice() & preprocess.ExtractRaw.GalvoROI() &
            key).fetch['rate_trace', 'slice', dj.key]
        traces = np.float64(np.stack(t.flatten() for t in traces))

        #  fetch and clean up the trace time
        trace_time = (preprocess.Sync() & key).fetch1['frame_times'].squeeze()  # calcium scan frame times
        n_slices = (preprocess.Prepare.Galvo() & key).fetch1['nslices']
        trace_time = trace_time[:n_slices*traces.shape[1]]  # truncate if interrupted scan
        assert n_slices*traces[0].size == trace_time.size, 'trace times must be a multiple of n_slices'
        slice_interval = (trace_time[1:]-trace_time[:-1]).mean()
        frame_interval = slice_interval * n_slices
        trace_time = trace_time[::n_slices]    # keep trace times for the first slice only

        # compute and interpolate cumulative traces on time of first slice
        assert traces.ndim == 2 and traces.shape[0] == len(trace_keys), 'incorrect trace dimensions'
        traces = interp1d(trace_time, np.cumsum(traces, axis=1)*frame_interval)

        # insert responses for each trace and trial with time adjustment for slices
        latency = 0.01  # s
        self.insert1(dict(key, latency=1000*latency))
        table = DirectionalResponse.Trial()
        for onset, offset, trial_key in zip(*(Directional.Trial() & key).fetch['onset', 'offset', dj.key]):
            for islice in set(slices):
                ix = np.where(slices == islice)[0]
                try:
                    responses = (traces(offset+latency-slice_interval*islice) -
                                 traces(onset+latency-slice_interval*islice))[ix]/(offset-onset)
                except ValueError:
                    pass
                else:
                    table.insert((dict(trial_key, response=response, **trace_keys[i])
                                  for i, response in zip(ix, responses)),
                                 ignore_extra_fields=True)
        print('Done')


schema.spawn_missing_classes()
