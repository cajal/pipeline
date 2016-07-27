"""
Analysis of visual responses: receptive fields, tuning curves, pixelwise maps
"""

import datajoint as dj
from . import preprocess, vis   # needed for foreign keys

from distutils.version import StrictVersion
assert StrictVersion(dj.__version__) >= StrictVersion('0.2.8')

schema = dj.schema('pipeline_tuning', locals())


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
    regr_coef_maps: longblob  # regression coefficients, width x height x nConds
    r2_map: longblob  # pixelwise r-squared after gaussinization
    dof_map: longblob  # degrees of in original signal, width x height
    """


@schema
class Cos2Map(dj.Computed):
    definition = """  # pixelwise cosine fit to directional responses
    -> OriMap
    -----
    cos2_amp   : longblob   # dF/F at preferred direction
    cos2_r2    : longblob   # fraction of variance explained (after gaussinization)
    cos2_fp    : longblob   # p-value of F-test (after gaussinization)
    pref_ori   : longblob   # (radians) preferred direction
    """

schema.spawn_missing_classes()
