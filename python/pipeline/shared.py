from warnings import warn
import datajoint as dj

schema = dj.schema('pipeline_shared', locals())

@schema
class Slice(dj.Lookup):
    definition = """  # slices in resonant scans
    slice       : tinyint
    """
    contents = [[i] for i in range(1, 13)]


@schema
class Channel(dj.Lookup):
    definition = """  # recording channel, directly related to experiment.PMTFilterSet.Channel
    channel     : tinyint
    """
    contents = [[i] for i in range(1, 5)]

@schema
class SegmentationMethod(dj.Lookup):
    definition = """ # methods for mask extraction for multi-photon scans
    segmentation_method         : tinyint
    ---
    name                        : varchar(16)
    details                     : varchar(255)
    language                    : enum('matlab', 'python')  # implementation language
    """

    contents = [
    [1, 'manual', '', 'matlab'],
    [2, 'nmf', 'constrained non-negative matrix factorization from Pnevmatikakis et al. (2016)', 'python']
    ]

@schema
class ClassificationMethod(dj.Lookup):
    definition = """ # methods to classify extracted masks
    classification_method         : tinyint
    ---
    name                        : varchar(16)
    details                     : varchar(255)
    language                    : enum('matlab', 'python')  # implementation language
    """

    contents = [
    [1, 'manual', 'masks classified by visual inspection', 'python'],
    [2, 'cnn', 'classification made by a trained convolutional network', 'python']
    ]

@schema
class MaskType(dj.Lookup):
    definition = """ # possible classifications for a segmented mask
    type        : varchar(16)
    """
    contents = [
        ['soma'],
        ['axon'],
        ['dendrite'],
        ['neuropil'],
        ['artifact'],
        ['unknown']
    ]

@schema
class SpikeMethod(dj.Lookup):
    definition = """
    spike_method        : tinyint                   # spike inference method
    ---
    name                : varchar(16)               # short name to identify the spike inference method
    details             : varchar(255)              # more details
    language            : enum('matlab', 'python')  # implementation language
    """

    contents = [
        [2, 'oopsi', 'nonnegative sparse deconvolution from Vogelstein (2010)', 'python'],
        [3, 'stm', 'spike triggered mixture model from Theis et al. (2016)', 'python'],
        [5, 'nmf', 'noise constrained deconvolution from Pnevmatikakis et al. (2016)', 'python']
    ]