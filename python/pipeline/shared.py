""" Lookup tables shared among multi-photon pipelines. """

import datajoint as dj

schema = dj.schema('pipeline_shared', locals(), create_tables=False)

@schema
class Field(dj.Lookup):
    definition = """ # fields in mesoscope scans
    field       : tinyint
    """
    contents = [[i] for i in range(1, 25)]

@schema
class Channel(dj.Lookup):
    definition = """  # recording channel, directly related to experiment.PMTFilterSet.Channel
    channel     : tinyint
    """
    contents = [[i] for i in range(1, 5)]

@schema
class PipelineVersion(dj.Lookup):
    definition = """ # versions for the reso pipeline

    pipe_version                    : smallint
    """
    contents = [[i] for i in range(3)]

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
    [2, 'nmf', 'constrained non-negative matrix factorization from Pnevmatikakis et al. (2016)', 'python'],
    [3, 'nmf-patches', 'same as nmf but initialized in small image patches', 'python'],
    [4, 'nmf-boutons', 'nmf for axonal terminals', 'python']
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
    [2, 'cnn-caiman', 'classification made by a trained convolutional network', 'python']
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
