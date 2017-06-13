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
    definition = """
    #  methods for trace extraction from raw data for either AOD or Galvo data

    extract_method      : tinyint
    ---
    segmentation        : varchar(16)
    """

    contents = zip([1, 2], ['manual', 'nmf'])

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
    spike_method            :  smallint         # spike inference method
    ---
    spike_method_name       : varchar(16)       #  short name to identify the spike inference method
    spike_method_details    : varchar(255)      #  more details
    language                : enum('matlab', 'python')   #  implementation language
    """

    contents = [
        [2, "oopsi", "nonnegative sparse deconvolution from Vogelstein (2010)", "python"],
        [3, "stm", "spike triggered mixture model from Theis et al. (2016)", "python"],
        [5, "nmf", "noise constrained deconvolution from Pnevmatikakis et al., 2016", "python"]
    ]

    def spike_traces(self, X, fps):
        try:
            import c2s
        except ImportError:
            warn("c2s was not found. You won't be able to populate ExtracSpikes")
        assert self.fetch1['language'] == 'python', "This tuple cannot be computed in python."
        if self.fetch1['spike_method'] == 3:
            N = len(X)
            for i, trace in enumerate(X):
                print('Predicting trace %i/%i' % (i + 1, N))
                tr0 = np.array(trace.pop('trace').squeeze())
                start = notnan(tr0)
                end = notnan(tr0, len(tr0) - 1, increment=-1)
                trace['calcium'] = np.atleast_2d(tr0[start:end + 1])

                trace['fps'] = fps
                data = c2s.preprocess([trace], fps=fps)
                data = c2s.predict(data, verbosity=0)

                tr0[start:end + 1] = data[0].pop('predictions')
                data[0]['rate_trace'] = tr0.T
                data[0].pop('calcium')
                data[0].pop('fps')

                yield data[0]