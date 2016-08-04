_report_on = {
    'aod pipeline': ['aod_monet', 'aodpre'],
    'reso pipeline': ['pre', 'rf', 'trk', 'trippy', 'monet']
}

class PipelineException(Exception):
    def __init__(self, message, keys=None):
        # Call the base class constructor with the parameters it needs
        super(Exception, self).__init__(message)

        self.keys = keys

    def __str__(self):
        return """
        Pipeline Exception raised while processing {0}
        """.format(repr(self.keys))
