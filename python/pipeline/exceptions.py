class PipelineException(Exception):
    """Base pipeline exception. Prints the message plus any specific info."""
    def __init__(self, message, info=None):
        info_message = '\nError info: ' + repr(info) if info else ''
        super().__init__(message + info_message)
        self.info = info