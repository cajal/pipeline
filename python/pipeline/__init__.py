import os

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

class DataJointError(Exception):
    """
    Base class for errors specific to DataJoint internal operation.
    """
    pass

# ----------- loads local configuration from file ----------------
from .settings import Config, LOCALCONFIG, GLOBALCONFIG
config = Config()


if os.path.exists(LOCALCONFIG):  # pragma: no cover
    local_config_file = os.path.expanduser(LOCALCONFIG)
    print("Loading local settings from {0:s}".format(local_config_file))
    config.load(local_config_file)
elif os.path.exists(os.path.expanduser('~/') + GLOBALCONFIG):  # pragma: no cover
    local_config_file = os.path.expanduser('~/') + GLOBALCONFIG
    print("Loading local settings from {0:s}".format(local_config_file))
    config.load(local_config_file)
else:
    print("""Cannot find configuration settings. Using default configuration. To change that, either
    * modify the local copy of %s that pipeline just saved for you
    * put a file named %s with the same configuration format in your home
          """ % (LOCALCONFIG, GLOBALCONFIG))
    local_config_file = os.path.expanduser(LOCALCONFIG)
    config.save(local_config_file)

