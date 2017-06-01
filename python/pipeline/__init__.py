import os

_report_on = {
    'aod pipeline': ['aod_monet', 'aodpre'],
    'reso pipeline': ['pre', 'rf', 'trk', 'trippy', 'monet']
}

#--- switch matplotlib backend if there is no way to display things.
import matplotlib
try:    
    from tkinter import TclError
    try:
        import matplotlib.pyplot as plt
        del plt  # don't really wanted to import it, just testing
    except TclError:
        print('No display found. Switching matplotlib backend to "Agg"')
        matplotlib.use('Agg', warn=False, force=True)
except ImportError:
    matplotlib.use('Agg', warn=False, force=True)
        

class PipelineException(Exception):
    """Base pipeline exception. Prints the message plus any passed info."""
    def __init__(self, message, info=None):
        info_message = '\nError info: ' + repr(info) if info else ''
        super().__init__(message + info_message)
        self.info = info


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

