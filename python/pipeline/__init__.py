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
        

# ----------- loads local configuration from file ----------------
import os
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

