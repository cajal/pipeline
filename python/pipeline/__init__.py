import os

_report_on = {
    'aod pipeline': ['aod_monet', 'aodpre'],
    'reso pipeline': ['pre', 'rf', 'trk', 'trippy', 'monet']
}


#--- switch matplotlib backend if there is no way to display things.
cmd = 'python3 -c "import matplotlib.pyplot as plt; plt.figure()" 2> /dev/null'
if os.system(cmd): # if command fails
    print('No display found. Switching matplotlib backend to "Agg"')
    import matplotlib; matplotlib.use('Agg'); del matplotlib


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