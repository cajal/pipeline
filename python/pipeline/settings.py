"""
Settings for DataJoint.
"""
import json
from collections import OrderedDict
from .exceptions import PipelineException
import collections
from pprint import pprint, pformat

LOCALCONFIG = 'pipeline_config.json'
GLOBALCONFIG = '.pipeline_config.json'
validators = collections.defaultdict(lambda: lambda value: True)

default = OrderedDict({
    'path.mounts': '/mnt/',
    'display.tracking': False
})


class Config(collections.MutableMapping):

    instance = None

    def __init__(self, *args, **kwargs):
            if not Config.instance:
                Config.instance = Config.__Config(*args, **kwargs)
            else:
                Config.instance._conf.update(dict(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __getitem__(self, item):
        return self.instance.__getitem__(item)

    def __setitem__(self, item, value):
        self.instance.__setitem__(item, value)

    def __str__(self):
        return pformat(self.instance._conf, indent=4)

    def __repr__(self):
        return self.__str__()

    def __delitem__(self, key):
        del self.instance._conf[key]

    def __iter__(self):
        return iter(self.instance._conf)

    def __len__(self):
        return len(self.instance._conf)


    class __Config:
        """
        Stores datajoint settings. Behaves like a dictionary, but applies validator functions
        when certain keys are set.

        The default parameters are stored in pipeline.settings.default . If a local config file
        exists, the settings specified in this file override the default settings.

        """

        def __init__(self, *args, **kwargs):
            self._conf = dict(default)
            self._conf.update(dict(*args, **kwargs))  # use the free update to set keys

        def __getitem__(self, key):
            return self._conf[key]

        def __setitem__(self, key, value):
            if isinstance(value, collections.Mapping):
                raise ValueError("Nested settings are not supported!")
            if validators[key](value):
                self._conf[key] = value
            else:
                raise PipelineException(u'Validator for {0:s} did not pass'.format(key, ))

        def save(self, filename=None):
            """
            Saves the settings in JSON format to the given file path.
            :param filename: filename of the local JSON settings file. If None, the local config file is used.
            """
            if filename is None:
                filename = LOCALCONFIG
            with open(filename, 'w') as fid:
                json.dump(self._conf, fid, indent=4)

        def load(self, filename):
            """
            Updates the setting from config file in JSON format.
            :param filename: filename of the local JSON settings file. If None, the local config file is used.
            """
            if filename is None:
                filename = LOCALCONFIG
            with open(filename, 'r') as fid:
                self._conf.update(json.load(fid))