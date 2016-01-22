import datajoint as dj
from . import rf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

schema = dj.schema('microns_nmf', locals())


@schema
class Settings(dj.Lookup):
    definition = None


@schema
class Segment(dj.Imported):
    definition = None

    def _make_tuples(self, key):
        raise NotImplementedError('This table is populated from matlab.')


@schema
class SegmentationTile(dj.Computed):
    definition = None

    def _make_tuples(self, key):
        raise NotImplementedError('This table is populated from matlab.')


@schema
class SelectedMask(dj.Computed):
    definition = None

    def _make_tuples(self, key):
        raise NotImplementedError('This table is populated from matlab.')


@schema
class MaskAverageTrace(dj.Computed):
    definition = None

    def _make_tuples(self, key):
        raise NotImplementedError('This table is populated from matlab.')