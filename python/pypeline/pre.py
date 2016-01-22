import datajoint as dj

schema = dj.schema('dimitri_pre', locals())

@schema
class AlignMotion(dj.Imported):
    definition = None

    def _make_tuples(self, key):
        raise NotImplementedError("This schema is populated from Matlab")

@schema
class AlignRaster(dj.Computed):
    definition = None

    def _make_tuples(self, key):
        raise NotImplementedError("This schema is populated from Matlab")

@schema
class Check(dj.Imported):
    definition = None

    def _make_tuples(self, key):
        raise NotImplementedError("This schema is populated from Matlab")

@schema
class ManualSegment(dj.Imported):
    definition = None

    def _make_tuples(self, key):
        raise NotImplementedError("This schema is populated from Matlab")

@schema
class ScanCheck(dj.Imported):
    definition = None

    def _make_tuples(self, key):
        raise NotImplementedError("This schema is populated from Matlab")

@schema
class ScanInfo(dj.Imported):
    definition = None

    def _make_tuples(self, key):
        raise NotImplementedError("This schema is populated from Matlab")
