import datajoint as dj

schema = dj.schema('common', locals())

@schema
class Animal(dj.Manual):
    definition = None

@schema
class BrainSliceImage(dj.Manual):
    definition = None

@schema
class BrainSliceRegistration(dj.Manual):
    definition = None

@schema
class Injection(dj.Manual):
    definition = None

@schema
class MpSession(dj.Manual):
    definition = None

@schema
class MpSlice(dj.Manual):
    definition = None

@schema
class OpticalMovie(dj.Manual):
    definition = None

@schema
class OpticalSession(dj.Manual):
    definition = None

@schema
class TpPatch(dj.Manual):
    definition = None

@schema
class TpScan(dj.Manual):
    definition = None

@schema
class TpSession(dj.Manual):
    definition = None

@schema
class TpStack(dj.Manual):
    definition = None

@schema
class Virus(dj.Manual):
    definition = None

@schema
class WholeCell(dj.Manual):
    definition = None

@schema
class WholeCellSession(dj.Manual):
    definition = None

