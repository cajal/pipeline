import datajoint as dj
from . import rf
schema = dj.schema('pipeline_cell_segmentation', locals())

@schema
class Labeller(dj.Lookup):
    definition = """
    # table that stores the identity of the stack labeller

    labeller_id     : int   # unique identifier of labeller
    ---
    name            : char(20) # real name of the labeller
    """

    @property
    def contents(self):
        yield from enumerate(['Fabian', 'Manolis', 'Jake', 'Shan'])

@schema
class ManualLocation(dj.Manual):
    definition = """
    ->rf.Scan
    ->rf.Session
    ->Labeller
    ---
    i               : int   # row location of cell
    j               : int   # col location of cell
    k               : int   # depth location of cell
    """

    def label(self, key):
        pass
