""" Schemas for viral injections """
import numpy as np
import datajoint as dj
from commons import virus
from . import meso, reso, shared, mice
from .exceptions import PipelineException


schema = dj.schema('pipeline_injection', locals(), create_tables=False)
CURRENT_VERSION = 1


@schema
class Target(dj.Lookup):
    definition = """
    # target area list
    target                         : varchar(20)                       # abbreviation of brain areas
    ---
    """
    contents = [['OB'], ['V1'], ['LM/LI'], ['PM'], ['cingulate cortex'], ['SC'], ['substantial nigra'], 
                ['hippocampus'], ['LP'], ['dLGN'], ['vLGN'], ['subthalamic nuclei'], ['amygdala'], 
                ['striatum'], ['TRN']]


@schema
class InjectionSession(dj.Manual):
    definition = """
    # Injection Session
    -> mice.Mice
    injection_session=1           : tinyint unsigned                    # injection session order, differ only by AP and ML, not DV
    injection_datetime            : datetime                            # time of injection 'YYYY-MM-DD HH:MM:SS' 
    ---
    injection_session_notes=''    : varchar(255)                        # notes
    """


@schema
class InjectionSite(dj.Manual):
    definition = """
    # Injection sites information, all injections are based on intrinsic imaging results 
    ->InjectionSession
    injection_site=1              : tinyint unsigned                    # injection sites order
    ---
    -> [nullable] Target
    coord_ant_post                : decimal(3,2)                        # (mm) anterior posterior coordinate from transverse vein
    coord_med_lat                 : decimal(3,2)                        # (mm) medial lateral coordinate from midline
    coord_depth                   : decimal(3,2)                        # (mm) depth from brain surface
    coord_depth_initial           : decimal(3,2)                        # (mm) the furtherest depth the syringe was lowered at initially to create space for injection
    tip_size                      : tinyint unsigned                    # (um), 40um-60um
    -> [nullable] virus.Virus
    virus_total                   : tinyint unsigned                    # (nL) the total virus volume for each injection site
    virus_withdraw_speed=10       : tinyint unsigned                    # (nL/s) virus withdraw speed for syringe by nano pump
    virus_injection_speed=4       : tinyint unsigned                    # (nL/s) virus injection speed for syringe by nano pump
    visualize_level_drop          : bool                                # visualize whether or not virus level drop inside tip
    injection_site_notes=''       : varchar(255)                        # notes
    """