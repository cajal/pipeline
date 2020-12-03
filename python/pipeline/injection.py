""" Schemas for viral injections """
import numpy as np
import datajoint as dj
from commons import virus
from . import mice
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
    injection_session=1           : tinyint unsigned                    # injection session index
    injection_datetime            : datetime                            # time of injection 'YYYY-MM-DD HH:MM:SS' 
    ---
    injection_session_notes=''    : varchar(255)                        # notes
    """


@schema
class InjectionSite(dj.Manual):
    definition = """
    # Injection site information
    ->InjectionSession
    injection_site=1              : tinyint unsigned                    # injection site index
    ---
    -> [nullable] Target
    coord_ant_post                : decimal(3,2)                        # (mm) anterior posterior coordinate from transverse vein
    coord_med_lat                 : decimal(3,2)                        # (mm) medial lateral coordinate from midline
    coord_depth                   : decimal(3,2)                        # (mm) depth from brain surface
    coord_depth_initial           : decimal(3,2)                        # (mm) the deepest initial penetration of the pipette
    tip_size                      : tinyint unsigned                    # (um) 
    -> [nullable] virus.Virus
    virus_total                   : tinyint unsigned                    # (nL) the total virus volume for each injection site
    virus_injection_speed=4       : tinyint unsigned                    # (nL/s) speed of injection with nano pump
    visualize_confirmation        : bool                                # Was virus extrusion visually confirmed?
    injection_site_notes=''       : varchar(255)                        # notes
    """
