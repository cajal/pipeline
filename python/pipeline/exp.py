import datajoint as dj
from commons import mice
schema = dj.schema('pipeline_experimental_data', locals())

# TODO:
# * should fluorophore move from Session to Channel?
# * what about rf.Eye
# * what about rf.EyeFrame
# * what about rf.Locate
# * what about rf.Motion3D
# * what about Sync
# * what about rf.FlashMAp



@schema
class Lens(dj.Lookup):
    definition = """
    # objective lens list
    setup           : char(4)                # two-photon setup
    lens            : char(4)                # objective lens
    ---

    """

@schema
class FOV(dj.Lookup):
    definition = """
    # field-of-view sizes for all lenses and magnifications
    -> rf.Lens
    mag             : decimal(5,2)           # ScanImage zoom factor
    ---
    height                      : decimal(5,1)                  # measured width of field of view along axis of pipette (medial/lateral on mouse)
    width                       : decimal(5,1)                  # measured width of field of view perpendicular to pipette (rostral/caudal on mouse)
    take=1                      : tinyint                       #
    fov_date                    : date                          # fov measurement date
    INDEX(lens)
    """

@schema
class Session(dj.Manual):
    definition = """
    # session

    -> mice.Mice
    session         : smallint               # session index
    ---
    -> rf.Fluorophore
    -> rf.Lens
    session_date                : date                          # date
    scan_path                   : varchar(255)                  # file path for TIFF stacks
    hd5_path                    : varchar(255)                  # file path for HD5 files
    file_base                   : varchar(255)                  # file base name
    anesthesia="awake"          : enum('isoflurane','fentanyl','awake') # per protocol
    craniotomy_notes            : varchar(4095)                 # free-text notes
    session_notes               : varchar(4095)                 # free-text notes
    session_ts=CURRENT_TIMESTAMP: timestamp                     # automatic
    INDEX(lens)
    """

@schema
class Site(dj.Manual):
    definition = """
    # sites grouping several scans

    site: smallint  # site number
    -----
    """

@schema
class Scan(dj.Manual):
    definition = """
    # scanimage scan info
    -> Session
    scan_idx        : smallint               # number of TIFF stack file
    ---
    -> Site
    file_num                    : smallint                      # number of HD5 file
    depth=0                     : int                           # manual depth measurement
    laser_wavelength            : float                         # (nm)
    laser_power                 : float                         # (mW) to brain
    cortical_area="unknown"     : enum('other','unknown','V1','LM','AL','PM') # Location of scan
    scan_notes                  : varchar(4095)                 # free-notes
    scan_ts=CURRENT_TIMESTAMP   : timestamp                     # don't edit
    """

@schema
class Stack(dj.Manual):
    definition = """
    # scanimage scan info for structural stacks
    -> Session
    stack_idx: smallint  # number of TIFF stack file
    ---
    -> Site
    bottom_z: int  # z location at bottom of the stack
    surf_z: int  # z location of surface
    laser_wavelength: int  # (nm)
    laser_power: int  # (mW) to brain
    stack_notes: varchar(4095)  # free-notes
    scan_ts = CURRENT_TIMESTAMP: timestamp  # don't edit
    """

@schema
class StackInfo(dj.Imported):
    definition = """
    # header information
    -> Stack
    ---
    nchannels                   : tinyint                       # number of recorded channels
    nslices                     : int                           # number of slices (hStackManager_numSlices)
    frames_per_slice            : int                           # number of frames per slice (hStackManager_framesPerSlice)
    px_width                    : smallint                      # pixels per line
    px_height                   : smallint                      # lines per frame
    zoom                        : decimal(4,1)                  # zoom factor
    um_width                    : float                         # width in microns
    um_height                   : float                         # height in microns
    slice_pitch                 : float                         # (um) distance between slices (hStackManager_stackZStepSize)

    """

@schema
class Sync(dj.Imported):
    definition = """
    # mapping of h5,ca imaging, and vis stim clocks
    -> Scan
    ---
    -> psy.Session
    first_trial                 : int                           # first trial in recording
    last_trial                  : int                           # last trial in recording
    vis_time                    : longblob                      # h5 patch data sample times on visual stimulus (Mac Pro) clock
    frame_times                 : longblob                      # times of frames and slices
    sync_ts=CURRENT_TIMESTAMP   : timestamp                     # automatic

    """