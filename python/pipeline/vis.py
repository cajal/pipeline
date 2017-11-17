"""
This schema copies recent data from common_psy for uploaded to the cloud
"""
import datajoint as dj
from . import mice  # needed for referencing
import numpy as np

schema = dj.schema('pipeline_vis', locals())

schema.spawn_missing_classes()


@schema
class Session(dj.Manual):
    definition = """  # Visual stimulus session, populated by the stimulus program.
    -> mice.Mice
    psy_id               : smallint unsigned            # unique psy session number
    ---
    stimulus="grating"   : varchar(255)                 # experiment type
    monitor_distance     : float                        # (cm) eye-to-monitor distance
    monitor_size=19      : float                        # (inches) size diagonal dimension
    monitor_aspect=1.25  : float                        # physical aspect ratio of monitor
    resolution_x=1280    : smallint                     # (pixels)
    resolution_y=1024    : smallint                     # display resolution along y
    psy_ts=CURRENT_TIMESTAMP : timestamp                    # automatic
    """


@schema
class Condition(dj.Manual):
    definition = """  # trial condition -- one condition per trial. All stimulus conditions refer to Condition.
    -> Session
    cond_idx             : smallint unsigned            # condition index
    """


@schema
class Trial(dj.Manual):
    definition = """  #  visual stimulus trial
    -> Session
    trial_idx            : int                          # trial index within sessions
    ---
    -> Condition
    flip_times           : mediumblob                   # (s) row array of flip times
    last_flip_count      : int unsigned                 # the last flip number in this trial
    trial_ts=CURRENT_TIMESTAMP : timestamp                    # automatic
    """


@schema
class Movie(dj.Lookup):
    definition = """   # movies used for generating clips and stills
    movie_name           : char(8)                      # short movie title
    ---
    path                 : varchar(255)                 #
    movie_class          : enum('mousecam','object3d','madmax') #
    original_file        : varchar(255)                 #
    file_template        : varchar(255)                 # filename template with full path
    file_duration        : float                        # (s) duration of each file (must be equal)
    codec="-c:v libx264 -preset slow -crf 5" : varchar(255)                 #
    movie_description    : varchar(255)                 # full movie title
    """

    class Still(dj.Part):
        definition = """  #cached still frames from the movie
        -> Movie
        still_id             : int                          # ids of still images from the movie
        ---
        still_frame          : longblob                     # uint8 grayscale movie
        """

    class Clip(dj.Part):
        definition = """  # clips from movies
        -> Movie
        clip_number          : int                          # clip index
        ---
        file_name            : varchar(255)                 # full file name
        clip                 : longblob                     #
        """


@schema
class MovieSeqCond(dj.Manual):
    definition = """  # random sequences of still frames
    -> Condition
    ---
    -> Movie
    rng_seed             : int                          # random number generator seed
    pre_blank_period     : float                        # (s)
    duration             : float                        # (s) of each still
    seq_length           : smallint                     # number of frames in the sequence
    movie_still_ids      : blob                         # sequence of stills
    """


@schema
class MovieStillCond(dj.Manual):
    definition = """
    # a still frame condition
    -> Condition
    ---
    -> Movie.Still
    pre_blank_period     : float                        # (s)
    duration             : float                        # (s)
    """


@schema
class MovieClipCond(dj.Manual):
    definition = """  # movie clip conditions
    -> Condition
    ---
    -> Movie.Clip
    cut_after            : float                        # (s) cuts off after this duration
    """


@schema
class MonetLookup(dj.Lookup):
    definition = """  # cached noise maps to save computation time
    moving_noise_version : smallint                     # algorithm version; increment when code changes
    moving_noise_paramhash : char(10)                     # hash of the lookup parameters
    ---
    params               : blob                         # cell array of params
    cached_movie         : longblob                     # [y,x,frames]
    moving_noise_lookup_ts=CURRENT_TIMESTAMP : timestamp     # automatic
    """


@schema
class Monet(dj.Manual):
    definition = """  # pink noise with periods of motion and orientation$
    -> Condition
    ---
    -> MonetLookup
    rng_seed             : double                       # random number generate seed
    luminance            : float                        # (cd/m^2)
    contrast             : float                        # michelson contrast
    tex_ydim             : smallint                     # (pixels) texture dimension
    tex_xdim             : smallint                     # (pixels) texture dimension
    spatial_freq_half    : float                        # (cy/deg) spatial frequency modulated to 50 percent
    spatial_freq_stop    : float                        # (cy/deg), spatial lowpass cutoff
    temp_bandwidth       : float                        # (Hz) temporal bandwidth of the stimulus
    ori_on_secs          : float                        # seconds of movement and orientation
    ori_off_secs         : float                        # seconds without movement
    n_dirs               : smallint                     # number of directions
    ori_bands            : tinyint                      # orientation width expressed in units of 2*pi/n_dirs
    ori_modulation       : float                        # mixin-coefficient of orientation biased noise
    speed                : float                        # (degrees/s)
    frame_downsample     : tinyint                      # 1=60 fps, 2=30 fps, 3=20 fps, 4=15 fps, etc
    """


@schema
class Trippy(dj.Manual):
    definition = """
    # randomized curvy dynamic gratings
    -> Condition
    ---
    version              : tinyint                      # trippy version
    rng_seed             : double                       # random number generate seed
    packed_phase_movie   : longblob                     # phase movie before spatial and temporal interpolation
    luminance            : float                        # (cd/m^2)
    contrast             : float                        # michelson contrast
    tex_ydim             : smallint                     # (pixels) texture dimension
    tex_xdim             : smallint                     # (pixels) texture dimension
    duration             : float                        # (s) trial duration
    frame_downsample     : tinyint                      # 1=60 fps, 2=30 fps, 3=20 fps, 4=15 fps, etc
    xnodes               : tinyint                      # x dimension of low-res phase movie
    ynodes               : tinyint                      # y dimension of low-res phase movie
    up_factor            : tinyint                      # spatial upscale factor
    temp_freq            : float                        # (Hz) temporal frequency if the phase pattern were static
    temp_kernel_length   : smallint                     # length of Hanning kernel used for temporal filter. Controls the rate of change of the phase pattern.
    spatial_freq         : float                        # (cy/degree) approximate max. The actual frequencies may be higher.
    """


@schema
class FlashingBar(dj.Manual):
    definition = """  # flashing bar
    -> Condition
    ---
    pre_blank            : float                        # (s) blank period preceding trials
    luminance            : float                        # (cd/m^2) mid-value luminance
    contrast             : float                        # (0-1) Michelson contrast of values 0..255
    bg_color             : tinyint unsigned             # background color 1-254
    orientation          : decimal(4,1)                 # (degrees) 0=horizontal,  90=vertical
    offset               : float                        # normalized by half-diagonal
    width                : float                        # normalized by half-diagonal
    trial_duration       : float                        # (s) ON time of flashing bar
    pattern_frequency    : float                        # (Hz) will be rounded to the nearest fraction of fps
    """


@schema
class Grating(dj.Manual):
    definition = """  # drifting gratings with apertures
    -> Condition
    ---
    direction            : decimal(4,1)                 # 0-360 degrees
    spatial_freq         : decimal(4,2)                 # cycles/degree
    temp_freq            : decimal(4,2)                 # Hz
    pre_blank = 0        : float                        # (s) blank period preceding trials
    luminance            : float                        # cd/m^2 mean
    contrast             : float                        # Michelson contrast 0-1
    aperture_radius = 0  : float                        # in units of half-diagonal, 0=no aperture
    aperture_x = 0       : float                        # aperture x coordinate, in units of half-diagonal, 0 = center
    aperture_y = 0       : float                        # aperture y coordinate, in units of half-diagonal, 0 = center
    grating              : enum('sqr','sin')            # sinusoidal or square, etc.
    init_phase           : float                        # 0..1
    trial_duration       : float                        # s, does not include pre_blank duration
    phase2_fraction = 0  : float                        # fraction of trial spent in phase 2
    phase2_temp_freq = 0 : float                        # (Hz)
    second_photodiode = 0 : tinyint                      # 1=paint a photodiode patch in the upper right corner
    second_photodiode_time = 0.0 : decimal(4,1)                 # time delay of the second photodiode relative to the stimulus onset
    """


@schema
class MadAlexFrameSet(dj.Lookup):
    definition = """
    # holds training and testing frames for MadAlex stimulus

    frame_set_id       : tinyint # identifier for the frame set
    ---
    uniques            : longblob # integer array with unique image ids
    repeats_shared     : longblob # integer array with repeated image ids
    repeats_nonshared  : longblob # integer array with repeated image ids
    """

    @property
    def contents(self):
        np.random.seed(1706)
        all_images = np.arange(1, 12001, dtype=np.int32)
        ri = np.random.randint
        rp = np.random.permutation
        repeats_shared = all_images[ri(0, len(all_images), size=25)]

        all_images = np.setdiff1d(all_images, repeats_shared)
        for frame_set_id in range(1,4):
            uniques = all_images[ri(0, len(all_images), size=1400)]
            all_images = np.setdiff1d(all_images, uniques)

            repeats_nonshared = all_images[ri(0, len(all_images), size=25)]
            all_images = np.setdiff1d(all_images, repeats_nonshared)

            yield dict(frame_set_id=frame_set_id,
                       uniques=uniques,
                       repeats_shared=repeats_shared,
                       repeats_nonshared=repeats_nonshared)


@schema
class MatisseCenterLoc(dj.Lookup):
    definition = """  # location of center for Matisse scans.
        id          : smallint   # one is default, 2 is online change.
        ---
        x_loc              : decimal(4,3)
        y_loc              : decimal(4,3)
        r                  : decimal(4,3)
        """
    contents = [
        [1, 0, 0, 0.25], [2, 0, 0, 0.2]
    ]

def migrate():
    from .legacy import psy
    from . import experiment

    # copy basic structure: Session, Condition, Trial
    sessions = psy.Session() & (experiment.Session() & 'session_date>"2016-02-02"' & 'animal_id>1000')
    trial = psy.Trial() & sessions
    condition = psy.Condition() & sessions
    Session().insert(sessions - Session())
    Condition().insert(condition - Condition())
    Trial().insert(trial - Trial())

    # copy Monet
    monet = psy.MovingNoise() & sessions
    monetLookup = psy.MovingNoiseLookup() & monet
    MonetLookup().insert(monetLookup - MonetLookup())
    Monet().insert(monet - Monet())

    # copy Trippy
    trippy = psy.Trippy() & sessions
    Trippy().insert(trippy - Trippy())

    # copy MovieClip and MovieStill
    Movie().insert(psy.MovieInfo() - Movie())
    Movie.Clip().insert(psy.MovieClipStore() - Movie.Clip())
    Movie.Still().insert(psy.MovieStill() - Movie.Still())
    MovieClipCond().insert((psy.MovieClipCond() - MovieClipCond()) & Condition())
    MovieStillCond().insert((psy.MovieStillCond() - MovieStillCond()) & Condition())
    MovieClipCond().insert(
        psy.MadMax().proj('clip_number', 'cut_after', movie_name="'MadMax'") & Condition() - MovieClipCond())
    MovieSeqCond().insert(psy.MovieSeqCond() & Condition())
    FlashingBar().insert(psy.FlashingBar() & Condition())
    Grating().insert(psy.Grating() & Condition())

    orphan_conditions = (
        Condition() - Monet() - Trippy() - MovieClipCond() -
        MovieStillCond() - MovieSeqCond() - FlashingBar() - Grating())


schema.spawn_missing_classes()
