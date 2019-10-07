import datajoint as dj
from . import reso
from . import experiment
from . import mice
from . import  notify, shared
import scanreader
from datajoint.jobs import key_hash
import matplotlib.pyplot as plt
import numpy as np
from .utils import galvo_corrections, signal, quality, mask_classification, performance
from .exceptions import PipelineException
import os
import time
import multiprocessing as mp
from .exceptions import PipelineException
from .utils.signal import mirrconv
import tifffile
import cv2

patchtable=dj.schema("yizhou_patch",locals(),create_tables=True)

#right arm, about patch 

@patchtable
class PatchSession(dj.Manual): 
    definition = """ 
        -> mice.Mice    
        patch_session:int #session is different from reso session.
        ---
        # session here is one experiment, most likely, in one day.
        folder='':varchar(256) # the path to the folder that contains file of the cell.
        """
@patchtable
class Cell(dj.Manual):
    definition = """
    -> PatchSession #include animal id etc, patch session.
    ---
    notes='':varchar(256) # the notes about the cell. 
    # where does the dates etc go? notes?
    """
@patchtable
class Recording(dj.Manual): 
    definition = """
    -> Cell #include animal id etc, patch session, notes about the cell.
    ---
    filename='':varchar(256) 
    i1:external
    command:external
    vm:external
    frametimes:external
    ephystimes:external
    igain:int
    vgain:int
    ilowpass:int
    vlowpass:int
    vhighpass:int

    """
@patchtable
class Patchspikes(dj.Manual):
    definition = """ 
        -> Recording #which has animal, patch session, notes, raw file, setting, data.
        ---
        spike_ts:external # timestamps. on same clock as trace.
        """
    


    
    























#left arm, about imaging 

@patchtable
class Scaninfo(dj.Manual):
    definition = """   
    # those will be cp from reso.scaninfo, but only contain 1 entry
    animal_id:int
    session:int
    scan_idx:int
    pipe_version:int
    field:int #from correctionchannel.  field was a prim key
    ---
    #
    nfields                 : tinyint           # number of fields
    nchannels               : tinyint           # number of channels
    nframes                 : int               # number of recorded frames
    nframes_requested       : int               # number of requested frames (from header)
    px_height               : smallint          # lines per frame
    px_width                : smallint          # pixels per line
    um_height               : float             # height in microns
    um_width                : float             # width in microns
    x                       : float             # (um) center of scan in the motor coordinate system
    y                       : float             # (um) center of scan in the motor coordinate system
    fps                     : float             # (Hz) frames per second
    zoom                    : decimal(5,2)      # zoom factor
    bidirectional           : boolean           # true = bidirectional scanning
    usecs_per_line          : float             # microseconds per scan line
    fill_fraction           : float             # raster scan temporal fill fraction (see scanimage)
    valid_depth       : boolean           # whether depth has been manually check
    # above cp from reso.scaninfo
    channel:int  # from reso.correctionchannel, just one value so place it here.
    z:float # from scaninfo.field.
    delay_image:longblob #from scaninfo.field
    
    """    
        
@patchtable
class Segmentationtask(dj.Manual):
    definition = """ # defines the target of segmentation and the channel to use
    -> experiment.Scan
    -> shared.Field
    -> shared.Channel
    -> shared.SegmentationMethod
    ---
    -> experiment.Compartment
    """
    def fill(self, key, channel=1, segmentation_method=6, compartment='soma'):
        for field_key in (Field() & key).fetch(dj.key):
            tuple_ = {**field_key, 'channel': channel, 'compartment': compartment,
                      'segmentation_method': segmentation_method}
            self.insert1(tuple_, ignore_extra_fields=True, skip_duplicates=True)
    def estimate_num_components(self):
        """ Estimates the number of components per field using simple rules of thumb.

        For somatic scans, estimate number of neurons based on:
        (100x100x100)um^3 = 1e6 um^3 -> 100 neurons; (1x1x1)mm^3 = 1e9 um^3 -> 100K neurons

        For axonal/dendritic scans, just ten times our estimate of neurons.

        :returns: Number of components
        :rtype: int
        """
        # Get field dimensions (in microns)
        scan = (Scaninfo() & self & {'pipe_version': CURRENT_VERSION})
        field_height, field_width = scan.fetch1('um_height', 'um_width')
        field_thickness = 10  # assumption
        field_volume = field_width * field_height * field_thickness
        # Estimate number of components
        compartment = self.fetch1('compartment')
        if compartment == 'soma':
            num_components = field_volume * 0.0001
        elif compartment == 'axon':
            num_components = field_volume * 0.0005  # five times as many neurons
        elif compartment == 'bouton':
            num_components = field_volume * 0.001   # 10 times as many neurons
        else:
            PipelineException("Compartment type '{}' not recognized".format(compartment))
        return int(round(num_components))
@patchtable
class DoNotSegment(dj.Manual):
    definition = """ # field/channels that should not be segmented (used for web interface only)

    -> experiment.Scan
    -> shared.Field
    -> shared.Channel
    """
        
@patchtable
class Rastercorrection(dj.Computed):
    definition = """ 
                             # animal_id, session, scan_idx, version
    -> Scaninfo
    # need correction channel, now the correction channel channel is in scaninfo as col, not an independent table.
    # now using the first channel to correct, normally its the green channel.
    ---
    raster_template     : longblob      # average frame from the middle of the movie
    raster_phase        : float         # difference between expected and recorded scan angle
    """    
    def make(self, key):
        from scipy.signal import tukey
        # Read the scan
        fps=(Scaninfo() & key).fetch1('fps')
        print("fps is ",fps)
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        print(scan_filename)
        print(scanreader.core.expand_wildcard(scan_filename))
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)
        # Select correction channel
        channel =  (Scaninfo()&key).fetch1('channel')-1
        print(key)
        field_id = key['field'] - 1
        # Load some frames from the middle of the scan
        middle_frame =  int(np.floor(scan.num_frames / 2))
        frames = slice(max(middle_frame - int(60*fps), 0), middle_frame + int(60*fps))
        startts=time.time()
        import tifffile
        scan_filename=scanreader.core.expand_wildcard(scan_filename)[0]
        # one recording session has one tiff
        f=tifffile.TiffFile(scan_filename)
        a=f.asarray(frames)
        print('pipe, a shape,', a.shape)
        mini_scan=np.moveaxis(a,0,2) #(f,x,y) to (y,x,f)
        print('pipe, mini_scan shape t1,', mini_scan.shape)
        mini_scan = mini_scan.astype(np.float32, copy=False)
#         mini_scan = scan[field_id, :, :, channel, frames]
        print('the time takes to get miniscan is:')
        print(time.time()-startts)
        # Create results tuple
        tuple_ = key.copy()
        # Create template (average frame tapered to avoid edge artifacts)
        taper = np.sqrt(np.outer(tukey(scan.image_height, 0.4),
                                 tukey(scan.image_width, 0.4)))
        anscombed = 2 * np.sqrt(mini_scan - mini_scan.min() + 3 / 8)  # anscombe transform
        template = np.mean(anscombed, axis=-1) * taper
        tuple_['raster_template'] = template
        # Compute raster correction parameters
        if scan.is_bidirectional:
            tuple_['raster_phase'] = galvo_corrections.compute_raster_phase(template,
                                                         scan.temporal_fill_fraction)
        else:
            tuple_['raster_phase'] = 0
        # Insert
        self.insert1(tuple_)
    def get_correct_raster(self):
        """ Returns a function to perform raster correction on the scan. """
        raster_phase = self.fetch1('raster_phase')
        fill_fraction = (Scaninfo() & self).fetch1('fill_fraction')
        if abs(raster_phase) < 1e-7:
            correct_raster = lambda scan: scan.astype(np.float32, copy=False)
        else:
            correct_raster = lambda scan: galvo_corrections.correct_raster(scan,
                                                             raster_phase, fill_fraction)
        return correct_raster

        
        
@patchtable
class  MotionCorrection(dj.Computed):
    definition = """ 
        ->Rastercorrection
        ---
        motion_template                 : longblob      # image used as alignment template
        y_shifts                        : longblob      # (pixels) y motion correction shifts
        x_shifts                        : longblob      # (pixels) x motion correction shifts
        y_std                           : float         # (pixels) standard deviation of y shifts
        x_std                           : float         # (pixels) standard deviation of x shifts
        outlier_frames                  : longblob      # mask with true for frames with outlier shifts (already corrected)
        align_time=CURRENT_TIMESTAMP    : timestamp     # automatic
        
    """
    
    def make(self, key):
        """Computes the motion shifts per frame needed to correct the scan."""
        from scipy import ndimage
        from pipeline.utils import galvo_corrections, signal, quality, mask_classification, performance
        import pyfftw
        from imreg_dft import utils
        import numpy as np
        from scipy import interpolate  as interp
        from scipy import signal
        from scipy import ndimage
        import tifffile
        # Read the scan
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)
        # Get some params
        px_height, px_width = (Scaninfo() & key).fetch1('px_height', 'px_width')
        channel = (Scaninfo() & key).fetch1('channel') - 1
        fps=(Scaninfo() & key).fetch1('fps')
        field_id = key['field'] - 1
        # Load some frames from middle of scan to compute template
        skip_rows = int(round(px_height * 0.10))  
        # we discard some rows/cols to avoid edge artifacts
        # so actually the motion correction is only looking at the center 80%
        # tested zero padding instead of taper, similar results
        # zero padding can preserve more data, use full mini scan 
        # taper is applied to mini scan, so less data than miniscan
        # i guess best way is to apply taper to full scan, than zero pad
        skip_cols = int(round(px_width * 0.10))
        middle_frame = int(np.floor(scan.num_frames / 2))
        fromframe=int(max(middle_frame - 100*fps, 0))
        toframe=int(middle_frame + 100*fps)
        print("gettign the frames in between ",fromframe,toframe)
        starttime=time.time()
        scan_filename=scanreader.core.expand_wildcard(scan_filename)[0]
        nchan=(Scaninfo()&key).fetch1('nchannels')
        correct_raster = (Rastercorrection() & key).get_correct_raster()
        # one recording session has one tiff
        f=tifffile.TiffFile(scan_filename)
        wholescan=f.asarray()# its the full scan
        wholescan=np.moveaxis(wholescan,0,2)#(f,x,y) to (y,x,f)
        # if there are multi channels, use the first channle (green)
        if nchan!=1:
            wholescan=wholescan[:,:,::nchan]
            # the channel frames are interlaved. green and red channel.
            # so ::nch is equal to 0::nch, taking all frames from first ch
        wholescan = correct_raster(wholescan)
        mini_scan=wholescan[:,:,fromframe:toframe] # now its mini scan temporally
        print('pipe, mini_scan shape ,', mini_scan.shape)
        mini_scan=mini_scan[skip_rows: -skip_rows, skip_cols: -skip_cols,:] # now its mini both temporally and spatially.
        # right now the mini scan is a smaller frame, and only contain frames about several miniutes from the center.
        mini_scan = mini_scan.astype(np.float32, copy=False)
        # Create template
        mini_scan = 2 * np.sqrt(mini_scan - mini_scan.min() + 3 / 8)  # *
        template = np.mean(mini_scan, axis=-1)
        print('pipe, tem shape,', template.shape)
        template = ndimage.gaussian_filter(template, 0.7)  # **
        wholescan=wholescan[skip_rows: -skip_rows, skip_cols: -skip_cols,:]#prepare the whole scan into same shape as template.
        # * Anscombe tranform to normalize noise, increase contrast and decrease outliers' leverage
        # ** Small amount of gaussian smoothing to get rid of high frequency noise
        # now, had template
        # template is avg of frames from center of scan, both spatially and temporally, with o.7 gaussian smoothed.
    
        # with the template, start motion correction
        def yccompute_motion_shifts(data,template,method=1, in_place=True, num_threads=8):
            """ Compute shifts in y and x for rigid subpixel motion correction.
            :param np.array template: 2-d template image. Each frame in scan is aligned to this.
            for the mechanism, search for sub pixel shift/registration based on fft. it is a typical pixel registration method used in motion correction. for information rich image, gives promising results.
            """
            print(data.shape)
            # Get some paramsdata
            image_height, image_width, num_frames = data.shape
            taper = np.outer(signal.tukey(image_height, 0.2), signal.tukey(image_width, 0.2))
            taperlarge=(np.outer(signal.tukey(image_height, 0.6), signal.tukey(image_width, 0.6)))
            # Prepare fftw
            frame = pyfftw.empty_aligned((image_height, image_width), dtype='complex64')
            fft = pyfftw.builders.fft2(frame, threads=num_threads, overwrite_input=in_place,
                                       avoid_copy=True)
            ifft = pyfftw.builders.ifft2(frame, threads=num_threads, overwrite_input=in_place,
                                         avoid_copy=True)
        
############ testing methods about windows functions.#######################
# for more information, search for window function and sliding window filtering####
# the idea is, for this high frequency sampling, single frames have very low snr
# think about like we need longger exposer to photo a night sky
# thus, using a window function to combine frames together to register is necessray, for sure.
# about different window functions. rectangle (simple avergaing) gives a slight enphasize on the center frame in fft. and other window functions could either focus more on the center, or more flat. (search for fft window function)
# ideally, we want to correct the center frame so we want focus more on the center
# about window size. larger window gives a stablier image, more signal. smaller winodw focus more on the center, but gives a crappy image.
# about taking mean or max. i dont have a clear understanding of how the data look at ( should have spent more time on data first ), but seems max gives higher snr in a smaller window. maybe because noise is possion with a smaller max, but signal has a much higher max ( when not firing)
# to do, window function/window size/ winodow mean max/preprosess the winodw
# preprocess, including smoothing, normalizing, transformations,
# mean idea, play with contrast, and try to make different window similar in value
# ( first test normalizing the whole data)
            if method==1: # using a pretty large rectangle window, with gaussian
                template_freq = fft(template * taper).conj() # we only need the conjugate
                abs_template_freq = abs(template_freq)
                eps = abs_template_freq.max() * 1e-15
                # Compute subpixel shifts per image
                y_shifts = np.empty(num_frames)
                x_shifts = np.empty(num_frames)
                starttime=time.time()
                for i in range(num_frames):
                    theimage=(
                    +np.max(data[:, :, max(0,int(i-0.05*fps)):min(num_frames,int(i+0.05*fps))],2)
                    *1
                    * taper)
                    image_freq=fft(ndimage.gaussian_filter(theimage,0.7))
                    cross_power = (image_freq * template_freq) / (abs(image_freq) * abs_template_freq + eps)
                    shifted_cross_power = np.fft.fftshift(abs(ifft(cross_power)))
                    # Get best shift
                    shifts = np.unravel_index(np.argmax(taperlarge*shifted_cross_power), shifted_cross_power.shape)
                    # here using a larger taper to focus on near shift 
                    # instead of moving large distance to match the avg frames of block with the template
                    #     print(shifts)
                    shifts = utils._interpolate(shifted_cross_power, shifts, rad=3)
                    # Map back to deviations from center
                    y_shifts[i] = shifts[0] - image_height // 2
                    x_shifts[i] = shifts[1] - image_width // 2
                    if i%10000==0:
                        print('now at ',i,time.time()-starttime)
                return y_shifts, x_shifts
                
            elif method==2: # using single frame
                template_freq = fft(template * taper).conj() # we only need the conjugate
                abs_template_freq = abs(template_freq)
                eps = abs_template_freq.max() * 1e-15
                # Compute subpixel shifts per image
                y_shifts = np.empty(num_frames)
                x_shifts = np.empty(num_frames)
                starttime=time.time()
                for i in range(num_frames):
                    theimage = (
    #                     np.mean(data[:, :, max(0,int(i-1)):min(num_frames,int(i+1))],2)
    #                     *1
    #                     +np.mean(data[:, :, max(0,int(i-0.01*fps)):min(num_frames,int(i+0.01*fps))],2)
    #                     *1/3
    #                     * taper
                        (data[:,:,i])
                    )
                    theimage=(theimage-np.min(theimage))/(np.max(theimage)-np.min(theimage))
                    image_freq=fft(ndimage.gaussian_filter(theimage,0.7))
    #                     view=np.zeros((image_height, image_width))
    #                     for m in range(image_height):
    #                         for n in range(image_width):
    #                                 view[m,n]=((m-image_height/2)**2+(n-image_width/2)**2)**0.5
    #                     view[image_height,image_width]=.9
                    cross_power = (image_freq * template_freq) / (abs(image_freq) * abs_template_freq + eps)
                    shifted_cross_power = np.fft.fftshift(abs(ifft(cross_power)))
                    # Get best shift
                    shifts = np.unravel_index(np.argmax(shifted_cross_power), shifted_cross_power.shape)
                    # here using a larger taper to focus on near shift 
                    # instead of moving large distance to match the avg frames of block with the template
                    #     print(shifts)
                    shifts = utils._interpolate(shifted_cross_power, shifts, rad=3)
                    # Map back to deviations from center
                    y_shifts[i] = shifts[0] - image_height // 2
                    x_shifts[i] = shifts[1] - image_width // 2
                    if i%10000==0:
                        print('now at ',i,time.time()-starttime)
                return y_shifts, x_shifts
        
            elif method==3: # using zero padding
                padwid=3
                frame = pyfftw.empty_aligned((image_height+2*padwid, image_width+2*padwid), dtype='complex64')
                fft = pyfftw.builders.fft2(frame, threads=num_threads, overwrite_input=in_place,
                                       avoid_copy=True)
                ifft = pyfftw.builders.ifft2(frame, threads=num_threads, overwrite_input=in_place,
                                         avoid_copy=True)
                template_freq = fft(np.pad((template),padwid,mode='constant')).conj() # we only need the conjugate
                abs_template_freq = abs(template_freq)
                eps = abs_template_freq.max() * 1e-15
                # Compute subpixel shifts per image
                y_shifts = np.empty(num_frames)
                x_shifts = np.empty(num_frames)
                starttime=time.time()
                for i in range(num_frames):
                    theimage=(
                    +np.max(data[:, :, max(0,int(i-0.025*fps)):min(num_frames,int(i+0.025*fps))],2)
                    *1)
                    image_freq=fft(ndimage.gaussian_filter(np.pad(theimage,padwid,mode='constant'),0.7))
                    cross_power = (image_freq * template_freq) / (abs(image_freq) * abs_template_freq + eps)
                    shifted_cross_power = np.fft.fftshift(abs(ifft(cross_power)))
                    shifts = np.unravel_index(np.argmax(shifted_cross_power), shifted_cross_power.shape)
                    shifts = utils._interpolate(shifted_cross_power, shifts, rad=3)
                    y_shifts[i] = shifts[0] - image_height // 2
                    x_shifts[i] = shifts[1] - image_width // 2
                    if i%10000==0:
                        print('now at ',i,time.time()-starttime)
                return y_shifts, x_shifts
            
        def postshift(tuple_,key):
            if type(tuple_)==dict:
                y_shifts=tuple_['y_shifts']
                x_shifts=tuple_['x_shifts']
            elif type(tuple_)==tuple:
                (y_shifts,x_shifts)=tuple_
            # Detect outliers
            max_y_shift, max_x_shift = 20 / (reso.ScanInfo() & key).microns_per_pixel
            y_shifts, x_shifts, outliers = galvo_corrections.fix_outliers(y_shifts, x_shifts,max_y_shift,max_x_shift)
            # Center shifts around zero
            y_shifts -= np.median(y_shifts)
            x_shifts -= np.median(x_shifts)
            # Create results tuple
            tuple_ = key.copy()
            tuple_['field'] = field_id + 1
            tuple_['motion_template'] = template
            tuple_['y_shifts'] = y_shifts
            tuple_['x_shifts'] = x_shifts
            tuple_['outlier_frames'] = outliers
            tuple_['y_std'] = np.std(y_shifts)
            tuple_['x_std'] = np.std(x_shifts)
            return tuple_
        def mcagain(wholescan,tuple_,fromframe,toframe,method):
            wholescan=galvo_corrections.correct_motion(wholescan,tuple_['x_shifts'], tuple_['y_shifts'])
            mini_scan=wholescan[:,:,fromframe:toframe] # mini spatially to mini spatially/temporaly
            mini_scan = mini_scan.astype(np.float32, copy=False)
            mini_scan = 2 * np.sqrt(mini_scan - mini_scan.min() + 3 / 8)
            template = np.mean(mini_scan, axis=-1)
            template = ndimage.gaussian_filter(template, 0.7)
            # define the method used in second motion correction here.
            y_shifts2, x_shifts2=yccompute_motion_shifts(wholescan,template,method=method, in_place=True, num_threads=8)
            tuple_['x_shifts']=tuple_['x_shifts']+x_shifts2
            tuple_['y_shifts']=tuple_['y_shifts']+y_shifts2
            return tuple_
        # 1st round of motion correction
        y_shifts, x_shifts=yccompute_motion_shifts(wholescan,template,method=1, in_place=True, num_threads=8)
        tuple_=postshift((y_shifts, x_shifts),key)
        tuple_=mcagain(wholescan,tuple_,fromframe,toframe,method=2)
        tuple_=postshift(tuple_,key)
        ####################################################
        # # if use second time motion correction ###################   
        # idea is, a typical motion correction way is to use recurrsive till the shifts are below a certain threshold ( mag, std, etc) from articles iv read, about 6-8 times will give a nearly perfect shifts.
        # so, im guess running it at least 2 times i would see an improvement. but not significant
        # plan is, either using a same, good, method to run multi times, since the method is good, it works better with a sharper template, gives better result
        # or, using a high frequncy noise prone method first, get rid of low freq motion to achieve a slighly better template, then a smaller window high temporal resolution method to further correct the rest high freq motion.
        # wholescan is the spatially smaller version of full scan, already rastercorrected
        # now, correct it with last result.
       

        # about outliers and shifts
        # ideally, the mean of shifts is 0, small amplitude, reasonable frequency
        # because vibration generally around the center, avg to 0
        # small vibration insteard of large. based on um. around 5 um?
        # should be less than 20 hz frequncy?
        # tested using a butter low pass filter to apply to shifts, not very good.
        # tested fft, 2nd threshold cut off, ifft, even worse.( because of ft edge artifact.)
        # maybe because the shift is not good, and filter wont rescue it.
        # things to test, find the frequence from ca scan shifts, (better filter), 
        # Insert
        self.insert1(tuple_)
        # Notify after all fields have been processed
        scan_key = {'animal_id': key['animal_id'], 'session': key['session'],
                    'scan_idx': key['scan_idx'], 'pipe_version': key['pipe_version']}
        if len(MotionCorrection - Scaninfo & scan_key) > 0:
            self.notify(scan_key, scan.num_frames, scan.num_fields)
            
    @notify.ignore_exceptions
    def notify(self, key, num_frames, num_fields):
        fps = (Scaninfo() & key).fetch1('fps')
        seconds = np.arange(num_frames) / fps
        fig, axes = plt.subplots(num_fields, 1, figsize=(15, 4 * num_fields), sharey=True)
        axes = [axes] if num_fields == 1 else axes # make list if single axis object
        for i in range(num_fields):
            y_shifts, x_shifts = (self & key & {'field': i + 1}).fetch1('y_shifts',
                                                                        'x_shifts')
            axes[i].set_title('Shifts for field {}'.format(i + 1))
            axes[i].plot(seconds, y_shifts, label='y shifts')
            axes[i].plot(seconds, x_shifts, label='x shifts')
            axes[i].set_ylabel('Pixels')
            axes[i].set_xlabel('Seconds')
            axes[i].legend()
        fig.tight_layout()
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename, bbox_inches='tight')
        plt.close(fig)
        msg = 'motion shifts for {animal_id}-{session}-{scan_idx}'.format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=img_filename, file_title=msg)
   
    def get_correct_motionf(self): # with lowpass filter 20hz
        """ Returns a function to perform motion correction on scans. """
        import scipy
        x_shifts, y_shifts = self.fetch1('x_shifts', 'y_shifts')
        b, a = scipy.signal.butter(4, 20, 'low',fs=400)# order, threshold
        # using fs=400 but should use actuall fps. for testing 396 fps i used 400.
        x_shifts=scipy.signal.lfilter(b,a,x_shifts)
        y_shifts=scipy.signal.lfilter(b,a,y_shifts)
        return lambda scan, indices=slice(None): galvo_corrections.correct_motion(scan,
                                                 x_shifts[indices], y_shifts[indices])
    def get_correct_motion(self):
        """ Returns a function to perform motion correction on scans. """
        x_shifts, y_shifts = self.fetch1('x_shifts', 'y_shifts')
        return lambda scan, indices=slice(None): galvo_corrections.correct_motion(scan,
                                                 x_shifts[indices], y_shifts[indices])

    def getvideo(self,fname,filter=False):
        correct_raster = (Rastercorrection()&self).get_correct_raster()
        if filter:
            correct_motion = self.get_correct_motionf()
        else:
            correct_motion = self.get_correct_motion()
        nchan=(Scaninfo()&self).fetch1('nchannels')
        fps=(Scaninfo()&self).fetch1('fps')
        dim=((Scaninfo()&self).fetch1('px_width'),(Scaninfo()&self).fetch1('px_height'))
        f=tifffile.TiffFile(scanreader.core.expand_wildcard((experiment.Scan() & self ).local_filenames_as_wildcard)[0])
        wholecan=f.asarray()
        wholecan=np.moveaxis(wholecan,0,2)
        wholecan=wholecan[:,:,::nchan]
        num_frames=(Scaninfo()&self).fetch1('nframes')
        savevideo(fps,dim,num_frames,correct_motion(correct_raster(wholecan)),
                  video_filename=fname)
    def savevideo(fps,dim,num_frames,clipdata,video_filename='original.avi'):
        fourcc=cv2.VideoWriter_fourcc('M',"J","P","G")
        out=cv2.VideoWriter(video_filename,fourcc,fps,dim)
        for i in range(num_frames):
            ardata=cv2.normalize(clipdata[:, :, i],None,255,0,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
            ardata3c=cv2.merge([ardata,ardata,ardata])
            out.write(ardata3c)
        out.release()

@patchtable
class Segmentation(dj.Computed):
    definition = """ 
        -> MotionCorrection
        -> segmentation task
        ---
        time:timestamp
        """
    class Mask(dj.Part):
        definition = """ 
        -> Segmentation
        mask_id         : smallint
        ---
        pixels          : longblob      # indices into the image in column major (Fortran) order
        weights         : longblob      # weights of the mask at the indices above
        """
        def get_mask_as_image(self):
            """ Return this mask as an image (2-d numpy array)."""
            # Get params
            pixels, weights = self.fetch('pixels', 'weights')
            image_height, image_width = (ScanInfo() & self).fetch1('px_height', 'px_width')

            # Reshape mask
            mask = Segmentation.reshape_masks(pixels, weights, image_height, image_width)

            return np.squeeze(mask)
    class Method1(dj.Part):
        definition = """ 
        ->Segmentation
        ---
        paramenter:int
        """

    def make(self, key):
        # Create masks
        if key['segmentation_method'] == 1:  # manual
            Segmentation.Manual().make(key)
        else:
            msg = 'Unrecognized segmentation method {}'.format(key['segmentation_method'])
            raise PipelineException(msg)
@patchtable
class Trace(dj.Manual):
    definition = """ 
        ->Segmentation
        ---
        trace:external
    """  
@patchtable
class Tracespikes(dj.Manual):
    definition = """ 
        ->Trace
        ---
        spikets:external
    """
# combining primary keys
@patchtable
class Ephys2p(dj.Manual):
    definition = """ 
        ->Recording
        ->Segmentation
        ---
        
    """ 
    @property
    def key_source(self):
        return Recording()# & {'pipe_version': CURRENT_VERSION}
        # only using the key from xxx() to find the key in other tb
    



    
# dj.ERD(patchtable)