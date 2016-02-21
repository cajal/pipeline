import warnings
import h5py
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as iu_spline
from pipeline import PipelineException
import matplotlib.pyplot as plt
import seaborn as sns

ANALOG_PACKET_LEN = 2000

WAVEFORMDESCR = ['Current Input 1',
                 'Voltage Input 1',
                 'Sync Photodiode',
                 'Stimulation Photodiode',
                 'LED Level Input',
                 'Patch Command Input',
                 'Shutter',
                 'Current Input 2',
                 'Voltage Input 2',
                 'Scan Image Sync']
SETTINGSDESCR = ['Current Gain', 'Voltage Gain', 'Current Low Pass', 'Voltage Low Pass', 'Voltage High Pass']
iGains = np.asarray([0.1, 0.2, 0.5, 1, 2, 5, 10], dtype=float)
vGains =  np.asarray([10, 20, 50, 100, 200, 500, 1000], dtype=float)
iLowPassCorners =  np.asarray([20, 50, 100, 200, 300, 500, 700, 1000, 1300, 2000, 3000, 5000, 8000, 10000, 13000, 20000], dtype=float)
vLowPassCorners =  np.asarray([20, 50, 100, 200, 300, 500, 700, 1000, 1300, 2000, 3000, 5000, 8000, 10000, 13000, 20000], dtype=float)
vHighPassCorners =  np.asarray([0, 0.1, 0.3, 0.5, 1, 3, 5, 10, 30, 50, 100, 300, 500, 800, 1000, 3000], dtype=float)

def read_video_hdf5(hdf_path):
    data = {}
    with h5py.File(hdf_path, 'r+', driver='family', memb_size=0) as fid:

        data['ball'] = np.asarray(fid['ball'])
        wf = np.asarray(np.asarray(fid['waveform'])).T
        sets = np.asarray(np.asarray(fid['settings'])).T
        data['cam1ts'] = np.asarray(fid['behaviorvideotimestamp']).squeeze()
        data['cam2ts'] = np.asarray(fid['eyetrackingvideotimestamp']).squeeze()

        waveformDescStr = fid.attrs['waveform Channels Description'].decode('utf-8')
        settingsDescStr= fid.attrs['settings Channels Description'].decode('utf-8')


        assert [e.strip() for e in waveformDescStr.split(',')] == WAVEFORMDESCR,\
                    'waveform Channels Description is wrong for this file version'

        assert [e.strip() for e in settingsDescStr.split(',')] == SETTINGSDESCR,\
            'settings Channels Description is wrong for this file version'

        #
        # convert waveform to structure
        data['i1'], data['i2'] = wf[:,0], wf[:,7]
        data['v1'], data['v2'] = wf[:,1], wf[:,8]

        data['syncPd'] = wf[:,2]
        data['stimPd'] = wf[:,3]
        data['led'] = wf[:,4]
        data['command'] = wf[:,5]
        data['shutter'] = wf[:,6]
        data['scanImage'] = wf[:,9]
        data['ts'] = wf[:,10]
        data['analogPacketLen'] = ANALOG_PACKET_LEN

        settings = {}
        if np.any(np.round(sets)):
            # deal with setting telegraphs on NPI amp
            settings['iGain'] = iGains[np.unique(np.round(sets[:,0])).astype(int)]
            assert len(settings['iGain'])==1,'Current gain changed during recording'

            settings['vGain'] = vGains[np.unique(np.round(sets[:,1])).astype(int)]
            assert len(settings['vGain'])==1, 'Voltage gain changed during recording'

            settings['iLowPass'] = iLowPassCorners[np.unique(np.round(sets[:,2])).astype(int)+9]
            assert len(settings['iLowPass'])==1, 'Current low pass filter changed during recording'

            settings['vLowPass'] = vLowPassCorners[np.unique(np.round(sets[:,3])).astype(int)+9]
            assert len(settings['vLowPass'])==1,'Voltage low pass filter changed during recording'

            settings['vHighPass'] = vHighPassCorners[np.unique(np.round(sets[:,4])).astype(int)+9]
            assert len(settings['vHighPass'])==1,'Voltage high pass filter changed during recording'

        else:
            # constant settings on unused NPI amp
            settings['iGain'] = 1
            settings['vGain'] = 1
            settings['iLowPass'] = 1
            settings['vLowPass'] = 1
            settings['vHighPass'] = 1
            warnings.warn('Unable to read settings telegraphs from NPI Amp')



        # settings on AxoClamp 2B are constant
        settings2 = {}
        settings2['iGain'] = 0.1
        settings2['vGain'] = 10
        settings2['vLowPass'] = 30000
        settings2['iLowPass'] = 3000
        settings2['vHighPass'] = 0

        settings = [settings, settings2]
        # apply gains to voltage and current
        data['v1'] = data['v1']/settings[0]['vGain']
        data['i1'] = data['i1']/settings[0]['iGain']
        data['v2'] = data['v2']/settings[1]['vGain']
        data['i2'] = data['i2']/settings[1]['iGain']
    return data


def ts2sec(ts, packetLen=0):
    """
    Convert 10MHz timestamps from Saumil's patching program (ts) to seconds (s)

    :param ts: timestamps
    :param packetLen: length of timestamped packets
    :returns:
        timestamps converted to seconds
        system time (in seconds) of t=0
        bad camera indices from 2^31:2^32 in camera timestamps prior to 4/10/13
    """
    ts = ts.astype(float)

    # find bad indices in camera timestamps and replace with linear est
    bad_idx = ts ==2**31-1
    if bad_idx.sum() > 10:
        raise PipelineException('Bad camera ts...')
        x = np.where(~bad_idx)[0]
        x_bad = np.where(bad_idx)[0]
        f = iu_spline(x, ts[~bad_idx], k=1)
        ts[bad_idx] = f(x_bad)

    #  remove wraparound
    wrapInd = np.where(np.diff(ts)<0)[0]
    while not len(wrapInd) == 0:
        ts[wrapInd[0]+1:] += 2**32
        wrapInd = np.where(np.diff(ts)<0)[0]

    s = ts/1e7

    # Remove offset, and if not monotonically increasing (i.e. for packeted ts), interpolate
    if np.any(np.diff(s)<=0):
        # Check to make sure it's packets
        diffs = np.where(np.diff(s)>0)[0]
        assert packetLen == diffs[0]+1

        # Subtract duration of first packet from all timestamps
        packetDur = s[packetLen-1]-s[0]
        s -= packetDur

        # Interpolate
        not_zero = np.hstack((0, diffs+1))
        f = iu_spline(not_zero, s[not_zero], k=1)
        s = f(np.arange(len(s)))
    start = s[0]
    s -= start

    return s, start, bad_idx


class ROIGrabber:
    def __init__(self, img):
        self.img = img
        self.start = None
        self.current = None
        self.end = None
        self.pressed = False
        self.roi = None
        sns.set_style('white')
        self.fig, self.ax = plt.subplots(facecolor='w')

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.replot()
        plt.show(block=True)

    def draw_rect(self, fr, to, color='dodgerblue'):
        x = np.vstack((fr, to))
        fr = x.min(axis=0)
        to = x.max(axis=0)
        self.ax.plot(fr[0]*np.ones(2), [fr[1], to[1]], color=color, lw=2)
        self.ax.plot(to[0]*np.ones(2), [fr[1], to[1]], color=color, lw=2)
        self.ax.plot([fr[0], to[0]], fr[1]*np.ones(2), color=color, lw=2)
        self.ax.plot([fr[0], to[0]], to[1]*np.ones(2), color=color, lw=2)
        self.ax.plot(fr[0], fr[1],'ok',mfc='gold')
        self.ax.plot(to[0], to[1],'ok',mfc='deeppink')

    def replot(self):
        self.ax.clear()
        self.ax.imshow(self.img, cmap=plt.cm.gray)

        if self.pressed:
            self.draw_rect(self.start, self.current, color='lime')
        elif self.start is not None and self.end is not None:
            self.draw_rect(self.start, self.current)
        self.ax.axis('tight')
        self.ax.set_aspect(1)
        self.ax.set_title('Press q when done', fontsize=16, fontweight='bold')
        plt.draw()


    def on_key(self, event):
        if event.key == 'q':
            plt.close(self.fig)
        self.replot()

    def on_press(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.pressed = True
            self.start = np.asarray([event.xdata, event.ydata])

    def on_release(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.end = np.asarray([event.xdata, event.ydata])
        else:
            self.end = self.current
            x = np.vstack((self.start, self.end))
            self.roi = np.hstack((x.min(axis=0),x.max(axis=0)))


        self.pressed = False
        self.replot()

    def on_move(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.current = np.asarray([event.xdata, event.ydata])
            if self.pressed:
                print(self.current)
                self.replot()

