import warnings
import h5py
import numpy as np
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
iGains = [0.1, 0.2, 0.5, 1, 2, 5, 10]
vGains = [10, 20, 50, 100, 200, 500, 1000]
iLowPassCorners = [20, 50, 100, 200, 300, 500, 700, 1000, 1300, 2000, 3000, 5000, 8000, 10000, 13000, 20000]
vLowPassCorners = [20, 50, 100, 200, 300, 500, 700, 1000, 1300, 2000, 3000, 5000, 8000, 10000, 13000, 20000]
vHighPassCorners = [0, 0.1, 0.3, 0.5, 1, 3, 5, 10, 30, 50, 100, 300, 500, 800, 1000, 3000]

def readHDF5(hdf_path):
    data = {}
    with h5py.File(hdf_path, 'r+', driver='family', memb_size=0) as fid:

        data['ball'] = np.asarray(fid['ball'])
        wf = np.asarray(np.asarray(fid['waveform']))
        sets = np.asarray(np.asarray(fid['settings']))
        data['cam1ts'] = np.asarray(fid['behaviorvideotimestamp'])
        data['cam2ts'] = np.asarray(fid['eyetrackingvideotimestamp'])

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

            settings['vGain'] = vGains[np.unique(round(sets[:,1])).astype(int)]
            assert len(settings['vGain'])==1, 'Voltage gain changed during recording'

            settings['iLowPass'] = iLowPassCorners[np.unique(round(sets[:,2])).astype(int)+9]
            assert len(settings['iLowPass'])==1, 'Current low pass filter changed during recording'

            settings['vLowPass'] = vLowPassCorners[np.unique(round(sets[:,3])).astype(int)+9]
            assert len(settings['vLowPass'])==1,'Voltage low pass filter changed during recording'

            settings['vHighPass'] = vHighPassCorners[np.unique(round(sets[:,4])).astype(int)+9]
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