function [data, settings, ver] = readHD5(F)
% F: Key structure or full filename including path (for example '/path/Patchfile10.h5')
%
% data: all variables extracted from file
% settings: settings telegraphs from NPI amp
% ver: file version

if isstruct(F)
    if count(experiment.ScanBehaviorFile & F)
        F = fullfile(getLocalPath(fetch1(experiment.Session & F,'scan_path')), ...
            fetch1(experiment.Scan & F, 'filename'));
    end
end

f=dir(F);

% Check file existsgetLocalPath(fetch1(experiment.Session & F,'behavior_path'))
assert(length(f)==1,['Cannot find file ' F]);

% Use 'Date Modified' to determine file version.  so hacky.
if f.datenum > datenum('2014-06-03 08:00') %%&& f.datenum < datenum('')
    ver = 7;
else
    error 'File version not known'
end

% Append '%d.h5' to filename in place of trailing 'x.h5'
F = [F(1:end-4) '%d.h5'];


% *** Packet length is set at 2000 for analog channels ***
ANALOG_PACKET_LEN = 2000;

% open file using family driver
fapl = H5P.create('H5P_FILE_ACCESS');
H5P.set_fapl_family(fapl,2^31,'H5P_DEFAULT');
fp = H5F.open(F,'H5F_ACC_RDONLY',fapl);
H5P.close(fapl);

%data1d = H5Tools.readDataset(fp,'dataset1d')
data.ball = H5Tools.readDataset(fp,'ball') ;
wf = H5Tools.readDataset(fp,'waveform') ;
sets = H5Tools.readDataset(fp,'settings') ;
data.cam1ts = H5Tools.readDataset(fp,'behaviorvideotimestamp') ;
data.cam2ts = H5Tools.readDataset(fp,'eyetrackingvideotimestamp') ;

waveformDescStr=H5Tools.readAttribute(fp,'waveform Channels Description')';
assert(strcmp(deblank(waveformDescStr),'Current Input 1, Voltage Input 1, Sync Photodiode, Stimulation Photodiode, LED Level Input, Patch Command Input, Shutter, Current Input 2, Voltage Input 2, Scan Image Sync'),...
    'waveform Channels Description is wrong for this file version');

settingsDescStr=H5Tools.readAttribute(fp,'settings Channels Description')';
assert(strcmp(deblank(settingsDescStr),'Current Gain, Voltage Gain, Current Low Pass, Voltage Low Pass, Voltage High Pass'),...
    'settings Channels Description is wrong for this file version');

%convert waveform to structure
data.i1 = wf(:,1);
data.v1 = wf(:,2);
data.i2 = wf(:,8);
data.v2 = wf(:,9);
data.syncPd = wf(:,3);
data.stimPd = wf(:,4);
data.led = wf(:,5);
data.command = wf(:,6);
data.shutter = wf(:,7);
data.scanImage = wf(:,10);
data.ts = wf(:,11);
data.analogPacketLen = ANALOG_PACKET_LEN;

if ~any(~round(sets(:)))
    % deal with setting telegraphs on NPI amp
    iGains = [0.1 0.2 0.5 1 2 5 10];
    settings.iGain = iGains(unique(round(sets(:,1))));
    assert(length(settings.iGain)==1,'Current gain changed during recording');
    
    vGains = [10 20 50 100 200 500 1000];
    settings.vGain = vGains(unique(round(sets(:,2))));
    assert(length(settings.vGain)==1,'Voltage gain changed during recording');
    
    iLowPassCorners = [20 50 100 200 300 500 700 1000 1300 2000 3000 5000 8000 10000 13000 20000];
    settings.iLowPass = iLowPassCorners(unique(round(sets(:,3)))+9);
    assert(length(settings.iLowPass)==1,'Current low pass filter changed during recording');
    
    vLowPassCorners = [20 50 100 200 300 500 700 1000 1300 2000 3000 5000 8000 10000 13000 20000];
    settings.vLowPass = vLowPassCorners(unique(round(sets(:,4)))+9);
    assert(length(settings.vLowPass)==1,'Voltage low pass filter changed during recording');
    
    vHighPassCorners = [0 0.1 0.3 0.5 1 3 5 10 30 50 100 300 500 800 1000 3000];
    settings.vHighPass = vHighPassCorners(unique(round(sets(:,5)))+9);
    assert(length(settings.vHighPass)==1,'Voltage high pass filter changed during recording');
else
    % constant settings on unused NPI amp
    
    settings.iGain = 1;
    settings.vGain = 1;
    settings.iLowPass = 1;
    settings.vLowPass = 1;
    settings.vHighPass = 1;
    %warning('Unable to read settings telegraphs from NPI Amp');
end


% settings on AxoClamp 2B are constant
settings(2).iGain = 0.1;
settings(2).vGain = 10;
settings(2).vLowPass = 30000;
settings(2).iLowPass = 3000;
settings(2).vHighPass = 0;

% apply gains to voltage and current
data.v1 = data.v1/settings(1).vGain;
data.i1 = data.i1/settings(1).iGain;
data.v2 = data.v2/settings(2).vGain;
data.i2 = data.i2/settings(2).iGain;


H5F.close(fp);

