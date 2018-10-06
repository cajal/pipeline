function data = readAudioHD5(F,varargin)
% F: Key structure or full filename including path (for example '/path/Patchfile10.h5')
%
% data: audio data extracted from file
% ver: 1.0

if isstruct(F)
    if count(experiment.ScanBehaviorFile & F)
        F = fullfile(getLocalPath(fetch1(experiment.Session & F,'behavior_path')), ...
            fetch1(experiment.ScanBehaviorFile & F, 'filename'));
    else
        filename = fetch1(experiment.Scan & F,'filename');
        filename = strsplit(filename, '_');
        scanId = str2double(filename{2});
        filename = filename{1};
        F = fullfile( ...
            getLocalPath(fetch1(experiment.Session & F,'behavior_path')), ...
            sprintf('%s%d0.h5', filename, scanId));
    end
end

% Check that file exists
F = strrep(F,'%d','0'); % FIX trailing %d.h5 not found error
f = dir(F);
assert(length(f)==1,['Cannot find file ' F]);
F = [F(1:end-4) '%d.h5']; % Append '%d.h5' to filename in place of trailing 'x.h5'

try
    % open file using family driver
    fapl = H5P.create('H5P_FILE_ACCESS');
    H5P.set_fapl_family(fapl,2^31,'H5P_DEFAULT');
    fp = H5F.open(F,'H5F_ACC_RDONLY',fapl);
    H5P.close(fapl);
catch
    error 'File version not known'
end

try
    audioVersion = H5Tools.readAttribute(fp,'Audio_Version');
    switch audioVersion
        case 1.0
            down_sample_factor = 1 ;
            if (nargin > 1)
                down_sample_factor = varargin{1} ;
            end
            data.audio_fs = H5Tools.readAttribute(fp,'Audio_Fs')/down_sample_factor;
            data.audio_blocksize = H5Tools.readAttribute(fp,'Audio_samples_per_channel');
            total_samples_needed = max(H5Tools.getDatasetDim(fp, 'Audio Signals')) ;
            done = false ;
            src_ptr = 1 ;
            dest_ptr = 1 ;
            samples_needed = data.audio_blocksize ;
            sample_remaining = total_samples_needed ;
            data.mic_data = single(zeros(ceil(total_samples_needed/down_sample_factor),1)) ;
            data.mic_ts = data.mic_data ;
            while ~done
                wf = H5Tools.readDataset(fp,'Audio Signals', 'range', [1 src_ptr], [2 src_ptr+samples_needed-1]) ;
                try
                	tmic = single(decimate(wf(:,1),down_sample_factor,'FIR')) ;
                catch
                    tmic = 0 ;
                end
                data.mic_data(dest_ptr:dest_ptr+length(tmic)-1) = tmic ;
                data.mic_ts(dest_ptr:dest_ptr+length(tmic)-1) = single(downsample(wf(:,2),down_sample_factor)) ;
                samples_remaining = sample_remaining - samples_needed ;
                src_ptr = src_ptr+samples_needed ;
                dest_ptr = dest_ptr+length(tmic) ;
                if (samples_remaining > 0)
                    if (samples_remaining >= data.audio_blocksize)
                        samples_needed = data.audio_blocksize ;
                    else
                        samples_needed = samples_remaining ;
                    end
                else
                    done = true ;
                end 
            end
    end
catch
    fprintf('Error reading audio data\n') ;
end
        
% close file
H5F.close(fp);
