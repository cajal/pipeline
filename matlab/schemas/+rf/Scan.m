%{
rf.Scan (manual) # scanimage scan info
-> rf.Session
scan_idx        : smallint               # number of TIFF stack file
---
-> rf.Site
file_num                    : smallint                      # number of HD5 file
depth=0                     : int                           # manual depth measurement
laser_wavelength            : float                         # (nm)
laser_power                 : float                         # (mW) to brain
cortical_area="unknown"     : enum('other','unknown','V1','LM','AL','PM') # Location of scan
scan_notes                  : varchar(4095)                 # free-notes
scan_ts=CURRENT_TIMESTAMP   : timestamp                     # don't edit
%}

classdef Scan < dj.Relvar
    
    methods
        function filenames = getFilename(self, increment, basename)
            if nargin<=1
                increment=0;
            end
            keys = fetch(self);
            n = length(keys);
            filenames = cell(n,1);
            for i = 1:n
                key = keys(i);
                assert(length(key)==1, 'one scan at a time please')
                path = fetch1(rf.Session(key), ...
                    'scan_path');
                filenames{i} = sprintf('%s%03u', ...
                    getLocalPath(fullfile(path, basename)), key.filenum+increment);
            end
        end
    end
end