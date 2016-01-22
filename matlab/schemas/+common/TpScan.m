%{
common.TpScan (manual) # scanimage scan info
->common.TpSession
scan_idx : smallint # scanimage-generated sequential number
-----
surfz               : float   # (um) z-coord at pial surface
depth=0             : int     # manual depth measurement 
laser_wavelength    : float # (nm)
laser_power         : float # (mW) to brain
cortical_area="V1"  : enum('other','unknown','V1','LM','AL','PM') # Location of scan
scan_notes = ""     : varchar(4095)  #  free-notes
scan_ts = CURRENT_TIMESTAMP : timestamp   # don't edit
%}

classdef TpScan < dj.Relvar

	properties(Constant)
		table = dj.Table('common.TpScan')
	end

	methods
		function self = TpScan(varargin)
			self.restrict(varargin)
        end
        
        function filenames = getFilename(self, increment, overrideBasename)
            if nargin<=1
                increment=0;
            end
            keys = fetch(self);
            n = length(keys);
            filenames = cell(n,1);
            for i = 1:n
                key = keys(i);
                assert(length(key)==1, 'one scan at a time please')
                [path, basename] = fetch1(common.TpSession(key), ...
                    'data_path', 'basename');
                if nargin>=3
                    basename = overrideBasename;
                end
                filenames{i} = sprintf('%s%03u', ...
                    getLocalPath(fullfile(path, basename)), key.scan_idx+increment);
            end
        end
	end
end