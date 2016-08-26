%{
monet.MonetCleanRF (computed) # cleaned up and normalized receptive fields
-> MonetRFMap
-----
clean_map : longblob   # cleaned up receptive field map
%}

classdef CleanRF < dj.Relvar & dj.AutoPopulate

	properties
        popRel = tuning.MonetRF & tuning.MonetRFMap
	end

	methods(Access=protected)

		function makeTuples(self, key)
            % to be implemented
            error('Not yet in matlab!')
		end
	end

end