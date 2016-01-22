%{
common.TpPatch (manual) # markes two-photon scans in which 
-> common.TpScan

-----

%}

classdef TpPatch < dj.Relvar

	properties(Constant)
		table = dj.Table('common.TpPatch')
	end
end
