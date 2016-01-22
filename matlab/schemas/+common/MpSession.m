%{
common.MpSession (manual) # multipatch session info
-> common.MpSlice
mp_sess                             : smallint      # mutipatch recording session number for this slice
-----
mp_sess_purpose='stimulation'       : enum('stimulation','firingpattern','spontaneous','other') # purpose
mp_sess_path =""                    : varchar(4095) #
mp_sess_notes =""                   : varchar(4095) # 
mp_session_ts = CURRENT_TIMESTAMP   : timestamp  # automatic but editable  
%}

classdef MpSession < dj.Relvar

	properties(Constant)
		table = dj.Table('common.MpSession')
	end

	methods
		function self = MpSession(varargin)
			self.restrict(varargin)
		end
	end
end
