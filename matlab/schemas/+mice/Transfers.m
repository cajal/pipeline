%{
mice.Transfers (manual) # completed transfers

-> mice.Mice
dot                 : date                       # date of transfer
---
from_owner="none"   :enum('Jake','Manolis','Dimitri','Shan','Keith','Cathryn','Deumani','Matt','Megan','Paul','Shuang','Other','Available','none') # previous owner
to_owner="none"    :enum('Jake','Manolis','Dimitri','Shan','Keith','Cathryn','Deumani','Matt','Megan','Paul','Shuang','Other','Available','none')   # new owner 
from_facility="unknown"  : enum('TMF','Taub','Other','unknown')           # animal's previous facility
to_facility="unknown"  : enum('TMF','Taub','Other','unknown')           # animal's new facility
from_room="unknown"      : enum('VD4','T014','T057','T086D','Other','unknown') # animal's previous room 
to_room="unknown"      : enum('VD4','T014','T057','T086D','Other','unknown') # animal's new room 
from_rack           : tinyint                                        # animal's previous rack 
to_rack          : tinyint                                        # animal's new rack 
from_row=""              : char                                           # animal's previous row
to_row=""              : char                                           # animal's new row

transfer_notes=""    : varchar(4096)             # other comments 
transfer_ts=CURRENT_TIMESTAMP : timestamp        # automatic
%}



classdef Transfers < dj.Relvar

	properties(Constant)
		table = dj.Table('mice.Transfers')
	end

	methods
		function self = Requests(varargin)
			self.restrict(varargin)
        end
        function makeTuples(self,key)
            self.insert(key)
        end
	end
end