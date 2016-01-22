%{
mice.Mice (manual) # Basic mouse info

animal_id           : int                                            # id number
---
other_id=""         : varchar(20)                                    # alternative id number
dob=null            : date                                           # animal's date of birth
dow=null            : date                                           # animal's date of weaning
sex="unknown"       : enum('M','F','unknown')                        # animal's sex
color="unknown"     : enum('Black','Brown','White','unknown')        # animal's color
ear_punch="unknown" : enum('None','R','L','RL','RR','LL','unknown')  # animal's ear punch
owner="none"        : enum('Jake','Manolis','Xiaolong','Dimitri','Shan','Keith','Cathryn','Deumani','Matt','Megan','Paul','Shuang','Other','Available','none') # mouse's owner
facility="unknown"  : enum('TMF','Taub','Other','unknown')           # animal's curent facility 
room="unknown"      : enum('VD4','T014','T057','T086D','Other','unknown') # animal's current room 
rack=null           : tinyint                                        # animal's curent rack 
row=""              : char                                           # animal's curent row

mouse_notes=""      : varchar(4096)                                  # other comments and distinguishing features
mouse_ts=CURRENT_TIMESTAMP : timestamp                               # automatic
%}



classdef Mice < dj.Relvar

	properties(Constant)
		table = dj.Table('mice.Mice')
	end

	methods
		function self = Mice(varargin)
			self.restrict(varargin)
        end
        function makeTuples(self,key)
            self.insert(key)
        end
	end
end
