%{
vis2p.Experiments (manual) # Root table for two-photon experiments. Populated manually.
-> vis2p.Mice
exp_date        : date                   # experiment date
---
operator                    : enum('Dimitri','Manolis','Patrick','Cathryn','Shan') # m) experimenter's name
mass=null                   : decimal(4,1)                  # m) mouse mass in grams
anesthesia="none"           : enum('urethane','none','fentanyl','isoflurane') # m) 'urethane' or 'isoflurane' or 'none'
directory                   : varchar(512)                  # m) the current location of the 2-photon data
dyes="OGB,SR"               : enum('OGB,CR','OGB,SR,CR','OGB,SR','ALX','CR','SR','FL4,CR','FL4,SR','FL4','none','OGB') # m) the combination of dyes that where used
exp_notes                   : varchar(1023)                 # m) free form notes about the experiment
setup="2P1"                 : enum('2P2','2P1')             # m) the setup that the experiment was done
archive                     : varchar(120)                  # m) the location of the archived raw data, e.g. RAID backup disks
process="yes"               : enum('no','yes')              # m) sets the analysis status
%}


classdef Experiments < dj.Relvar
	methods

		function self = Experiments(varargin)
			self.restrict(varargin{:})
		end
	end

end