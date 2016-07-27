%{
vis2p.Scans (manual) # Scans$
-> vis2p.Experiments
scan_idx        : smallint unsigned     # scan number within this experiment, forms the _%03u suffix of the scan files
---
file_name                   : varchar(61)                   # m) file prefix if different from the default YYMMDD
scan_prog="MPScan"          : enum('AOD','Unirec','Imager','ScanImage','MPScan')# m) Scanning program
aim="orientation tuning"    : enum('AF Imaging','orientation tuning','visual response','injection','patching','spontaneous','receptive field mapping','stack','multi-tuning','old rf maping','Intrinsic Imaging','vesselMap','shock','other')# m) the aim of this scan
stim_engine="State"         : enum('State','VisStim','other','None')# m) Visual stimulus generating software
lens=null                   : tinyint unsigned              # m) lens magnification
mag=null                    : decimal(4,2) unsigned         # m) scan magnification
laser_wavelength=null       : smallint unsigned             # m) (nm)
attenuator_setting=null     : decimal(4,1)                  # m) before ~June 2010 this was percentage in MPScan, now this is the angle in the APTUser GUI
x=null                      : int                           # m) (um) objective manipulator x position
y=null                      : int                           # m) (um) objective manipulator y position
z=null                      : int                           # m) (um) objective manipulator z position
surfz=null                  : int                           # m) (um) objective manipulator z position at pial surface
scan_rotation=null          : decimal(4,1)                  # m) (degrees clockwise) the rotation of the scanning direction
notes                       : varchar(256)                  # m) optional free-form comments
cell_patch=null             : tinyint unsigned              #
state="anesthetized"        : enum('anesthetized','awake')  #
problem_type="none!"        : enum('missing or poor photodiode signal','frames mismatch timestamps','invalid stimulus','missing or poor ephys photodiode','corrupt 2p file','none!','water left','non responsive','do not analyze','missing 2p file')# possible problems
area=null                   : enum('?','V1/V2','V2','V1')   #
%}


classdef Scans < dj.Relvar
    methods
        filename = getFilename( obj )
            
        function self = Scans(varargin)
            self.restrict(varargin{:})
        end
        
        tpr = tpReader( obj )
        data = getData(obj,channel);
       
    end
end
