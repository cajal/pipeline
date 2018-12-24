%{
    twop_setup             : varchar(256)   # two photon setup name, e.g. 2P1
    setup                  : varchar(256)   # Setup name of behavior/stim computer, e.g. at-stim01
---
    ip                     : varchar(16)    # setup IP address, e.g. at-stim01.ad.bcm.edu
    state="ready"          : enum('systemReady','sessionRunning','stimRunning','stimPaused')  #
    state_control="ready"  : enum('startSession','startStim','stopStim','stopSession','pauseStim','resumeStim','Initialize','')  #
    animal_id=null         : int # animal id
    session=null           : int # session number
    scan_idx=null          : int             # 
    stimulus=""            : varchar   #
    next_trial=null        : int  #
    last_ping=null         : timestamp
    task_idx=7             : int             # task identification number
    task="train"           : enum('train','calibrate')
    trial_done=1           : int # 0=trial running, 1=trial finished
    exp_done=1		   : int # 0=exp running, 1=all trials finished
%}
classdef SetupControl < dj.Manual
end
