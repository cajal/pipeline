%{
# speaker sound pressure calibration
speaker_id              : int               # some number that identifies the speaker
rig                     : char              # setup name
location                : tinyint           # 1 = Left, 2 = Right
trial                   : int               # trial number of calibration
---
sampling_freq           : float             # sampling frequency used during calibration
test_voltage            : float             # test voltage in volts used to drive the audio input which via amp drives the speaker
desired_spl             : float             # desired sound pressure after equalization in dB
frequencies             : mediumblob        # frequencies presented during calibration
response                : mediumblob        # compensatory response to equalize all presented frequencies to a fixed output level when speaker isdriven by 1V
fir_coefficients        : mediumblob        # coefficients of equalizing fir filter
ts                      : timestamp         # timestamp
%}

classdef SpeakerCalibration < dj.Manual
end

