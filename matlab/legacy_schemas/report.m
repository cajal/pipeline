function report

fprintf '\nReso pipeline:\n'
r = rf.Session & 'session_date>"2016-02"';
%r = 'animal_id=8623';%r = 'animal_id=8628';
fprintf('Mice %s\n', sprintf('%d ', fetchn(common.Animal & r, 'animal_id')))

progress(pre.ScanInfo, r)
progress(pre.ScanCheck, r)
progress(pre.AlignRaster, r)
progress(pre.AlignMotion, r)
progress(pre.AverageFrame, r)
progress(pre.ManualSegment, r)
progress(pre.Segment, r)
progress(pre.ExtractTraces, r)
progress(pre.ExtractSpikes, r)
progress(rf.Sync, r)
progress(monet.DriftTrialSet, r)
progress(monet.DriftResponseSet, r)
progress(monet.VonMises, r)
progress(monet.RF, r)
progress(monet.CleanRF, r)
progress(monet.Fit, r)
progress(monet.OriDesign, r)
progress(monet.OriMap, r)

% fprintf '\nAOD pipeline:\n'
% r = fetch(vis2p.Experiments &  'exp_date>"2016-02"', '(mouse_id)->animal_id');
% fprintf('Mice %s\n', sprintf('%d ', fetchn(common.Animal & r, 'animal_id')))
% 
% progress(aodpre.Scan)
% progress(aodpre.Sync)
% progress(aodpre.Set)
% progress(aodpre.ComputeTraces)
% progress(aodpre.ExtractSpikes)
% progress(aod_monet.DriftTrialSet, r)
% progress(aod_monet.DriftResponseSet, r)
% progress(aod_monet.VonMises, r)
% progress(aod_monet.RF, r)
