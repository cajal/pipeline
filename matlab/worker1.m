r  = rf.Session & 'session_date>"2016-02"';
while true
    parpopulate(pre.ScanInfo, r)
    parpopulate(rf.Sync, r)
    parpopulate(pre.ScanCheck, r)
    parpopulate(pre.AlignRaster, r)
    parpopulate(pre.AlignMotion)
    parpopulate(pre.AverageFrame)
    parpopulate(pre.Segment, 'segment_method=1')
    parpopulate(pre.ExtractTraces)
    pause(1000)
end
