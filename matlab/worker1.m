r  = rf.Session & 'session_date>"2016-02"';
while true
    parpopulate(pre.ScanInfo, r)
    parpopulate(rf.Sync)
    parpopulate(pre.ScanCheck, r)
    parpopulate(pre.AlignRaster, r)
    parpopulate(pre.AlignMotion, r)
    parpopulate(pre.AverageFrame, r)
    parpopulate(pre.Segment)
    parpopulate(pre.ExtractTraces)
    pause(1000)
end
