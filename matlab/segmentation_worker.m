r  = rf.Session & 'session_date>"2016-02"';
while true
    parpopulate(pre.ExtractRaw, r)
    pause(1000)
end
