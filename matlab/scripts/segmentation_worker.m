r  = experiment.Session & 'session_date>"2016-02"';
while true
    parpopulate(preprocess.ExtractRaw, r)
    pause(1000)
end
