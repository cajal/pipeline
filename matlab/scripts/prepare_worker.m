r  = experiment.Session & 'session_date>"2016-02"';
while true
    parpopulate(preprocess.Prepare, r)
    pause(1000)
end
