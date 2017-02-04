r  = experiment.Session & 'session_date>"2016-02"' & experiment.AutoProcessing;

while true
    while count((preprocess.Prepare().popRel & r) - preprocess.Prepare)
    	  parpopulate(preprocess.Prepare, r)
    end
    while count((preprocess.Sync().popRel & r) - preprocess.Sync)
    	  parpopulate(preprocess.Sync, r)
    end
    pause(1000)
end
