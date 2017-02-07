r  = experiment.Session * experiment.AutoProcessing & 'session_date>"2016-02"';

while true
    while progress(preprocess.Prepare(), r)
    	  parpopulate(preprocess.Prepare, r)
    end
    while progress(preprocess.Sync, r)
    	  parpopulate(preprocess.Sync, r)
    end
    pause(1000)
end
