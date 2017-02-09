r  = experiment.Session * experiment.AutoProcessing & 'session_date>"2016-02"';

while true
    while progress(preprocess.Prepare(), r)
    	  parpopulate(preprocess.Prepare, r)
    end
    while progress(preprocess.Sync, r)
    	  parpopulate(preprocess.Sync, r)
    end
    while progress(preprocess.Treadmill, r)
    	  parpopulate(preprocess.Treadmill, r)
    end
    while progress(preprocess.BehaviorSync, r)
    	  parpopulate(preprocess.BehaviorSync, r)
    end
    while progress(preprocess.Spikes, r)
    	  parpopulate(preprocess.Spikes, r)
    end
    pause(1000)
end
