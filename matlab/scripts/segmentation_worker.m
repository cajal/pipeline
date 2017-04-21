r  = experiment.Session * experiment.AutoProcessing & 'session_date>"2016-02"';

while true
    while progress(preprocess.ExtractRaw, r)
    	  parpopulate(preprocess.ExtractRaw, r)
    end
    pause(1000)
end
