r  = experiment.Session & 'session_date>"2016-02"' & experiment.AutoProcessing;

while true
    while count((preprocess.ExtractRaw().popRel & r) - preprocess.ExtractRaw)
    	  parpopulate(preprocess.ExtractRaw, r)
    end
    pause(1000)
end
