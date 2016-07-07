%{
preprocess.Prepare (imported) # master table that gathers data about the scans of different types, prepares for trace extraction
-> experiment.Scan
---
%}


classdef Prepare < dj.Relvar & dj.AutoPopulate

	properties
		popRel = experiment.Scan  
	end

	methods(Access=protected)

		function makeTuples(self, key)
			self.insert(key)
            software = fetch1(experiment.Scan & key, 'software');
            switch software
                case 'scanimage'
                    reader = preprocess.getGalvoReader(key);
                    makeTuples(preprocess.PrepareGalvo, key, reader)
                    makeTuples(preprocess.PrepareGalvoMotion, key, reader)
                    makeTuples(preprocess.PrepareGalvoAverageFrame, key, reader)
                otherwise
                    error('"%s" is not implemented yet', software)
            end
		end
	end

end