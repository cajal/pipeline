%{
# master table that gathers data about the scans of different types, prepares for trace extraction
-> experiment.Scan
---
%}


classdef Prepare < dj.Imported
    
    properties
        keySource = experiment.Scan - experiment.ScanIgnored & 'aim="2pScan"'
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            global STORE_IN_MEMORY
            self.insert(key)
            software = fetch1(experiment.Scan & key, 'software');
            switch software
                case 'scanimage'
                    reader = preprocess.getGalvoReader(key);
                    if STORE_IN_MEMORY
                        fprintf 'reading entire movie.. '
                        tic
                        movie = reader(:,:,:,:,:);
                        fprintf('%f4.1s\n', toc)
                    else
                        movie = reader;
                    end
                    makeTuples(preprocess.PrepareGalvo, key, reader, movie)
                    makeTuples(preprocess.PrepareGalvoMotion, key, reader, movie)
                    makeTuples(preprocess.PrepareGalvoAverageFrame, key, movie)
                case 'aod'
                    makeTuples(preprocess.PrepareAod,key)
                otherwise
                    error('"%s" is not implemented yet', software)
            end
        end
    end
    
end