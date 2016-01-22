%{
pre.Template (imported) # template for motion correction
-> pre.ScanCheck
-----
template :longblob   # template image
%}

classdef Template < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = pre.ScanCheck
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            reader = rf.getReader(key);
            nFrames = min(5000, reader.nFrames);
            if nFrames == reader.nFrames
                frames = 1:nFrames;
            else
                frames = randsample(1:reader.nFrames, nFrames)
                
                nFrames = min(5000, reader.nFrames);
                self.insert(key)
            end
        end
        
    end
end