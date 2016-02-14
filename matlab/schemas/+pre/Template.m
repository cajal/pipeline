%{
pre.Template (imported) # template for motion correction
-> pre.ScanInfo
-> pre.Slice
channel : tinyint   # channel number
-----
template :longblob   # template image
%}

classdef Template < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel  = pre.ScanInfo
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            reader = pre.getReader(key,'~/cache');
            for slice = 1:reader.nslices
                key.slice = slice;
                for channel = 1:reader.nchannels
                    key.channel = channel;
                    frames = reader(:,:,channel,slice,round(linspace(1,reader.nframes,5000)));                    
                    template = mean(frames,3);                    
                    key.template = template;
                    self.insert(key);
                end
            end
        end
        
    end
end