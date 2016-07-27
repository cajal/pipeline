%{
aodpre.Set (imported) # Points in space and timeseries for each scan 
-> aodpre.Scan
-----
sampling_rate  : float  # sampling rate reported by the acquisition computer
%}

classdef Set < dj.Relvar & dj.AutoPopulate

	properties
		popRel  = aodpre.Scan
	end

	methods(Access=protected)

		function makeTuples(self, key)
            file = fetch1(aodpre.Scan & key, 'hdf5_file');
            reader = aodReader(getLocalPath(file), 'Functional');
            tuple = key;
            tuple.sampling_rate = reader.Fs;
			self.insert(tuple)
            
            coordinates = reader.coordinates;
            scanPoint = aodpre.ScanPoint;
            timeseries = aodpre.Timeseries;
            sz = reader.reshapedSize;
            for point_id = 1:size(coordinates, 1)
                % insert scan points
                tuple = key;
                tuple.point_id = point_id;
                tuple.x = coordinates(point_id, 1);
                tuple.y = coordinates(point_id, 2);
                tuple.z = coordinates(point_id, 3);
                scanPoint.insert(tuple)
                
                % insert time series
                tuple = key;
                for channel=1:sz(3)
                    tuple.channel = channel;
                    tuple.point_id = point_id;
                    tuple.trace = single(reader(:, point_id, channel));
                    timeseries.insert(tuple)
                end                
            end
		end
	end

end