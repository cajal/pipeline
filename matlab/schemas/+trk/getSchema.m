function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'trk', 'pipeline_pupiltracking');
end
obj = schemaObject;
end
