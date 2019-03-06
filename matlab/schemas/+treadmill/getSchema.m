function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'treadmill', 'pipeline_treadmill');
end
obj = schemaObject;
end
