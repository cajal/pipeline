function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'quality', 'pipeline_quality');
end
obj = schemaObject;
end
