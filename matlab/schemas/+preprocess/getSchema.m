function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'preprocess', 'pipeline_preprocess');
end
obj = schemaObject;
end
