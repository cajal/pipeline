function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'pre', 'pipeline_preprocessing');
end
obj = schemaObject;
end
