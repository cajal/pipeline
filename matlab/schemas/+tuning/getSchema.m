function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'fields', 'pipeline_tuning');
end
obj = schemaObject;
end
