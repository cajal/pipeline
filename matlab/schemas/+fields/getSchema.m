function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'fields', 'pipeline_fields');
end
obj = schemaObject;
end
