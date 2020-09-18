function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'vreso', 'pipeline_vreso');
end
obj = schemaObject;
end
