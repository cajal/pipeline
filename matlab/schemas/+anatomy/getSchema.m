function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'anatomy', 'pipeline_anatomy');
end
obj = schemaObject;
end
