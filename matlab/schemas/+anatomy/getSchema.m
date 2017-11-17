function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'map', 'pipeline_anatomy');
end
obj = schemaObject;
end
