function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'sync', 'pipeline_sync');
end
obj = schemaObject;
end
