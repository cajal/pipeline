function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'shared', 'pipeline_shared');
end
obj = schemaObject;
end
