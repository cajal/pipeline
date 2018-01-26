function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'fuse', 'pipeline_fuse');
end
obj = schemaObject;
end
