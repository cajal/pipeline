function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'meso', 'pipeline_meso');
end
obj = schemaObject;
end
