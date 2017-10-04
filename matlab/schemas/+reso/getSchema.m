function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'reso', 'pipeline_reso');
end
obj = schemaObject;
end
