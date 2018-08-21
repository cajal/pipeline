function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'eye', 'pipeline_eye');
end
obj = schemaObject;
end
