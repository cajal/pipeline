function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'trippy', 'pipeline_trippy');
end
obj = schemaObject;
end
