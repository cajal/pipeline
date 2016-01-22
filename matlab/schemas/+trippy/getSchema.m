function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'trippy', 'dimitri_trippy');
end
obj = schemaObject;
end
