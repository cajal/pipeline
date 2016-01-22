function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'pre', 'dimitri_pre');
end
obj = schemaObject;
end
