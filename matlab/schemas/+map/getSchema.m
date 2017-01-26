function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'map', 'manolis_map');
end
obj = schemaObject;
end
