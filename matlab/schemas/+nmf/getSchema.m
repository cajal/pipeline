function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'nmf', 'microns_nmf');
end
obj = schemaObject;
end
