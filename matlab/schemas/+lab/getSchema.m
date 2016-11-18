function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'lab', 'common_lab');
end
obj = schemaObject;
end
