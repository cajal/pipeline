function obj = getSchema
persistent schemaObject

if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'common', 'common');
end

obj = schemaObject;
end