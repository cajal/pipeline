function obj = getSchema
persistent schemaObject

if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'mice', 'common_mice');
end

obj = schemaObject;
end