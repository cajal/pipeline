function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'stk', 'pipeline_stacks');
end
obj = schemaObject;
end
