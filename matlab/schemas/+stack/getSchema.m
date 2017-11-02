function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'stack', 'pipeline_stack');
end
obj = schemaObject;
end
