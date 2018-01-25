function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'tune', 'pipeline_tune');
end
obj = schemaObject;
end
