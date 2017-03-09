function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'stimulus', 'pipeline_stimulus');
end
obj = schemaObject;
end
