function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'vis', 'pipeline_vis');
end
obj = schemaObject;
end
