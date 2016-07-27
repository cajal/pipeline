function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'experiment', 'pipeline_experiment');
end
obj = schemaObject;
end
