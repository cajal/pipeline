function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    psy.getSchema;
    common.getSchema;
    schemaObject = dj.Schema(dj.conn, 'rf', 'pipeline_rf');
end
obj = schemaObject;
end
