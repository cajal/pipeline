function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'aod_monet', 'pipeline_aod_monet');
end
obj = schemaObject;
end
