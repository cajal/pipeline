function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'aodpre', 'pipeline_aod_preprocessing');
end
obj = schemaObject;
end
