function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    conn2 = dj.Connection('127.0.0.1', getenv('DJ_USER'), getenv('DJ_PASS'));
    query(conn2, 'status');
    schemaObject = dj.Schema(conn2, 'pipeline_behavior', 'pipeline_behavior');
end
obj = schemaObject;
