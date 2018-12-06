function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    conn2 = dj.Connection('at-stim05.ad.bcm.edu', 'atlab', 'Lajac876');
    query(conn2, 'status');
    schemaObject = dj.Schema(conn2, 'pipeline_behavior', 'pipeline_behavior');
end
obj = schemaObject;
