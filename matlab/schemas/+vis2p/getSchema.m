function obj = getSchema
persistent schemaObject

if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'vis2p', 'vis2p_manolis');
end

obj = schemaObject;
end