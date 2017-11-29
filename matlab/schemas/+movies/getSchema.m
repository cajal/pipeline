function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'movies', 'pipeline_movies');
end
obj = schemaObject;
end
