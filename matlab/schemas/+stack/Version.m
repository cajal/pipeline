%{
# versions for the stack pipeline
-> shared.PipelineVersion
---
description                 : varchar(256)                  # any notes on this version
date=CURRENT_TIMESTAMP      : timestamp                     # automatic
%}


classdef Version < dj.Lookup
end