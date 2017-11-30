%{
# versions for the reso pipeline
-> shared.PipelineVersion
---
description                 : varchar(256)                  # any notes on this version
date=CURRENT_TIMESTAMP      : timestamp                     # automatic
%}


classdef Version < dj.Manual
end