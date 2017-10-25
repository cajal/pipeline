%{
# versions for the stack pipeline
stack_version               : smallint                      # 
---
description                 : varchar(256)                  # any notes on this version
date=CURRENT_TIMESTAMP      : timestamp                     # automatic
%}


classdef Version < dj.Lookup
end