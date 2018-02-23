%{
# declare scan fields to register to a stack as well as channels and method used
 (stack_session) -> stack.CorrectedStack
 (scan_session) -> `pipeline_experiment`.`scan`
-> `pipeline_shared`.`#field`
 (stack_channel) -> `pipeline_shared`.`#channel`
 (scan_channel) -> `pipeline_shared`.`#channel`
-> `pipeline_shared`.`#registration_method`
%}


classdef RegistrationTask < dj.Manual
end