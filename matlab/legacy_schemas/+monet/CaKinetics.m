%{
monet.CaKinetics (lookup) # options for calcium response kinetics. 
ca_kinetics  : tinyint    # calcium option number
-----
transient_shape  : enum('exp','onAlpha')  # calcium transient shape
latency = 0      : float                  # (s) assumed neural response latency
tau = -1         : float                  # (s) time constant (used by some integration functions), -1=use optimal tau
explanation      : varchar(255)           # explanation of calcium response kinents
%}

classdef CaKinetics < dj.Relvar
    
    methods
        function fill(self)
            tuples = cell2struct({
                1   'exp'   0.05  1.5   'instantaneous rise, exponential delay'
                11  'onAlpha'  0.05  1.5 'alpha function response to on onset only'
                }', {'ca_kinetics', 'transient_shape', 'latency', 'tau', 'explanation'});
            self.inserti(tuples)            
        end
    end
end