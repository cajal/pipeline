%{
common.Injection (manual) # record of injections

-> common.Animal
injection       : tinyint               # number of injection (first, second, etc...)
---
injection_guidance="2P"     : enum('2P','stereotactic','other')# guidance method
injection_type="virus"      : enum('virus','dye','beads','other')# substance injected
virus_id=null               : int                           # id number of injected virus
injection_date=null         : date                          # date of injection
injection_site              : enum('V1','V2','LM','AL','PM','Visual Cortex','Nucleus Basalis','LGN','Retina','Unknown')# site of injection
injection_size="unknown"    : enum('very large','large','medium','small','very small','unknown')# qualitative size
injection_has_stack=0       : tinyint                       # true if there is an imaging stack of the injection site
injection_note=null         : varchar(4096)                 # 
injection_ts=CURRENT_TIMESTAMP: timestamp                   # automatic
%}

classdef Injection < dj.Relvar
    
    properties(Constant)
        table = dj.Table('common.Injection')
    end
end

        