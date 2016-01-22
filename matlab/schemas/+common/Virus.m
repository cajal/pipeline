%{
common.Virus (manual)       # table of viruses

virus_id                    : int                                               # unique id for each produced or purchased virus
-----
virus_name                  : varchar(255)                                      # full virus name
virus_gene                  : enum('GCaMP6s','GCaMP6m','GCaMP6f','ChR2-mCherry','ChR2-YFP','Cre','GCaMP3','GCaMP3-tdTomato','ArchT-GFP','ArchT-tdTomato','mCherry-WGA-Cre') # gene name
virus_opsin                 : enum('None','ChR2(H134R)','ChR2(E123T/T159C)','ArchT1.0') # opsin version
virus_type                  : enum('AAV','Rabies','Lenti')                      # type of virus
serotype = "N/A"            : enum('AAV2/1','AAV2','AAV2/5','AAV2/8','N/A')     # AAV serotype
virus_promoter = "N/A"      : enum('CamKIIa','hSyn','EF1a','CAG','CMV','N/A')   # viral promoter
virus_isfloxed              : boolean                                           # true if expression is dependent on Cre
virus_source                : enum('Penn','UNC','Homegrown')                    # source of virus
virus_lot                   : varchar(32)                                       # lot #
virus_titer                 : float                                             # titer
virus_notes=""              : varchar(4095)                                     # free-text notes
virus_ts=CURRENT_TIMESTAMP  : timestamp                                         # automatic
%}

classdef Virus < dj.Relvar

	properties(Constant)
		table = dj.Table('common.Virus')
	end

end
