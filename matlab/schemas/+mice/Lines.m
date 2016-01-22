%{
mice.Lines (manual) # Basic mouse line info$
line            : enum('Nestin-Cre','Nestin-CreER(W)','Nestin-CreER(J)','Wfs1-CreER','ChAT-Cre','Viaat-Cre','SST-Cre','PV-Cre','VIP-Cre','GAD67-GFP','KOPRCS','Ai9','R-EYFP','R-ChR2-EYFP','R-ChR2-tdTomato','R-TVA/G','R-Arch','R-GCaMP3','Confetti','Nuc4','Nuc24','Cyt47','C57Bl/6','Fvb','Etv1-CreER','TH-Cre','Hist GFP','Ntsr1-Cre','CamKII-Cre','DBH Cre','Ai96_GCaMP6s','tetO-GCaMP6s','PronucRosaTw','mESC Twitch','Halo','Emx-1 Cre','Ai93_GCamp6','Ai94_GCamp6','Camk2a-tTA')  # Mouse Line Abbreviation
---
line_full                   : varchar(100)                  # full line name
rec_strain                  : varchar(20)                   # recipient strain
donor_strain                : varchar(20)                   # donor strain
n=null                      : tinyint                       # minimumm number of backcrosses to recipient strain
seq                         : varchar(5000)                 # sequence of transgene, if available
line_notes                  : varchar(4096)                 # other comments
line_ts=CURRENT_TIMESTAMP   : timestamp                     # automatic
%}



classdef Lines < dj.Relvar

	properties(Constant)
		table = dj.Table('mice.Lines')
	end

	methods
		function self = Lines(varargin)
			self.restrict(varargin)
        end
        function makeTuples(self,key)
            self.insert(key)
        end
	end
end
