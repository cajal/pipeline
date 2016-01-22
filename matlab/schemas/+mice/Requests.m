%{
mice.Requests (manual) # requests for transgenic mice$
request_idx     : int                    # request number
---
requestor="none"            : enum('Jake','Manolis','Dimitri','Shan','Keith','Cathryn','Fabian','Deumani','Matt','Megan','Paul','Shuang','Other','Available','none') # person who requested the mice
dor=null                    : date                          # date of request
number_mice                 : int                           # number of mice requested
age=null                    : enum('any','P18-P21','P21-P28','4-6 Weeks') # age requested
line1=null                  : enum('','Nestin-Cre','Nestin-CreER(W)','Nestin-CreER(J)','Wfs1-CreER','ChAT-Cre','Viaat-Cre','SST-Cre','PV-Cre','VIP-Cre','GAD67-GFP','KOPRCS','Ai9','R-EYFP','R-ChR2-EYFP','R-ChR2-tdTomato','R-TVA/G','R-Arch','R-GCaMP3','Confetti','Nuc4','Nuc24','Cyt47','C57Bl/6','Fvb','Etv1-CreER','TH-Cre','Hist GFP','Ntsr1-Cre','CamKII-Cre','DBH Cre','Ai96_GCaMP6s','tetO-GCaMP6s','PronucRosaTw','mESC Twitch','Halo','Emx-1 Cre','Ai93_GCamp6','Ai94_GCamp6','Camk2a-tTA') # Mouse Line 1 Abbreviation
genotype1                   : enum('homozygous','heterozygous','hemizygous','positive','negative','wild type','') # genotype for line 1
line2=null                  : enum('','Nestin-Cre','Nestin-CreER(W)','Nestin-CreER(J)','Wfs1-CreER','ChAT-Cre','Viaat-Cre','SST-Cre','PV-Cre','VIP-Cre','GAD67-GFP','KOPRCS','Ai9','R-EYFP','R-ChR2-EYFP','R-ChR2-tdTomato','R-TVA/G','R-Arch','R-GCaMP3','Confetti','Nuc4','Nuc24','Cyt47','C57Bl/6','Fvb','Etv1-CreER','TH-Cre','Hist GFP','Ntsr1-Cre','CamKII-Cre','DBH Cre','Ai96_GCaMP6s','tetO-GCaMP6s','PronucRosaTw','mESC Twitch','Halo','Emx-1 Cre','Ai93_GCamp6','Ai94_GCamp6','Camk2a-tTA') # Mouse Line 2 Abbreviation
genotype2=null              : enum('homozygous','heterozygous','hemizygous','positive','negative','wild type','') # genotype for line 2
line3=null                  : enum('','Nestin-Cre','Nestin-CreER(W)','Nestin-CreER(J)','Wfs1-CreER','ChAT-Cre','Viaat-Cre','SST-Cre','PV-Cre','VIP-Cre','GAD67-GFP','KOPRCS','Ai9','R-EYFP','R-ChR2-EYFP','R-ChR2-tdTomato','R-TVA/G','R-Arch','R-GCaMP3','Confetti','Nuc4','Nuc24','Cyt47','C57Bl/6','Fvb','Etv1-CreER','TH-Cre','Hist GFP','Ntsr1-Cre','CamKII-Cre','DBH Cre','Ai96_GCaMP6s','tetO-GCaMP6s','PronucRosaTw','mESC Twitch','Halo','Emx-1 Cre','Ai93_GCamp6','Ai94_GCamp6','Camk2a-tTA') # Mouse Line 3 Abbreviation
genotype3=null              : enum('homozygous','heterozygous','hemizygous','positive','negative','wild type','') # genotype for line 3
request_notes=null          : varchar(4096)                 # other comments
request_ts=CURRENT_TIMESTAMP: timestamp                     # automatic
%}



classdef Requests < dj.Relvar

	properties(Constant)
		table = dj.Table('mice.Requests')
	end

	methods
		function self = Requests(varargin)
			self.restrict(varargin)
        end
        function makeTuples(self,key)
            self.insert(key)
        end
	end
end
