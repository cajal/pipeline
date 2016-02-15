%{
trk.SVM (lookup) # table that stores the paths for the SVMs for each VideoGroup
-> trk.VideoGroup
version         : int                    # version of the SVM
---
svm_path                    : varchar(200)                  # path to the SVM file
%}


classdef SVM < dj.Relvar
end