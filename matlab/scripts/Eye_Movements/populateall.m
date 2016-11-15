% populate tables related to eye position clustering
cd /mnt/lab/home/atlab/pipeline/pipeline/matlab/scripts/Eye_Movements
startup
parpopulate(preprocess.CirclesLookup) ;
parpopulate(preprocess.CircleMap) ;
parpopulate(preprocess.EyePosClusterCenter) ;
%parpopulate(preprocess.ClusteredEyePos) ;