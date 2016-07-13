% startup.m
setenv('DJ_HOST', 'at-database.ad.bcm.edu');
setenv('DJ_USER','atlab');
setenv('DJ_PASS','432statis');
addpath /home/atlab/pipeline
addpath /home/atlab/pipeline/datajoint-matlab
addpath(genpath('/home/atlab/pipeline/commons/schemas'));
addpath('/home/atlab/pipeline/commons/lib');
addpath('/home/atlab/pipeline/pipeline/matlab/');
addpath('/home/atlab/pipeline/pipeline/matlab/schemas');
addpath('/home/atlab/pipeline/pipeline/matlab/lib');
addpath(genpath('/home/atlab/pipeline/ca_source_extraction'));
addpath(genpath('/home/atlab/pipeline/hdf5matlab'));
addpath(genpath('/home/atlab/pipeline/oopsi'));
run /home/atlab/pipeline/mym/mymSetup;
run /home/atlab/pipeline/cvx/cvx_startup;
run /home/atlab/pipeline/hdf5matlab/setPath;
