% startup.m
if ispc
   drive = 'Z:' ;
elseif ismac
    drive = [filesep 'Volumes' filesep 'lab'] ;
else
    drive = [filesep 'mnt' filesep 'lab'] ;
end ;
setenv('DJ_HOST', 'at-database.ad.bcm.edu');
setenv('DJ_USER','spatel');
setenv('DJ_PASS','spatel12');
addpath([drive filesep 'home' filesep 'atlab' filesep 'pipeline']) ;
addpath([drive filesep 'home' filesep 'atlab' filesep 'pipeline' filesep 'datajoint-matlab']) ;
addpath(genpath([drive filesep 'home' filesep 'atlab' filesep 'pipeline' filesep 'commons' filesep 'schemas']));
addpath([drive filesep 'home' filesep 'atlab' filesep 'pipeline' filesep 'commons' filesep 'lib']);
addpath([drive filesep 'home' filesep 'atlab' filesep 'pipeline' filesep 'pipeline' filesep 'matlab' filesep]);
addpath([drive filesep 'home' filesep 'atlab' filesep 'pipeline' filesep 'pipeline' filesep 'matlab' filesep 'schemas']);
addpath([drive filesep 'home' filesep 'atlab' filesep 'pipeline' filesep 'pipeline' filesep 'matlab' filesep 'lib']);
addpath(genpath([drive filesep 'home' filesep 'atlab' filesep 'pipeline' filesep 'ca_source_extraction']));
addpath(genpath([drive filesep 'home' filesep 'atlab' filesep 'pipeline' filesep 'hdf5matlab']));
addpath(genpath([drive filesep 'home' filesep 'atlab' filesep 'pipeline' filesep 'DataStorage' filesep 'matlab']));
addpath(genpath([drive filesep 'home' filesep 'atlab' filesep 'pipeline' filesep 'DataStorage' filesep 'matlab' filesep 'lib']));
run([drive filesep 'home' filesep 'atlab' filesep 'pipeline' filesep 'mym' filesep 'mymSetup']);
run([drive filesep 'home' filesep 'atlab' filesep 'pipeline' filesep 'cvx' filesep 'cvx_startup']);
run([drive filesep 'home' filesep 'atlab' filesep 'pipeline' filesep 'hdf5matlab' filesep 'setPath']);
userscript
