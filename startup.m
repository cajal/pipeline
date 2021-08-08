% setup PsychToolbox
home_dir = getenv('HOME');
pt_path = fullfile(home_dir, 'toolbox/Psychtoolbox');
if exist(pt_path, 'dir') && ~any(regexp(path, [filesep 'Psychtoolbox']))
  fprintf('Psychtoolbox is not installed. Please run "configPsychtoolbox"\n');
end

setPath;
setDJ;
