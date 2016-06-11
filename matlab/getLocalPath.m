function p = getLocalPath(p,os)
% Converts path names to local operating system format using lab conventions.
%
%    localPath = getLocalPath(inputPath) converts inputPath to local OS format
%    using lab conventions. The following paths are converted:
%
%       input        Linux            Windows      Mac
%       /lab         /mnt/lab         Z:\          /Volumes/lab
%       /stor01      /mnt/stor01      Y:\          /Volumes/stor01
%       /stor02      /mnt/stor02      X:\          /Volumes/stor02
%       /scratch01   /mnt/scratch01   V:\          /Volumes/scratch01
%       /scratch03   /mnt/scratch03   T:\          /Volumes/scratch03
%       /stimulation /mnt/stor01/stimulation Y:\stor01\stimulation  /Volumes/stor01/stimulation
%       /processed   /mnt/stor01/processed   Y:\stor01\processed    /Volumes/stor01/processed
%       /raw         /mnt/at_scratch  W:           /Volumes/at_scratch
%       ~            $HOME            %homepath%   ~
%
%    localPath = getLocalPath(inputPath,OS) will return the path in the format
%    of the operating system specified in OS ('global' | 'linux' |'win' | 'mac')
%
% AE 2011-04-01

% determine operating system;
if nargin < 2
    os = computer;
end
os = os(1:min(3,length(os)));

% convert file separators if necessary
p = strrep(p,'\','/');

% a few fixes for outdated paths
p = strrep(p,'/stor01/hammer','/at_scratch/hammer');
p = strrep(p,'/stor02/hammer','/at_scratch/hammer');
p = strrep(p,'hammer/ben','hammer/Ben');

% local os' column
home = 'Windows Home';
switch lower(os)
    case 'glo'
        local = 1;
    case {'lin','gln'}
        local = 2;
        home = getenv('HOME');
    case {'win','pcw'}
        local = 3;
        home = [getenv('HOMEDRIVE') getenv('HOMEPATH')];
    case 'mac'
        local = 4;
    otherwise
        error('unknown OS');
end

% mapping table [INPUT LINUX WINDOWS MAC]
mapping = {
    '/stimulation','/mnt/stor01/stimulation','Y:/stimulation','/Volumes/stor01/stimulation'
    '/processed','/mnt/stor01/processed','Y:/processed','/Volumes/stor01/processed'
    '/lab','/mnt/lab','Z:','/Volumes/lab'
    '/stor01','/mnt/stor01','Y:','/Volumes/stor01'
    '/stor02','/mnt/stor02','X:','/Volumes/stor02'
    '/scratch01','/mnt/scratch01','V:','/Volumes/scratch01'
    '/scratch03','/mnt/scratch03','T:','/Volumes/scratch03'
    '/at_scratch','/mnt/at_scratch','W:','/Volumes/at_scratch'
    '/raw','/mnt/at_scratch','W:','/Volumes/at_scratch'
    '/2P2Drive','/mnt/2P2Drive','Q:','/Volumes/2P2Drive'
    '/manolism','/mnt/manolism','M:','/Volumes/M'
    '/dataCache','/media/Data','S:','xx'
    '~',home,home,'~'
    };

% convert path
sz = size(mapping);
for i = 1:sz(1)
    for j = 1:sz(2)
        n = length(mapping{i,j});
        if j ~= local && strncmpi(p,mapping{i,j},n)
            p = fullfile(mapping{i,local},p(n+2:end));
            break;
        end
    end
end

if filesep == '\' && ~isequal(lower(os),'glo')
    p = strrep(p,'/','\');
else
    p = strrep(p,'\','/');
end

