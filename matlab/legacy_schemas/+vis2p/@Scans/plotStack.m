function plotStack(obj,key)

channel = 'ch2';

global ind
global data
global pointi
global isAOD

name = fetch1( Experiments*Scans(key),'file_name' );
isAOD =strcmp(fetch1(Scans(key),'scan_prog'),'AOD');

if isAOD
    if ~strcmp(fetch1(Scans(key),'aim'),'stack')
        display('Detecting volume..')
        u_ind = strfind(name,'_');
        
        volumename = fetchn(Scans(rmfield(key,'scan_idx'),[...
            'file_name LIKE "' name(1:u_ind(1)) ...
            '%" and scan_idx > ' num2str(key.scan_idx -2) ...
            ' and scan_idx < ' num2str(key.scan_idx + 2) ...
            ' and aim = "Stack"' ...
            ]),'file_name');
        if isempty(volumename)
            volumename = fetchn(Scans(rmfield(key,'scan_idx'),[...
                'file_name LIKE "' name(1:u_ind(1)) ...
                '%" and scan_idx > ' num2str(key.scan_idx -3) ...
                ' and scan_idx < ' num2str(key.scan_idx + 3) ...
                ' and aim = "Stack"' ...
                ]),'file_name');
        end
        
        if isempty(volumename)
            display('No volume, skipping...')
            return
        end
        volumename = volumename{1};
    end
    
    dirs = dir(['M:\Mouse\'  volumename(1:10) '*']);
    volumename = ['M:\Mouse\' dirs(1).name '\AODAcq\' volumename];
    [x, y, z] = fetchn(MaskCells(key),'img_x','img_y','img_z');
    points = [x, y, z];
    fn = aodReader(volumename,'Volume',channel);
    
    % convert points to index
    pointi = nan(size(points));
    for icell = 1:size(points,1)
        pointi(icell,1) = find(roundall(points(icell,1),0.1) == roundall(fn.x,0.1));
        pointi(icell,2) = find(roundall(points(icell,2),0.1) == roundall(fn.y,0.1));
        pointi(icell,3) = find(roundall(points(icell,3),0.1) == roundall(fn.z,0.1));
    end
    
    data = normalize(fn(:,:,:,1));
    data = repmat(reshape(data,size(data,1),size(data,2),1,size(data,3)),[1 1 3 1]);
    
else
    tp = tpReader(Scans(key));
    
    gr = getData(tp.imCh{1}); gr= reshape(gr,size(gr,1),size(gr,2),1,size(gr,3));
    rd = getData(tp.imCh{2}); rd= reshape(rd,size(rd,1),size(rd,2),1,size(rd,3));
    
    data = normalize(double(rd));
    data(:,:,2,:) = normalize(double(gr));
    data(:,:,3,:) = zeros(size(gr));
end


h = figure('NumberTitle','off','Menubar','none',...
    'Name','Find the patched cell',...
    'Position',[560 728 400 400],...
    'KeyPressFcn',@dispkeyevent);

ind = 1;
image(normalize(data(:,:,:,ind)))
axis image
if isAOD
    hold on
    celli = find(pointi(:,3)==ind);
    for icell = 1:length(celli)
        text(pointi(celli(icell),2),pointi(celli(icell),1),'+',...
            'HorizontalAlignment','center','VerticalAlignment','middle')
        text(pointi(celli(icell),2),pointi(celli(icell),1),num2str(celli(icell)),...
            'VerticalAlignment','top')
    end
end

function dispkeyevent(~, event)
global ind
global data
global pointi
global isAOD

if strcmp(event.Key,'uparrow')
    if ind<size(data,4)
        ind = ind+1;
    end
elseif strcmp(event.Key,'downarrow')
    if ind>1
        ind = ind-1;
    end
end
clf
image(normalize(data(:,:,:,ind)))
axis image
if isAOD
    hold on
    celli = find(pointi(:,3)==ind);
    for icell = 1:length(celli)
        text(pointi(celli(icell),2),pointi(celli(icell),1),'+',...
            'HorizontalAlignment','center','VerticalAlignment','middle')
        text(pointi(celli(icell),2),pointi(celli(icell),1),num2str(celli(icell)),...
            'VerticalAlignment','top')
    end
end