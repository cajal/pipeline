function findViewMice(src,~)

figHand = get(src,'parent');

[h,m] = mice.GUIs.getUIData(figHand);

% clear old error messages

h.errorMessage = findobj(figHand,'tag','errorMessage');
h.errorBox = findobj(figHand,'tag','errorBox');

delete(h.errorMessage);
delete(h.errorBox);

% get UI data

h.range_start = findobj(figHand,'tag','rangeStart');
h.range_end = findobj(figHand,'tag','rangeEnd');
h.table = findobj(figHand,'tag','miceTable');


h.parent1 = findobj(figHand,'tag','parent1Field');
h.parent2 = findobj(figHand,'tag','parent2Field');
h.line1 = findobj(figHand,'tag','line1Field');
h.line2 = findobj(figHand,'tag','line2Field');
h.line3 = findobj(figHand,'tag','line3Field');
h.used = findobj(figHand,'tag','usedBox');

m.parent1 = get(h.parent1,'string');
m.parent2 = get(h.parent2,'string');
m.range_start = get(h.range_start,'string');
m.range_end = get(h.range_end,'string');
v = get(h.line1,'value');
s = get(h.line1,'string');
m.line1 = s{v};
v = get(h.line2,'value');
s = get(h.line2,'string');
m.line2 = s{v};
v = get(h.line3,'value');
s = get(h.line3,'string');
m.line3 = s{v};
v = get(h.genotype1,'value');
s = get(h.genotype1,'string');
m.genotype1 = s{v};
v = get(h.genotype2,'value');
s = get(h.genotype2,'string');
m.genotype2 = s{v};
v = get(h.genotype3,'value');
s = get(h.genotype3,'string');
m.genotype3 = s{v};
m.mouse_notes = get(h.mouse_notes,'string');
m.used = get(h.used,'Value');

% error checking

errorCount = 0;
errorString = {};

% both start and end ID must be entered, not just one or the other

if (isempty(m.range_start) && ~isempty(m.range_end)) || (isempty(m.range_end) && ~isempty(m.range_start))
    errorCount = errorCount + 1;
    errorString{errorCount} = 'ID range must include a start and end number.';
end

% All ID numbers must be in the database

rangeID = {};
kk=1;
if ~isempty(m.range_start) && ~isempty(m.range_end)
    idx = str2num(m.range_start):str2num(m.range_end);
    for i = 1:length(idx)
        fetchedMouse = fetch(mice.Mice & ['animal_id=' num2str(idx(i)) '' ]);
        if ~isempty(fetchedMouse)
            id(kk) = fetchedMouse;
            rangeID{kk} = id(kk).animal_id;
            kk=kk+1;
        end
    end
    if isempty(rangeID)
        errorCount = errorCount + 1;
        errorString{errorCount} = ['No mice found in this range of mice IDs.'];
    end
end

% Parents listed must have at least one offspring
parent1 = {};
parent2 = {};
a1 = {};
a2 = {};
a3 ={};
b1 = {};
b2 = {};
b3 = {};

if ~isempty(m.parent1) 
    a1 = fetch(mice.Parents & ['parent_id="' m.parent1 '"']);
    try parent1 = fetch(mice.Mice & ['animal_id=' m.parent1 ''],'*');
    end
    if isempty(parent1)
        try parent1 = fetch(mice.Mice & ['other_id="' m.parent1 '"'],'*');
        end
    end
    try a2 = fetch(mice.Parents & ['parent_id="' num2str(parent1.animal_id) '"']);
    end
    try a3 = fetch(mice.Parents & ['parent_id="' parent1.other_id '"']);
    end
    a = fetch(mice.Parents & ((mice.Parents & a1) | (mice.Parents & a2) | (mice.Parents & a3)));
end

if ~isempty(m.parent2) 
    b1 = fetch(mice.Parents & ['parent_id="' m.parent2 '"']);
    try parent2 = fetch(mice.Mice & ['animal_id=' m.parent2 ''],'*');
    end
    if isempty(parent2)
        try parent2 = fetch(mice.Mice & ['other_id="' m.parent2 '"'],'*');
        end
    end
    try b2 = fetch(mice.Parents & ['parent_id="' num2str(parent2.animal_id) '"']);
    end
    try b3 = fetch(mice.Parents & ['parent_id="' parent2.other_id '"']);
    end
    b = fetch(mice.Parents & ((mice.Parents & b1) | (mice.Parents & b2) | (mice.Parents & b3)));
end

offspringCount = 0;
offspringID = {};

if ~isempty(m.parent1) && ~isempty(m.parent2)
    for i = 1:size(a,1)
        for j = 1:size(b,1)
            if a(i).animal_id == b(j).animal_id
                offspringCount = offspringCount + 1;
                offspringID{offspringCount} = a(i).animal_id;
            end
        end
    end
    if offspringCount == 0
        errorCount = errorCount + 1;
        errorString{errorCount} = 'There are no offspring with these two parents.';
    end
end

if ~isempty(m.parent1) && isempty(m.parent2)
    for i = 1:size(a,1)
        offspringID{i} = a(i).animal_id;
    end
    if isempty(a)
        errorCount = errorCount + 1;
        errorString{errorCount} = ['There are no offspring from parent ' m.parent1 '.'];
    end
end

if isempty(m.parent1) && ~isempty(m.parent2)
    for i = 1:size(b,1)
        offspringID{i} = b(i).animal_id;
    end
    if isempty(b)
        errorCount = errorCount + 1;
        errorString{errorCount} = ['There are no offspring from parent ' m.parent2 '.'];
    end
end

% Combine all restrictions into a single list of ID's and continue to check
% errors
mouseCount = 0;
mouseID = {};

% find all mice in the specified range

if ~isempty(rangeID)
    for i = 1:length(rangeID)
        mouseCount = mouseCount + 1;
        mouseID{mouseCount} = rangeID{i};
    end
end

% restrict to mice with specified parents

mouseCount=0;
if ~isempty(offspringID)
    if ~isempty(mouseID)
        for i = 1:length(mouseID)
            for j = 1:length(offspringID)
                if mouseID{i} == offspringID{j}
                    mouseCount = mouseCount + 1;
                    newmouseID{mouseCount} = mouseID{i};
                end
            end
        end
    else
        for i = 1: length(offspringID)
            mouseCount = mouseCount + 1;
            newmouseID{mouseCount} = offspringID{i};
        end
    end
    try mouseID = newmouseID;
    catch
        errorCount = errorCount + 1;
        errorString{errorCount} = 'There are no mice matching these criteria.';
    end
end

% restrict to mice of specified sex

mouseCount=0;
newmouseID = {};
if ~isempty(m.sex)
    if ~isempty(mouseID)
        for i = 1:length(mouseID)
            a = fetch(mice.Mice & ['animal_id=' num2str(mouseID{i})],'sex');
            if strcmp(a.sex,m.sex)
                mouseCount = mouseCount + 1;
                newmouseID{mouseCount} = mouseID{i};
            end
        end 
    else
        a = fetch(mice.Mice & ['sex="' m.sex '"']);
        for i = 1: length(a)
            mouseCount = mouseCount + 1;
            newmouseID{mouseCount} = a(i).animal_id;
        end
    end
    try mouseID = newmouseID;
    if isempty(mouseID)
        errorCount = errorCount + 1;
        errorString{errorCount} = 'There are no mice matching these search criteria.';
    end
    end
end

% restrict to mice of specified color

mouseCount=0;
newmouseID = {};
if ~isempty(m.color)
    if ~isempty(mouseID)
        for i = 1:length(mouseID)
            a = fetch(mice.Mice & ['animal_id=' num2str(mouseID{i})],'color');
            if strcmp(a.color,m.color)
                mouseCount = mouseCount + 1;
                newmouseID{mouseCount} = mouseID{i};
            end
        end 
    else
        a = fetch(mice.Mice & ['color="' m.color '"']);
        for i = 1: length(a)
            mouseCount = mouseCount + 1;
            newmouseID{mouseCount} = a(i).animal_id;
        end
    end
    try mouseID = newmouseID;
    if isempty(mouseID)
        errorCount = errorCount + 1;
        errorString{errorCount} = 'There are no mice matching these search criteria.';
    end
    end
end

% restrict to mice of specified ear punch

mouseCount=0;
newmouseID = {};
if ~isempty(m.ear_punch)
    if ~isempty(mouseID)
        for i = 1:length(mouseID)
            a = fetch(mice.Mice & ['animal_id=' num2str(mouseID{i})],'ear_punch');
            if strcmp(a.ear_punch,m.ear_punch)
                mouseCount = mouseCount + 1;
                newmouseID{mouseCount} = mouseID{i};
            end
        end 
    else
        a = fetch(mice.Mice & ['ear_punch="' m.ear_punch '"']);
        for i = 1: length(a)
            mouseCount = mouseCount + 1;
            newmouseID{mouseCount} = a(i).animal_id;
        end
    end
    try mouseID = newmouseID;
    if isempty(mouseID)
        errorCount = errorCount + 1;
        errorString{errorCount} = 'There are no mice matching these search criteria.';
    end
    end
end

% restrict to mice of specified owner

mouseCount=0;
newmouseID = {};
if ~isempty(m.owner)
    if ~isempty(mouseID)
        for i = 1:length(mouseID)
            a = fetch(mice.Mice & ['animal_id=' num2str(mouseID{i})],'owner');
            if strcmp(a.owner,m.owner)
                mouseCount = mouseCount + 1;
                newmouseID{mouseCount} = mouseID{i};
            end
        end 
    else
        a = fetch(mice.Mice & ['owner="' m.owner '"']);
        for i = 1: length(a)
            mouseCount = mouseCount + 1;
            newmouseID{mouseCount} = a(i).animal_id;
        end
    end
    try mouseID = newmouseID;
    if isempty(mouseID)
        errorCount = errorCount + 1;
        errorString{errorCount} = 'There are no mice matching these search criteria.';
    end
    end
end

% restrict to mice of specified facility

mouseCount=0;
newmouseID = {};
if ~isempty(m.facility)
    if ~isempty(mouseID)
        for i = 1:length(mouseID)
            a = fetch(mice.Mice & ['animal_id=' num2str(mouseID{i})],'facility');
            if strcmp(a.facility,m.facility)
                mouseCount = mouseCount + 1;
                newmouseID{mouseCount} = mouseID{i};
            end
        end 
    else
        a = fetch(mice.Mice & ['facility="' m.facility '"']);
        for i = 1: length(a)
            mouseCount = mouseCount + 1;
            newmouseID{mouseCount} = a(i).animal_id;
        end
    end
    try mouseID = newmouseID;
    if isempty(mouseID)
        errorCount = errorCount + 1;
        errorString{errorCount} = 'There are no mice matching these search criteria.';
    end
    end
end

% restrict to mice of specified room

mouseCount=0;
newmouseID = {};
if ~isempty(m.room)
    if ~isempty(mouseID)
        for i = 1:length(mouseID)
            a = fetch(mice.Mice & ['animal_id=' num2str(mouseID{i})],'room');
            if strcmp(a.room,m.room)
                mouseCount = mouseCount + 1;
                newmouseID{mouseCount} = mouseID{i};
            end
        end 
    else
        a = fetch(mice.Mice & ['room="' m.room '"']);
        for i = 1: length(a)
            mouseCount = mouseCount + 1;
            newmouseID{mouseCount} = a(i).animal_id;
        end
    end
    try mouseID = newmouseID;
    if isempty(mouseID)
        errorCount = errorCount + 1;
        errorString{errorCount} = 'There are no mice matching these search criteria.';
    end
    end
end

% if genotype is specified, the line must also be specified

if ~isempty(m.genotype1) && isempty(m.line1)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Please select line 1 or remove genotype 1 restriction.';
end

if ~isempty(m.genotype2) && isempty(m.line2)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Please select line 2 or remove genotype 2 restriction.';
end

if ~isempty(m.genotype3) && isempty(m.line3)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Please select line 3 or remove genotype 3 restriction.';
end

% restrict by specified genotype

mouseCount=0;
newmouseID = {};
if ~isempty(m.line1)
    if ~isempty(mouseID)
        for i = 1:length(mouseID)
            a = fetch(mice.Genotypes & ['animal_id=' num2str(mouseID{i})],'*');
            for j = 1:size(a,1)
                if strcmp(a(j).line,m.line1)
                    if ~isempty(m.genotype1)
                        if strcmp(a(j).genotype,m.genotype1)
                            mouseCount = mouseCount + 1;
                            newmouseID{mouseCount} = mouseID{i};
                        end
                    else
                        mouseCount = mouseCount + 1;
                        newmouseID{mouseCount} = mouseID{i};
                    end
                end
            end
        end
    elseif ~isempty(m.genotype1)
        a = fetch(mice.Genotypes & ['line="' m.line1 '"'] & ['genotype="' m.genotype1 '"']);
        for i = 1:size(a,1)
            mouseCount = mouseCount + 1;
            newmouseID{mouseCount} = a(i).animal_id;
        end
    else 
        a = fetch(mice.Genotypes & ['line="' m.line1 '"']);
        for i = 1:size(a,1)
            mouseCount = mouseCount + 1;
            newmouseID{mouseCount} = a(i).animal_id;
        end
    end
    try mouseID = newmouseID;
        if isempty(mouseID)
            errorCount = errorCount + 1;
            errorString{errorCount} = 'There are no mice matching these search criteria.';
        end
    end
end

mouseCount=0;
newmouseID = {};
if ~isempty(m.line2)
    if ~isempty(mouseID)
        for i = 1:length(mouseID)
            a = fetch(mice.Genotypes & ['animal_id=' num2str(mouseID{i})],'*');
            for j = 1:size(a,1)
                if strcmp(a(j).line,m.line2)
                    if ~isempty(m.genotype2)
                        if strcmp(a(j).genotype,m.genotype2)
                            mouseCount = mouseCount + 1;
                            newmouseID{mouseCount} = mouseID{i};
                        end
                    else
                        mouseCount = mouseCount + 1;
                        newmouseID{mouseCount} = mouseID{i};
                    end
                end
            end
        end
    elseif ~isempty(m.genotype2)
        a = fetch(mice.Genotypes & ['line="' m.line2 '"'] & ['genotype="' m.genotype2 '"']);
        for i = 1:size(a,1)
            mouseCount = mouseCount + 1;
            newmouseID{mouseCount} = a(i).animal_id;
        end
    else 
        a = fetch(mice.Genotypes & ['line="' m.line2 '"']);
        for i = 1:size(a,1)
            mouseCount = mouseCount + 1;
            newmouseID{mouseCount} = a(i).animal_id;
        end
    end
    try mouseID = newmouseID;
        if isempty(mouseID)
            errorCount = errorCount + 1;
            errorString{errorCount} = 'There are no mice matching these search criteria.';
        end
    end
end

mouseCount=0;
newmouseID = {};
if ~isempty(m.line3)
    if ~isempty(mouseID)
        for i = 1:length(mouseID)
            a = fetch(mice.Genotypes & ['animal_id=' num2str(mouseID{i})],'*');
            for j = 1:size(a,1)
                if strcmp(a(j).line,m.line3)
                    if ~isempty(m.genotype3)
                        if strcmp(a(j).genotype,m.genotype3)
                            mouseCount = mouseCount + 1;
                            newmouseID{mouseCount} = mouseID{i};
                        end
                    else
                        mouseCount = mouseCount + 1;
                        newmouseID{mouseCount} = mouseID{i};
                    end
                end
            end
        end
    elseif ~isempty(m.genotype3)
        a = fetch(mice.Genotypes & ['line="' m.line3 '"'] & ['genotype="' m.genotype3 '"']);
        for i = 1:size(a,1)
            mouseCount = mouseCount + 1;
            newmouseID{mouseCount} = a(i).animal_id;
        end
    else 
        a = fetch(mice.Genotypes & ['line="' m.line3 '"']);
        for i = 1:size(a,1)
            mouseCount = mouseCount + 1;
            newmouseID{mouseCount} = a(i).animal_id;
        end
    end
    try mouseID = newmouseID;
        if isempty(mouseID)
            errorCount = errorCount + 1;
            errorString{errorCount} = 'There are no mice matching these search criteria.';
        end
    end
end

% restrict to living mice unless "include used mice" box is checked

mouseCount=0;
newmouseID = {};
if m.used == 0
    for i = 1:length(mouseID)
        a = fetch(mice.Death & ['animal_id=' num2str(mouseID{i})]);
        if isempty(a)
            mouseCount = mouseCount + 1;
            newmouseID{mouseCount} = mouseID{i};
        end
    end
    try mouseID = newmouseID;
        if isempty(mouseID)
            errorCount = errorCount + 1;
            errorString{errorCount} = 'There are no mice matching these search criteria.';
        end
    end
end

% if there are errors, display them to the user

if ~isempty(errorString)
    h.errorMessage = uicontrol('style','text','String',['Cannot find mice due to the following errors: '], 'position', [400 580 300 29],'fontsize',14,'tag','errorMessage');
    h.errorBox = uicontrol('style','listbox','string',errorString,'tag','errorBox','position',[400 550 300 29]);
    set(h.table,'data',{},'rowname','');
    return
end

% if there are no errors, display requested mice in table

mouseTable = cell(length(mouseID),22);

for i = 1:length(mouseID)
    mouseTable{i,1} = mouseID{i};
    mouse = fetch(mice.Mice & ['animal_id=' num2str(mouseID{i})],'*');
    mouseTable{i,2} = mouse.other_id;
    mouseTable{i,3} = mouse.dob;
    mouseTable{i,4} = mouse.dow;
    mouseTable{i,8} = mouse.sex;
    mouseTable{i,9} = mouse.color;
    mouseTable{i,10} = mouse.ear_punch;
    mouseTable{i,17} = mouse.owner;
    mouseTable{i,18} = mouse.facility;
    mouseTable{i,19} = mouse.room;
    mouseTable{i,20} = mouse.rack;
    mouseTable{i,21} = mouse.row;
    mouseTable{i,22} = mouse.mouse_notes;
    genotypes = fetch(mice.Genotypes & ['animal_id=' num2str(mouseID{i})],'*');
    for j = 1:size(genotypes,1)
        mouseTable{i,(2*j+9)} = genotypes(j).line;
        mouseTable{i,(2*j+10)} = genotypes(j).genotype;
    end
    parents = fetch(mice.Parents & ['animal_id=' num2str(mouseID{i})],'*');
    for j = 1:size(parents,1)
        mouseTable{i,(j+4)} = parents(j).parent_id;
    end
end
set(h.table,'data',mouseTable);

end
