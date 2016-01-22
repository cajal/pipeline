function plusCallback(src,~)

figHand = get(src,'parent');

[h,m] = mice.GUIs.getUIData(figHand);

% Clear previous error messages

h.errorMessage = findobj(figHand,'tag','errorMessage');
h.errorBox = findobj(figHand,'tag','errorBox');

delete(h.errorMessage);
delete(h.errorBox);

% Error checking

errorString = {};
errorCount = 0;

% Check when adding mouse to listbox:

% animal_id, dob, parent1, parent2 and line1 are required

if isempty(m.animal_id)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Ear tag number is required.';
end
    
if isempty(m.dob)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Date of birth is required.';
end
    
if isempty(m.parent1)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Father is required.';
end
    
if isempty(m.parent2)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Mother 1 is required.';
end
    
if isempty(m.line1)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Line 1 is required.';
end

% animal_id must be an int of length >=4

if ~ischar(m.animal_id)
    m.animal_id = char(m.animal_id);
end
x.animal_id = str2num(m.animal_id);

if isempty(x.animal_id)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Ear tag number must be numeric.';
end

if length(m.animal_id) < 4
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Ear tag number must be at least four numbers.';
end

% other_id, parent1, 2 and 3 must be varchar(20)

if length(m.other_id) > 20
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Alt ID must be less than 20 characters.';
end
    
if length(m.parent1) > 20
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Father ID must be less than 20 characters.';
end
    
if length(m.parent2) > 20
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Mother 1 ID must be less than 20 characters.';
end
    
if length(m.parent3) > 20
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Mother 2 ID must be less than 20 characters.';
end

% Parent id's cannot be repeated

p = {};

if ~isempty(m.parent1)
    p = [p m.parent1];
end
if ~isempty(m.parent2)
    p = [p m.parent2];
end
if ~isempty(m.parent3)
    p = [p m.parent3];
end

if length(p) ~= length(unique(p))
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Same parent is entered twice.';
end
    
% dob and dow must be dates

formatOut = 'yyyy-mm-dd';

if ~ischar(m.dob)
    m.dob = char(m.dob);
end

if ~ischar(m.dow)
    m.dow = char(m.dow);
end

if ~isempty(m.dob)
    try 
        m.dob = datestr(m.dob,formatOut);
    catch
        errorCount = errorCount + 1;
        errorString{errorCount} = 'Date of birth cannot be interpreted.';
    end
end

if ~isempty(m.dow) 
    try 
        m.dow = datestr(m.dow,formatOut);
    catch
        errorCount = errorCount + 1;
        errorString{errorCount} = 'Date of weaning cannot be interpreted.';
    end
end

% row must be a single letter

if ~isempty(m.row) && (isnumeric(m.row) || length(m.row) > 1)
    errorCount  = errorCount + 1;
    errorString{errorCount} = 'Row must be a single letter';
end

% mouse_notes must be varchar(4096)

if length(m.mouse_notes) > 4096
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Mouse notes cannot exceed 4096 characters.';
end

% check that animal id does not already exist in database
if errorCount == 0
    a = fetch(mice.Mice & ['animal_id=' m.animal_id]);
    if ~isempty(a)
        errorCount = errorCount + 1;
        errorString{errorCount} = 'Mouse ID already exists in the databse.';
    end
end

% If any line is C57Bl/6 or Fvb then it must be the only line and the
% genotype must be wild type

if (strcmp(m.line1,'C57Bl/6') || strcmp(m.line1,'Fvb')) && ~strcmp(m.genotype1, 'wild type')
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Lines C57Bl/6 and Fvb should only be used to designate pure wild type mice.';
end

if (strcmp(m.line1, 'C57Bl/6') || strcmp(m.line1,'Fvb')) && (~isempty(m.line2) || ~isempty(m.line3))
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Lines C57Bl/6 and Fvb should only be used to designate pure wild type mice.';
end

if (strcmp(m.line2, 'C57Bl/6') || strcmp(m.line2,'Fvb')) && (~isempty(m.line1) || ~isempty(m.line3))
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Lines C57Bl/6 and Fvb should only be used to designate pure wild type mice.';
end

if (strcmp(m.line3, 'C57Bl/6') || strcmp(m.line3,'Fvb')) && (~isempty(m.line1) || ~isempty(m.line2))
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Lines C57Bl/6 and Fvb should only be used to designate pure wild type mice.';
end

% wild type genotype can only be used for C57Bl/6 or Fvb lines.  

if strcmp(m.genotype1,'wild type') && ~strcmp(m.line1,'C57Bl/6') && ~strcmp(m.line1,'Fvb')
    errorCount = errorCount + 1;
    errorString{errorCount} = 'The wild type genotype should only be used to describe pure C57Bl/6 or Fvb lines.';
end

if strcmp(m.genotype2,'wild type') && ~strcmp(m.line2,'C57Bl/6') && ~strcmp(m.line2,'Fvb')
    errorCount = errorCount + 1;
    errorString{errorCount} = 'The wild type genotype should only be used to describe pure C57Bl/6 or Fvb lines.';
end

if strcmp(m.genotype3,'wild type') && ~strcmp(m.line3,'C57Bl/6') && ~strcmp(m.line3,'Fvb')
    errorCount = errorCount + 1;
    errorString{errorCount} = 'The wild type genotype should only be used to describe pure C57Bl/6 or Fvb lines.';
end

% at least one parent must share each line, (unless parents are not in the
% database)

if iscell(m.parent1)
    m.parent1 = char(m.parent1);
end

if iscell(m.parent2)
    m.parent2 = char(m.parent2);
end

if iscell(m.parent3)
    m.parent3 = char(m.parent3);
end

a = '';
if ~isempty(str2num(m.parent1))
    a = fetch(mice.Genotypes & ['animal_id=' m.parent1]);
else if ~isempty(m.parent1)
    a = fetch(mice.Mice & ['other_id="' m.parent1 '"']);
    a = fetch(mice.Genotypes & a);
    end
end

b = '';
if ~isempty(str2num(m.parent2))
    b = fetch(mice.Genotypes & ['animal_id=' m.parent2]);
else if ~isempty(m.parent2)
    b = fetch(mice.Mice & ['other_id="' m.parent2 '"']);
    b = fetch(mice.Genotypes & b);
    end
end

c = '';
if ~isempty(str2num(m.parent3))
    c = fetch(mice.Genotypes & ['animal_id=' m.parent3]);
elseif ~isempty(m.parent3)
    c = fetch(mice.Mice & ['other_id="' m.parent3 '"']);
    c = fetch(mice.Genotypes & c);
end

s = {};
sCount = 0;
if ~isempty(a)
    for i = 1:size(a,1)
        sCount = sCount + 1;
        s{sCount} = a(i).line;
    end
end

if ~isempty(b)
    for i = 1:size(b,1)
        sCount = sCount + 1;
        s{sCount} = b(i).line;
    end
end
if ~isempty(c)
    for i = 1:size(c,1)
        sCount = sCount + 1;
        s{sCount} = c(i).line;
    end
end

if ~isempty(a) && ~isempty(b)
    if ~isempty([a.line]) && ~isempty([b.line]) && ~isempty(m.line1) && ~any(strcmp(m.line1,s)) 
        errorCount = errorCount + 1;
        errorString{errorCount} = ['' m.line1 ' is not a line listed for the parents of this animal.'];
    end
    if ~isempty([a.line]) && ~isempty([b.line]) && ~isempty(m.line2) && ~any(strcmp(m.line2,s)) 
        errorCount = errorCount + 1;
        errorString{errorCount} = ['' m.line2 ' is not  a line listed for the parents of this animal.'];
    end
    if ~isempty([a.line]) && ~isempty([b.line]) && ~isempty(m.line3) && ~any(strcmp(m.line3,s)) 
        errorCount = errorCount + 1;
        errorString{errorCount} = ['' m.line3 ' is not  a line listed for the parents of this animal.'];
    end
end

% all lines from the parents must also be listed for the offspring

m.lines = {};
lineCount = 0;
if ~isempty(m.line1)
    lineCount = lineCount + 1;
    m.lines{lineCount} = m.line1;
end
if ~isempty(m.line2)
    lineCount = lineCount + 1;
    m.lines{lineCount} = m.line2;
end
if ~isempty(m.line3)
    lineCount = lineCount + 1;
    m.lines{lineCount} = m.line3;
end

for i = 1:length(s)
    if ~any(strcmp(s{i},m.lines)) && ~isempty(s{i}) && ~(strcmp(s{i},'C57Bl/6')) && ~(strcmp(s{i},'Fvb'))
        errorCount = errorCount + 1;
        errorString{errorCount} = 'A line from the parents is not listed.';
    end
end

% if any parent is homozygous, the offspring cannot be negative for that
% transgene

if ~isempty(a)
    a = fetch(mice.Genotypes & a & 'genotype = "homozygous"');
    if ~isempty(m.line1) && ~isempty(a) && strcmp(m.line1,a.line) && strcmp('negative',m.genotype1);
        errorCount = errorCount + 1;
        errorString{errorCount} = ['Animal cannot be negative for ' m.line1 ' if a parent is homozygous.'];
    end
    if ~isempty(m.line2) && ~isempty(a) && strcmp(m.line2,a.line) && strcmp('negative',m.genotype2);
        errorCount = errorCount + 1;
        errorString{errorCount} = ['Animal cannot be negative for ' m.line2 ' if a parent is homozygous.'];
    end
    if ~isempty(m.line3) && ~isempty(a) && strcmp(m.line3,a.line) && strcmp('negative',m.genotype3);
        errorCount = errorCount + 1;
        errorString{errorCount} = ['Animal cannot be negative for ' m.line3 ' if a parent is homozygous.'];
    end
end

if ~isempty(b)
    b = fetch(mice.Genotypes & b & 'genotype = "homozygous"');
    if ~isempty(m.line1) && ~isempty(b) && strcmp(m.line1,b.line) && strcmp('negative',m.genotype1);
        errorCount = errorCount + 1;
        errorString{errorCount} = ['Animal cannot be negative for ' m.line1 ' if a parent is homozygous.'];
    end
    if ~isempty(m.line2) && ~isempty(b) && strcmp(m.line2,b.line) && strcmp('negative',m.genotype2);
        errorCount = errorCount + 1;
        errorString{errorCount} = ['Animal cannot be negative for ' m.line2 ' if a parent is homozygous.'];
    end
    if ~isempty(m.line3) && ~isempty(b) && strcmp(m.line3,b.line) && strcmp('negative',m.genotype3);
        errorCount = errorCount + 1;
        errorString{errorCount} = ['Animal cannot be negative for ' m.line3 ' if a parent is homozygous.'];
    end
end

if ~isempty(c)
    c = fetch(mice.Genotypes & c & 'genotype = "homozygous"');
    if ~isempty(m.line1) && ~isempty(c) && strcmp(m.line1,c.line) && strcmp('negative',m.genotype1);
        errorCount = errorCount + 1;
        errorString{errorCount} = ['Animal cannot be negative for ' m.line1 ' if a parent is homozygous.'];
    end
    if ~isempty(m.line2) && ~isempty(c) && strcmp(m.line2,c.line) && strcmp('negative',m.genotype2);
        errorCount = errorCount + 1;
        errorString{errorCount} = ['Animal cannot be negative for ' m.line2 ' if a parent is homozygous.'];
    end
    if ~isempty(m.line3) && ~isempty(c) && strcmp(m.line3,c.line) && strcmp('negative',m.genotype3);
        errorCount = errorCount + 1;
        errorString{errorCount} = ['Animal cannot be negative for ' m.line3 ' if a parent is homozygous.'];
    end
end

% if the mouse is in Taub, the room cannot be VK3. if the mouse is in TMF
% the room must be VK3.

if strcmp(m.facility,'Taub') && strcmp(m.room,'VK3')
    errorCount = errorCount + 1;
    errorString{errorCount} = 'If the mouse is in Taub it cannot be in room VK3.';
end

if strcmp(m.facility,'TMF') && ~strcmp(m.room,'VK3')
    errorCount = errorCount + 1;
    errorString{errorCount} = 'If the mouse is in TMF it must be in room VK3.';
end

% add to  table if there are no errors 
mouseCount = size(m.new_mice,1);
duplicateCount = 0;

if ~isempty(m.other_id) && ~ischar(m.other_id)
    m.other_id = char(m.other_id);
end

if ~isempty(m.parent1) && ~ischar(m.parent1)
    m.parent1 = char(m.parent1);
end

if ~isempty(m.parent2) && ~ischar(m.parent2)
    m.parent2 = char(m.parent2);
end

if ~isempty(m.parent3) && ~ischar(m.parent3)
    m.parent3 = char(m.parent3);
end

if ~isempty(m.rack) && ~ischar(m.rack)
    m.rack = char(m.rack);
end

if ~isempty(m.row) && ~ischar(m.row)
    m.row = char(m.row);
end

if ~isempty(m.mouse_notes) && ~ischar(m.mouse_notes)
    m.mouse_notes = char(m.mouse_notes);
end

if ~isempty(m.new_mice)
    mouseTable = m.new_mice;
end

if isempty(errorString) && isempty(m.new_mice) 
    mouseCount = mouseCount + 1;
    mouseTable(mouseCount,:) = {m.animal_id m.other_id m.dob m.dow m.parent1 m.parent2 m.parent3 m.sex m.color m.ear_punch m.line1 m.genotype1 m.line2 m.genotype2 m.line3 m.genotype3 m.owner m.facility m.room m.rack m.row m.mouse_notes};
    set(h.new_mice,'Data',mouseTable);
end

if isempty(errorString) && ~isempty(m.new_mice)
    for i = 1: size(m.new_mice,1)
        if m.animal_id == m.new_mice{i,1}
            mouseTable(i,:) = {m.animal_id char(m.other_id) m.dob m.dow m.parent1 m.parent2 m.parent3 m.sex m.color m.ear_punch m.line1 m.genotype1 m.line2 m.genotype2 m.line3 m.genotype3 m.owner m.facility m.room m.rack m.row m.mouse_notes};
            duplicateCount = duplicateCount + 1;
            set(h.new_mice,'Data',mouseTable);
        end
    end
end

if isempty(errorString) && ~isempty(m.new_mice) && duplicateCount == 0
    mouseCount = mouseCount + 1;
    mouseTable(mouseCount,:) = {m.animal_id m.other_id m.dob m.dow m.parent1 m.parent2 m.parent3 m.sex m.color m.ear_punch m.line1 m.genotype1 m.line2 m.genotype2 m.line3 m.genotype3 m.owner m.facility m.room char(m.rack) m.row m.mouse_notes};
    set(h.new_mice,'Data',mouseTable);
end

if ~isempty(errorString)
    h.errorMessage = uicontrol('style','text','String',['Cannot add mouse due to the following errors: '], 'position', [300 569 500 16],'fontsize',14,'tag','errorMessage');
    h.errorBox = uicontrol('style','listbox','string',errorString,'tag','errorBox','position',[300 519 500 50]);
else h.autopopulate = findobj(figHand,'tag','autoBox');
    if get(h.autopopulate,'Value') == 1
        set(h.animal_id,'String',[x.animal_id+1]);
        set(h.other_id,'String','');
        set(h.ear_punch,'Value',1);
        set(h.mouse_notes,'String','');
    else mice.GUIs.clearEntry(src);
    end
end

end