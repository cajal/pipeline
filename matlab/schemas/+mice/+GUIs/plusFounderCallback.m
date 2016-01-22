function plusFounderCallback(src,~)

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

% animal_id, doa, line and source are required

if isempty(m.animal_id)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Ear tag number is required.';
end

if isempty(m.source)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Source is required.';
end

if isempty(m.line)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Line is required.';
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

% dob and dow must be dates

formatOut = 'yyyy-mm-dd';

if ~ischar(m.doa)
    m.doa = char(m.doa);
end

if ~ischar(m.dob)
    m.dob = char(m.dob);
end

if ~ischar(m.dow)
    m.dow = char(m.dow);
end

if ~isempty(m.doa)
    try 
        m.doa = datestr(m.doa,formatOut);
    catch
        errorCount = errorCount + 1;
        errorString{errorCount} = 'Date of arrival cannot be interpreted.';
    end
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

% rack must be tinyint

if ~ischar(m.rack)
    m.rack = char(m.rack);
end

if ~isempty(m.rack)
    x.rack = str2num(m.rack);
    if isempty(x.rack)
        errorCount = errorCount + 1;
        errorString{errorCount} = 'Rack must be a number.';
    end
end

if length(m.rack) > 2
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Rack must be less than three digits.';
end

% row must be a single letter

if ~isempty(m.row) && (isnumeric(m.row) || length(m.row) > 1)
    errorCount  = errorCount + 1;
    errorString{errorCount} = 'Row must be a single letter';
end

% founder_notes must be varchar(4096)

if length(m.founder_notes) > 4096
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Founder notes cannot exceed 4096 characters.';
end

% source must be varchar(4096)

if length(m.source) > 4096
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Source notes cannot exceed 4096 characters.';
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

if (strcmp(m.line,'C57Bl/6') || strcmp(m.line,'Fvb')) && ~strcmp(m.genotype, 'wild type')
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Lines C57Bl/6 and Fvb should only be used to designate pure wild type mice.';
end

% wild type genotype can only be used for C57Bl/6 or Fvb lines.  

if strcmp(m.genotype,'wild type') && ~strcmp(m.line,'C57Bl/6') && ~strcmp(m.line,'Fvb')
    errorCount = errorCount + 1;
    errorString{errorCount} = 'The wild type genotype should only be used to describe pure C57Bl/6 or Fvb lines.';
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

%
% add to  table if there are no errors 

mouseCount = size(m.new_mice,1);
duplicateCount = 0;

if ~isempty(m.other_id) && ~ischar(m.other_id)
    m.other_id = char(m.other_id);
end

if ~isempty(m.row) && ~ischar(m.row)
    m.row = char(m.row);
end

if ~isempty(m.founder_notes) && ~ischar(m.founder_notes)
    m.founder_notes = char(m.founder_notes);
end

if ~isempty(m.source) && ~ischar(m.source)
    m.source = char(m.source);
end

if ~isempty(m.new_mice)
    mouseTable = m.new_mice;
end

if isempty(errorString) && isempty(m.new_mice) 
    mouseCount = mouseCount + 1;
    mouseTable(mouseCount,:) = {m.animal_id m.other_id m.dob m.dow m.doa m.sex m.color m.ear_punch m.line m.genotype m.owner m.facility m.room m.rack m.row  m.source m.founder_notes};
    set(h.new_mice,'Data',mouseTable);
end

if isempty(errorString) && ~isempty(m.new_mice) 
    for i = 1: size(m.new_mice,1)
        if m.animal_id == m.new_mice{i,1}
            mouseTable(i,:) = {m.animal_id m.other_id m.dob m.dow m.doa m.sex m.color m.ear_punch m.line m.genotype m.owner m.facility m.room m.rack m.row  m.source m.founder_notes};
            duplicateCount = duplicateCount + 1;
            set(h.new_mice,'Data',mouseTable);
        end
    end
end

if isempty(errorString) && ~isempty(m.new_mice) && duplicateCount == 0
    mouseCount = mouseCount + 1;
    mouseTable(mouseCount,:) = {m.animal_id m.other_id m.dob m.dow m.doa m.sex m.color m.ear_punch m.line m.genotype m.owner m.facility m.room m.rack m.row  m.source m.founder_notes};
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
    else mice.GUIs.clearEntry(src);
    end
end

end