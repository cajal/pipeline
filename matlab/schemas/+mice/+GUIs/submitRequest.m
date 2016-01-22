function submitRequest(src,~)

% Get UI data

figHand = get(src,'parent');

h.requestor = findobj(figHand,'tag','requestorField');
h.dor = findobj(figHand,'tag','dor');
h.number_mice = findobj(figHand,'tag','numberMice');
h.age = findobj(figHand,'tag','ageField');
h.request_notes = findobj(figHand,'tag','requestNotes');
h.line1 = findobj(figHand,'tag','line1Field');
h.line2 = findobj(figHand,'tag','line2Field');
h.line3 = findobj(figHand,'tag','line3Field');
h.genotype1 = findobj(figHand,'tag','genotype1Field');
h.genotype2 = findobj(figHand,'tag','genotype2Field');
h.genotype3 = findobj(figHand,'tag','genotype3Field');
h.request_table = findobj(figHand,'tag','requestTable');

m.dor = get(h.dor,'string');
m.number_mice = get(h.number_mice,'string');
m.request_notes = get(h.request_notes,'string');

s = get(h.requestor,'string');
v = get(h.requestor,'value');
m.requestor = s{v};

s = get(h.age,'string');
v = get(h.age,'value');
m.age = s{v};

s = get(h.line1,'string');
v = get(h.line1,'value');
m.line1 = s{v};

s = get(h.line2,'string');
v = get(h.line2,'value');
m.line2 = s{v};

s = get(h.line3,'string');
v = get(h.line3,'value');
m.line3 = s{v};

s = get(h.genotype1,'string');
v = get(h.genotype1,'value');
m.genotype1 = s{v};

s = get(h.genotype2,'string');
v = get(h.genotype2,'value');
m.genotype2 = s{v};

s = get(h.genotype3,'string');
v = get(h.genotype3,'value');
m.genotype3 = s{v};

% clear previous error messages

h.errorMessage = findobj(figHand,'tag','errorMessage');
h.errorBox = findobj(figHand,'tag','errorBox');

delete(h.errorMessage);
delete(h.errorBox);

% Error Checking

errorCount = 0;
errorString = {};

% requestor, date, number of mice, line1 and genotype 1 are required

if isempty(m.requestor)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'The requestor field is required.';
end

if isempty(m.dor)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'The date of request is a required field.';
end

if isempty(m.number_mice)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'The number of mice  is a required field.';
end

if isempty(m.line1)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Line 1 is a required field.';
end

if isempty(m.genotype1)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Genotype 1 is a required field.';
end

% enter line 1 before line 2, before line 3

if ~isempty(m.line2) && isempty(m.line1)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Please enter line 1 before line 2.';
end

if ~isempty(m.line3) && (isempty(m.line1) || isempty(m.line2))
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Please enter lines 1 and 2 before line 3.';
end

% if a line is listed it must have a matching genotype

if ~isempty(m.line1) && isempty(m.genotype1)
    errorCount = errorCount + 1;
    errorString{errorCount} = ['Please select a genotype for ' m.line1 '.'];
end

if ~isempty(m.line2) && isempty(m.genotype2)
    errorCount = errorCount + 1;
    errorString{errorCount} = ['Please select a genotype for ' m.line2 '.'];
end

if ~isempty(m.line3) && isempty(m.genotype3)
    errorCount = errorCount + 1;
    errorString{errorCount} = ['Please select a genotype for ' m.line3 '.'];
end

% if a genotype is listed, it must have a matching line

if isempty(m.line1) && ~isempty(m.genotype1)
    errorCount = errorCount + 1;
    errorString{errorCount} = ['Please select Line 1 or clear Genotype 1.'];
end

if isempty(m.line2) && ~isempty(m.genotype2)
    errorCount = errorCount + 1;
    errorString{errorCount} = ['Please select Line 2 or clear Genotype 2.'];
end

if isempty(m.line3) && ~isempty(m.genotype3)
    errorCount = errorCount + 1;
    errorString{errorCount} = ['Please select Line 3 or clear Genotype 3.'];
end

%Date of request must be formatted correctly

formatOut = 'yyyy-mm-dd';

if ~ischar(m.dor)
    m.dor = char(m.dor);
end

if ~isempty(m.dor)
    try 
        m.dor = datestr(m.dor,formatOut);
    catch
        errorCount = errorCount + 1;
        errorString{errorCount} = 'Date of birth cannot be interpreted.';
    end
end

% if there are errors, display them to the user

if ~isempty(errorString)
    h.errorMessage = uicontrol('style','text','String',['Cannot submit request due to the following errors: '], 'position', [50 470 300 29],'fontsize',14,'tag','errorMessage');
    h.errorBox = uicontrol('style','listbox','string',errorString,'tag','errorBox','position',[350 470 300 29]);
    return
end

% if there are no errors, add the request to the database and clear
% everything

if isempty(m.age)
    m.age
end

if isempty(errorString)
    a = fetch(mice.Requests);
    request_number = (size(a,1) + 1);
    newRequest = struct('request_idx',request_number,'requestor',m.requestor,'dor',m.dor,'number_mice',str2num(m.number_mice),'age',m.age,'line1',m.line1,'genotype1',m.genotype1,'line2',m.line2,'genotype2',m.genotype2,'line3',m.genotype3,'request_notes',m.request_notes);
    if isempty(m.age)
        fields = 'age';
        newRequest = rmfield(newRequest,fields)
    end
    makeTuples(mice.Requests,newRequest);
    set(h.dor,'string','');
    set(h.number_mice,'string','');
    set(h.age,'value',1);
    set(h.request_notes,'string','');
    set(h.line1,'value',1);
    set(h.line2,'value',1);
    set(h.line3,'value',1);
    set(h.genotype1,'value',1);
    set(h.genotype2,'value',1);
    set(h.genotype3,'value',1);
    mice.GUIs.requestorCallback(src);
end