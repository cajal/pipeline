function submitLine(src,~)

figHand = get(src,'parent');

% Clear previous error messages

h.errorMessage = findobj(figHand,'tag','errorMessage');
h.errorBox = findobj(figHand,'tag','errorBox');

delete(h.errorMessage);
delete(h.errorBox);

% get UI handles and data

h.line = findobj(figHand,'tag','lineField');
h.line_full = findobj(figHand,'tag','lineFullField');
h.rec_strain = findobj(figHand,'tag','recipientField');
h.donor_strain = findobj(figHand,'tag','donorField');
h.n = findobj(figHand,'tag','nField');
h.seq = findobj(figHand,'tag','seqField');
h.line_notes = findobj(figHand,'tag','notesField');
h.modify_box = findobj(figHand,'tag','modifyBox');


m.line_full = get(h.line_full,'string');
m.rec_strain = get(h.rec_strain,'string');
m.donor_strain = get(h.donor_strain,'string');
m.nstring = get(h.n,'string');
m.seq = get(h.seq,'string');
m.line_notes = get(h.line_notes,'string');
m.modify_box = get(h.modify_box,'value');

s = getEnumValues(mice.Lines.table, 'line');
s = [' ' s];

if m.modify_box == 1
    v = get(h.line,'value');
    m.line = s{v};
else m.line = get(h.line,'string');
end

% error checking

errorCount = 0;
errorString = {};

if isempty(m.line) || strcmp(' ',m.line)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Line abbreviation is required.';
end

if m.modify_box == 0 && any(strcmp(m.line,s))
    errorCount = errorCount + 1;
    errorString{errorCount} = 'This line abbreviation is already being used.';
end

if ischar(m.nstring)
    m.n = str2num(m.nstring);
end

if isempty(m.nstring)
    m.n=nan;
end

if ~isempty(m.nstring) && isempty(m.n) 
    errorCount = errorCount + 1;
    errorString{errorCount} = 'N must be numeric or left blank.';
end

if ~isempty(errorString)
    h.errorMessage = uicontrol('style','text','String',['Cannot add line due to the following errors: '], 'position', [225 462 225 32],'fontsize',14,'tag','errorMessage');
    h.errorBox = uicontrol('style','listbox','string',errorString,'tag','errorBox','position',[225 430 225 32]);
    return
end

s = fetch(mice.Lines);
for i=1:size(s,1)
    lines{i} = s(i).line;
end

schema = mice.getSchema;
schema.conn.startTransaction
if m.modify_box == 1 && any(strcmp(m.line,lines))
    update(mice.Lines & ['line="' m.line '"'],'line_full',m.line_full);
    update(mice.Lines & ['line="' m.line '"'],'rec_strain',m.rec_strain);
    update(mice.Lines & ['line="' m.line '"'],'donor_strain',m.donor_strain);
    if ~isempty(m.n)
        update(mice.Lines & ['line="' m.line '"'],'n',m.n);
    else update(mice.Lines & ['line="' m.line '"'],'n');
    end
    update(mice.Lines & ['line="' m.line '"'],'seq',m.seq);
    update(mice.Lines & ['line="' m.line '"'],'line_notes',m.line_notes);
end

fields = {'modify_box','nstring'};
lineStruct = rmfield(m,fields);

if m.modify_box == 0
    s = getEnumValues(mice.Lines.table,'line');
    s = [s m.line];
    str= ['line : enum(''' s{1} ''''];
    for i=2:length(s)
        str = [str ', ''' s{i} ''''];
    end
    str=[str ') # Mouse Line Abbreviation'];
    v = dj.set('suppressPrompt');
    dj.set('suppressPrompt',true);
    alterAttribute(mice.Lines.table,'line',str);
    alterAttribute(mice.Genotypes.table,'line',str);
    alterAttribute(mice.Founders.table,'line',str);
    makeTuples(mice.Lines,lineStruct);
    s = [' ' s];
    str = ['line1=null : enum(''' s{1} ''''];
    for i = 2:length(s)
        str = [str ', ''' s{i} ''''];
    end
    str = [str ') # Mouse Line 1 Abbreviation'];
    alterAttribute(mice.Requests.table,'line1',str);
    str = ['line2=null : enum(''' s{1} ''''];
    for i = 2:length(s)
        str = [str ', ''' s{i} ''''];
    end
    str = [str ') # Mouse Line 2 Abbreviation'];
    alterAttribute(mice.Requests.table,'line2',str);
    str = ['line3=null : enum(''' s{1} ''''];
    for i = 2:length(s)
        str = [str ', ''' s{i} ''''];
    end
    str = [str ') # Mouse Line 3 Abbreviation'];
    alterAttribute(mice.Requests.table,'line3',str);
    dj.set('suppressPrompt',v);
end
schema.conn.commitTransaction
set(h.line,'string','');
set(h.line_full,'string','');
set(h.rec_strain,'string','');
set(h.donor_strain,'string','');
set(h.n,'string','');
set(h.seq,'string','');
set(h.line_notes,'string','');
set(h.modify_box,'string','');

end

