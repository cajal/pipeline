function submitUpdateMice(src,~)

figHand = get(src,'parent');

h.table = findobj(figHand,'tag','miceTable');
m.table = get(h.table,'data');
h.animal_id1 = findobj(figHand,'tag','animalID1');
h.animal_id2 = findobj(figHand,'tag','animalID2');
h.animal_id3 = findobj(figHand,'tag','animalID3');
h.animal_id4 = findobj(figHand,'tag','animalID4');
h.animal_id5 = findobj(figHand,'tag','animalID5');
h.animal_id6 = findobj(figHand,'tag','animalID6');
h.animal_id7 = findobj(figHand,'tag','animalID7');
h.animal_id8 = findobj(figHand,'tag','animalID8');
h.animal_id9 = findobj(figHand,'tag','animalID9');
h.animal_id10 = findobj(figHand,'tag','animalID10');
h.animal_id11 = findobj(figHand,'tag','animalID11');
h.animal_id12 = findobj(figHand,'tag','animalID12');
h.animal_id13 = findobj(figHand,'tag','animalID13');
h.range_start = findobj(figHand,'tag','rangeStart');
h.range_end = findobj(figHand,'tag','rangeEnd');

h.errorMessage = findobj(figHand,'tag','errorMessage');
h.errorBox = findobj(figHand,'tag','errorBox');

delete(h.errorMessage);
delete(h.errorBox);

if isempty(m.table)
    return
end

% error checking

errorCount = 0;
errorString = {};

% if line 1 is C57Bl/6 or FVB the genotype miust be wild type

for i = 1:size(m.table,1)
    if (strcmp('C57Bl/6',m.table(i,7)) || strcmp('Fvb',m.table(i,7))) && ~strcmp('wild type',m.table(i,8))
        errorCount = errorCount + 1;
        errorString{errorCount} = 'Lines C57Bl/6 and Fvb should only be used to designate pure wild type mice.';
    end
end

% wild type genotype can only be used for C57Bl/6 or Fvb lines

for i = 1:size(m.table,1)
    if strcmp('wild type',m.table(i,8)) && ~(strcmp('C57Bl/6',m.table(i,7)) || strcmp('Fvb',m.table(i,7)))
        errorCount = errorCount + 1;
        errorString{errorCount} = 'The wild type genotype should only be used to describe pure C57Bl/6 or Fvb lines.';
    end
    if strcmp('wild type',m.table(i,10)) && ~(strcmp('C57Bl/6',m.table(i,9)) || strcmp('Fvb',m.table(i,9)))
        errorCount = errorCount + 1;
        errorString{errorCount} = 'The wild type genotype should only be used to describe pure C57Bl/6 or Fvb lines.';
    end
    if strcmp('wild type',m.table(i,12)) && ~(strcmp('C57Bl/6',m.table(i,11)) || strcmp('Fvb',m.table(i,11)))
        errorCount = errorCount + 1;
        errorString{errorCount} = 'The wild type genotype should only be used to describe pure C57Bl/6 or Fvb lines.';
    end
end

% if a parent is homozygous for a transgene, the offspring cannot be
% negative for that transgene

for i = 1:size(m.table,1)
    parents = fetch(mice.Parents & ['animal_id=' m.table{i,1}]);
    for j = 1:size(parents,1)
        if isempty(str2num(parents(j).parent_id))
            parents(j).parent_id = fetch(mice.Mice & ['other_id="' parents(j).parent_id '"']);
            try parents(j).parent_id = num2str(parents(j).parent_id.animal_id);
            end
        end
        if ~isempty(parents(j).parent_id)
        homo_lines = fetch(mice.Genotypes & ['animal_id=' parents(j).parent_id] & 'genotype="homozygous"','line');
        for k = 1:size(homo_lines,1)
            if strcmp(m.table(i,7),homo_lines(k).line) && strcmp('negative',m.table(i,8))
                errorCount = errorCount + 1;
                errorString{errorCount} = ['Animal ' m.table{i,1} ' cannot be negative for ' homo_lines(k).line ' because parent ' parents(j).parent_id ' is homozygous.'];
            end
            if strcmp(m.table(i,9),homo_lines(k).line) && strcmp('negative',m.table(i,10))
                errorCount = errorCount + 1;
                errorString{errorCount} = ['Animal ' m.table{i,1} ' cannot be negative for ' homo_lines(k).line ' because parent ' parents(j).parent_id ' is homozygous.'];
            end
            if strcmp(m.table(i,11),homo_lines(k).line) && strcmp('negative',m.table(i,12))
                errorCount = errorCount + 1;
                errorString{errorCount} = ['Animal ' m.table{i,1} ' cannot be negative for ' homo_lines(k).line ' because parent ' parents(j).parent_id ' is homozygous.'];
            end
        end
        end
    end
end

% Format dates

formatOut = 'yyyy-mm-dd';
for i = 1:size(m.table,1)
    dob = m.table{i,2};
    dow = m.table{i,3};
    if ~ischar(dob)
        dob = char(dob);
    end
    if ~ischar(dow)
        dow = char(dow);
    end
    if ~isempty(dob)
        try 
            dob = datestr(dob,formatOut);
            m.table{i,2} = dob;
        catch
            errorCount = errorCount + 1;
            errorString{errorCount} = ['Date of birth for ' m.table{i,1} ' cannot be interpreted.'];
        end
    end
    if ~isempty(dow) 
        try 
            dow = datestr(dow,formatOut);
            m.table{i,3} = dow;
        catch
            errorCount = errorCount + 1;
            errorString{errorCount} = ['Date of weaning for ' m.table{i,1} ' cannot be interpreted.'];
        end
    end
end

% display error box

if ~isempty(errorString)
    h.errorMessage = uicontrol('style','text','String',['Cannot update genotypes due to the following errors: '], 'position', [160 370 250 29],'fontsize',14,'tag','errorMessage');
    h.errorBox = uicontrol('style','listbox','string',errorString,'tag','errorBox','position',[410 370 400 29]);
    return
end

% if there are no errors, update genotypes in database

if isempty(errorString)
    schema = mice.getSchema;
    schema.conn.startTransaction
    for i = 1:size(m.table,1)
        if ~isempty(m.table{i,2})
            update(mice.Mice & ['animal_id=' m.table{i,1}],'dob',m.table{i,2})
        end
        if ~isempty(m.table{i,3})
            update(mice.Mice & ['animal_id=' m.table{i,1}],'dow',m.table{i,3})
        end
        update(mice.Mice & ['animal_id=' m.table{i,1}],'sex',m.table{i,4})
        update(mice.Mice & ['animal_id=' m.table{i,1}],'color',m.table{i,5})
        update(mice.Mice & ['animal_id=' m.table{i,1}],'ear_punch',m.table{i,6})
        update(mice.Mice & ['animal_id=' m.table{i,1}],'mouse_notes',m.table{i,13})
        if ~isempty(m.table{i,7})
            update(mice.Genotypes & ['animal_id=' m.table{i,1}] & ['line="' m.table{i,7} '"'],'genotype',m.table{i,8})
        end
        if ~isempty(m.table{i,9})
            update(mice.Genotypes & ['animal_id=' m.table{i,1}] & ['line="' m.table{i,9} '"'],'genotype',m.table{i,10})
        end
        if ~isempty(m.table{i,11})
            update(mice.Genotypes & ['animal_id=' m.table{i,1}] & ['line="' m.table{i,11} '"'],'genotype',m.table{i,12})
        end
    end
    schema.conn.commitTransaction
    set(h.table,'Data',{},'RowName',' ');
    set(h.animal_id1,'string','');
    set(h.animal_id2,'string','');
    set(h.animal_id3,'string','');
    set(h.animal_id4,'string','');
    set(h.animal_id5,'string','');
    set(h.animal_id6,'string','');
    set(h.animal_id7,'string','');
    set(h.animal_id8,'string','');
    set(h.animal_id9,'string','');
    set(h.animal_id10,'string','');
    set(h.animal_id11,'string','');
    set(h.animal_id12,'string','');
    set(h.animal_id13,'string','');
    set(h.range_start,'string','');
    set(h.range_end,'string','');
end
    
end