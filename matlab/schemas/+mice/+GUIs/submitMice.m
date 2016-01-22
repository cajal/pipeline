function submitMice(src,~)

figHand = get(src,'parent');

[h,m] = mice.GUIs.getUIData(figHand);

% Clear previous error messages

h.errorMessage = findobj(figHand,'tag','errorMessage');
h.errorBox = findobj(figHand,'tag','errorBox');

delete(h.errorMessage);
delete(h.errorBox);


% Generate data structures compatible with Data Joint tables

genotypeCount = 0;
parentCount = 0;

if isempty(m.new_mice)
    return
end

s = struct('animal_id',m.new_mice(:,1),'other_id',m.new_mice(:,2),'dob',m.new_mice(:,3),'dow',m.new_mice(:,4),'parent1',m.new_mice(:,5),'parent2',m.new_mice(:,6),'parent3',m.new_mice(:,7),'sex',m.new_mice(:,8),'color',m.new_mice(:,9),'ear_punch',m.new_mice(:,10),'line1',m.new_mice(:,11),'genotype1',m.new_mice(:,12),'line2',m.new_mice(:,13),'genotype2',m.new_mice(:,14),'line3',m.new_mice(:,15),'genotype3',m.new_mice(:,16),'owner',m.new_mice(:,17),'facility',m.new_mice(:,18),'room',m.new_mice(:,19),'rack',m.new_mice(:,20),'row',m.new_mice(:,21),'mouse_notes',m.new_mice(:,22));
for i = 1:size(s,1)
    if ischar(s(i).animal_id)
        s(i).animal_id = str2num(s(i).animal_id);
    end
end

fields = {'parent1','parent2','parent3','line1','genotype1','line2','genotype2','line3','genotype3'};
mouseStruct = rmfield(s,fields);

parentStruct = {};
genotypeStruct = {};
for i = 1:size(s,1)
    if ~isempty(s(i).line1)
        genotypeCount = genotypeCount + 1;
        genotypeStruct(genotypeCount).animal_id = s(i).animal_id;
        genotypeStruct(genotypeCount).line = s(i).line1;
        genotypeStruct(genotypeCount).genotype = s(i).genotype1;
    end
    if ~isempty(s(i).line2)
        genotypeCount = genotypeCount + 1;
        genotypeStruct(genotypeCount).animal_id = s(i).animal_id;
        genotypeStruct(genotypeCount).line = s(i).line2;
        genotypeStruct(genotypeCount).genotype = s(i).genotype2;
    end
    if ~isempty(s(i).line3)
        genotypeCount = genotypeCount + 1;
        genotypeStruct(genotypeCount).animal_id = s(i).animal_id;
        genotypeStruct(genotypeCount).line = s(i).line3;
        genotypeStruct(genotypeCount).genotype = s(i).genotype3;
    end
    if ~isempty(s(i).parent1)
        parentCount = parentCount + 1;
        parentStruct(parentCount).animal_id = s(i).animal_id;
        parentStruct(parentCount).parent_id = s(i).parent1;
    end
    if ~isempty(s(i).parent2)
        parentCount = parentCount + 1;
        parentStruct(parentCount).animal_id = s(i).animal_id;
        parentStruct(parentCount).parent_id = s(i).parent2;
    end
    if ~isempty(s(i).parent3)
        parentCount = parentCount + 1;
        parentStruct(parentCount).animal_id = s(i).animal_id;
        parentStruct(parentCount).parent_id = s(i).parent3;
    end
end

% Check for errors within table before adding mice to database

errorString = {};
errorCount = 0;

mouse_id = m.new_mice(:,1);

if ~(length(unique(mouse_id)) == length(mouse_id))
    errorCount = errorCount + 1;
    errorString{errorCount} = 'A Mouse ID was entered twice.';
end

parents = {};
if ~isempty(parentStruct)
    parents = {parentStruct.parent_id};
end

for i = 1:length(mouse_id)
if ~isempty(parents) && any(strcmp(mouse_id{i},parents))
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Parents and children must be entered separately to allow for full error checking.';
end
end

other_id = {s.other_id};
for i = 1:length(other_id)
    if ~isempty(parents) && any(strcmp(other_id{i},parents))
        errorCount = errorCount + 1;
        errorString{errorCount} = 'Parents and children must be entered separately to allow for full error checking.';
    end
end

% if there are no errors, add mice to the database
if ~isempty(errorString)
    h.errorMessage = uicontrol('style','text','String',['Cannot add mouse due to the following errors: '], 'position', [300 569 500 16],'fontsize',14,'tag','errorMessage');
    h.errorBox = uicontrol('style','listbox','string',errorString,'tag','errorBox','position',[300 519 500 50]);
else schema = mice.getSchema;
    schema.conn.startTransaction
    for i = 1:size(mouseStruct,1)
        tuple = mouseStruct(i,:);
        if isempty(tuple.dow)
            tuple = rmfield(tuple,'dow');
        end
        makeTuples(mice.Mice,tuple);
    end
    makeTuples(mice.Parents,parentStruct);
    makeTuples(mice.Genotypes,genotypeStruct);
    schema.conn.commitTransaction
    set(h.new_mice,'Data',{},'RowName','');
    mice.GUIs.clearEntry(src);
end

end