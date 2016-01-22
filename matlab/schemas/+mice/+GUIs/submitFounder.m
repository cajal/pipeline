function submitFounder(src,~)

figHand = get(src,'parent');

[h,m] = mice.GUIs.getUIData(figHand);

% Generate data structures compatible with Data Joint tables

if isempty(m.new_mice)
    return
end

mouseStruct = struct('animal_id',m.new_mice(:,1),'other_id',m.new_mice(:,2),'dob',m.new_mice(:,3),'dow',m.new_mice(:,4),'sex',m.new_mice(:,6),'color',m.new_mice(:,7),'ear_punch',m.new_mice(:,8),'owner',m.new_mice(:,11),'facility',m.new_mice(:,12),'room',m.new_mice(:,13),'rack',m.new_mice(:,14),'row',m.new_mice(:,15),'mouse_notes','');
for i = 1:size(mouseStruct,1)
    if ischar(mouseStruct(i).animal_id)
        mouseStruct(i).animal_id = str2num(mouseStruct(i).animal_id);
    end
end

founderStruct = struct('animal_id',m.new_mice(:,1),'line',m.new_mice(:,9),'source',m.new_mice(:,16),'doa',m.new_mice(:,5),'founder_notes',m.new_mice(:,17));
for i = 1:size(founderStruct,1)
    if ischar(founderStruct(i).animal_id)
        founderStruct(i).animal_id = str2num(founderStruct(i).animal_id);
    end
end

genotypeStruct = struct('animal_id',m.new_mice(:,1),'line',m.new_mice(:,9),'genotype',m.new_mice(:,10),'genotype_notes','');
for i = 1:size(genotypeStruct,1)
    if ischar(genotypeStruct(i).animal_id)
        genotypeStruct(i).animal_id = str2num(genotypeStruct(i).animal_id);
    end
end

% Check for errors within table before adding mice to database

errorString = {};
errorCount = 0;

% Add error checking here is I think of them later

% if there are no errors, add mice to the database

if ~isempty(errorString)
    h.errorMessage = uicontrol('style','text','String',['Cannot add mouse due to the following errors: '], 'position', [150 760 500 16],'fontsize',14,'tag','errorMessage');
    h.errorBox = uicontrol('style','listbox','string',errorString,'tag','errorBox','position',[150 710 500 50]);
else schema = mice.getSchema;
    schema.conn.startTransaction
    for i = 1:size(mouseStruct,1)
        tuple = mouseStruct(i,:);
        if isempty(tuple.dob)
            tuple = rmfield(tuple,'dob');
        end
        if isempty(tuple.dow)
            tuple = rmfield(tuple,'dow');
        end
        makeTuples(mice.Mice,tuple);
    end
    for i = 1:size(founderStruct,1)
        tuple = founderStruct(i,:);
        if isempty(tuple.doa)
            tuple = rmfield(tuple,'doa');
        end
        makeTuples(mice.Founders,tuple);
    end
    makeTuples(mice.Genotypes,genotypeStruct);
    schema.conn.commitTransaction
    set(h.new_mice,'Data',{},'RowName','');
    mice.GUIs.clearEntry(src);
end

end