function submitTransfer(src,~)

% get ui data

figHand = get(src,'parent');

h.table = findobj(figHand,'tag','miceTable');
m.table = get(h.table,'data');

if isempty(m.table)
    return
end

h.dod = findobj(figHand,'tag','dod');
h.death_notes = findobj(figHand,'tag','deathNotes');
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

m.dod = get(h.dod,'string');
if ~isempty(m.dod)
    m.dod = datestr(m.dod,'yyyy-mm-dd');
end

m.death_notes = get(h.death_notes,'string');

% clear previous error messages

h.errorMessage = findobj(figHand,'tag','errorMessage');
h.errorBox = findobj(figHand,'tag','errorBox');

delete(h.errorMessage);
delete(h.errorBox);

% error checking

errorCount = 0;
errorString = {};

% DOD is a required field
if isempty(m.dod)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'DOD is required.';
end

if ~isempty(errorString)
    h.errorMessage = uicontrol('style','text','String',['Cannot submit due to the following errors: '], 'position', [160 470 300 29],'fontsize',14,'tag','errorMessage');
    h.errorBox = uicontrol('style','listbox','string',errorString,'tag','errorBox','position',[460 470 300 29]);
    return
end

% If no errors, add DOD's to database

for i = 1:size(m.table,1)
    deathStruct(i).animal_id = str2num(m.table{i,1});
    deathStruct(i).dod = m.dod;
    if ~isempty(m.death_notes)
        deathStruct(i).death_notes = m.death_notes;
    end
end

if isempty(errorString)
    makeTuples(mice.Death,deathStruct);
    set(h.table,'Data',{},'RowName','');
    set(h.dod,'string','');
    set(h.death_notes,'string','');
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

