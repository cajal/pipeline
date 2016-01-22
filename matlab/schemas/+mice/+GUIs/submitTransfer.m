function submitTransfer(src,~)

% get ui data

figHand = get(src,'parent');

h.table = findobj(figHand,'tag','miceTable');
m.table = get(h.table,'data');

if isempty(m.table)
    return
end

h.owner = findobj(figHand,'tag','owner');
h.facility = findobj(figHand,'tag','facility');
h.room = findobj(figHand,'tag','room');
h.rack = findobj(figHand,'tag','rack');
h.row = findobj(figHand,'tag','row');
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
h.dot = findobj(figHand,'tag','transferDate');


v = get(h.owner,'value');
s = get(h.owner,'string');
m.owner = s{v};

v = get(h.facility,'value');
s = get(h.facility,'string');
m.facility = s{v};

v = get(h.room,'value');
s = get(h.room,'string');
m.room = s{v};

m.rack = get(h.rack,'string');

m.row = get(h.row,'string');
m.dot = get(h.dot,'string');

% clear previous error messages

h.errorMessage = findobj(figHand,'tag','errorMessage');
h.errorBox = findobj(figHand,'tag','errorBox');

delete(h.errorMessage);
delete(h.errorBox);

% error checking

errorCount = 0;
errorString = {};

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

% Owner, facility and room and tansfer date are required

if isempty(m.owner)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Owner is required';
end

if isempty(m.facility)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Facility is required';
end

if isempty(m.room)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Room is required';
end

if isempty(m.dot)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Transfer date is required';
end

% transfer date must be a date

formatOut = 'yyyy-mm-dd';
if ~isempty(m.dot)
    try 
        m.dot = datestr(m.dot,formatOut);
    catch
        errorCount = errorCount + 1;
        errorString{errorCount} = 'Date of transfer cannot be interpreted.';
    end
end

% if there are  errors, display them to the user

if ~isempty(errorString)
    h.errorMessage = uicontrol('style','text','String',['Cannot submit transfer due to the following errors: '], 'position', [160 470 300 29],'fontsize',14,'tag','errorMessage');
    h.errorBox = uicontrol('style','listbox','string',errorString,'tag','errorBox','position',[460 470 300 29]);
    return
end

% if there are no errors, update info in database and add to transfer table

if isempty(errorString)
    schema = mice.getSchema;
    schema.conn.startTransaction
    for i = 1:size(m.table,1)
        from = fetch(mice.Mice & ['animal_id=' m.table{i,1}],'*');
        update(mice.Mice & ['animal_id=' m.table{i,1}],'owner',m.owner)
        update(mice.Mice & ['animal_id=' m.table{i,1}],'facility',m.facility)
        update(mice.Mice & ['animal_id=' m.table{i,1}],'room',m.room)
        update(mice.Mice & ['animal_id=' m.table{i,1}],'rack',m.rack)
        update(mice.Mice & ['animal_id=' m.table{i,1}],'row',m.row)
        newTransfer = struct('animal_id',str2num(m.table{i,1}),'dot',m.dot,'from_owner',from.owner,'to_owner',m.owner,'from_facility',from.facility,'to_facility',m.facility,'from_room',from.room,'to_room',m.room,'from_rack',from.rack,'to_rack',m.rack,'from_row',from.row,'to_row',m.row);
        if isempty(from.owner)
            newTransfer = rmfield(newTransfer,'from_owner');
        end
        if isempty(from.facility)
            newTransfer = rmfield(newTransfer,'from_facility');
        end
        if isempty(from.room)
            newTransfer = rmfield(newTransfer,'from_room');
        end
        if isempty(from.rack)
            newTransfer = rmfield(newTransfer,'from_rack');
        end
        if isempty(from.row)
            newTransfer = rmfield(newTransfer,'from_row');
        end
        if isempty(m.rack)
            newTransfer = rmfield(newTransfer,'to_rack');
        end
        if isempty(m.row)
            newTransfer = rmfield(newTransfer,'to_row');
        end
        makeTuples(mice.Transfers,newTransfer);
    end
    schema.conn.commitTransaction
    set(h.table,'Data',{},'RowName','');
    set(h.owner,'value',1);
    set(h.facility,'value',1);
    set(h.room,'value',1);
    set(h.rack,'string','');
    set(h.row,'string','');
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
    set(h.dot,'string','');
    url = 'http://www.ccmbioinfo.bcm.tmc.edu/transfer/Login.asp';
    web(url,'-new','-browser')
end

end