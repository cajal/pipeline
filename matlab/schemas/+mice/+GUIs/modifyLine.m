function modifyLine(src,~)

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
m.n = get(h.n,'string');
m.seq = get(h.seq,'string');
m.line_notes = get(h.line_notes,'string');
m.modify_box = get(h.modify_box,'value');

if m.modify_box == 1
    s = getEnumValues(mice.Lines.table,'line');
    s = [' ' s];
    set(h.line,'style','popupmenu','string',s,'Callback',@mice.GUIs.populateLine);
end

if m.modify_box == 0
    set(h.line,'style','edit','string','','Callback','','Value',1);
    set(h.line_full,'string','');
    set(h.rec_strain,'string','');
    set(h.donor_strain,'string','');
    set(h.n,'string','');
    set(h.seq,'string','');
    set(h.line_notes,'string','');
end

end