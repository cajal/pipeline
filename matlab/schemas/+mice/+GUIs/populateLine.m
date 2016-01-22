function populateLine(src,~)

figHand = get(src,'parent');

% get UI handles and data

h.line = findobj(figHand,'tag','lineField');
h.line_full = findobj(figHand,'tag','lineFullField');
h.rec_strain = findobj(figHand,'tag','recipientField');
h.donor_strain = findobj(figHand,'tag','donorField');
h.n = findobj(figHand,'tag','nField');
h.seq = findobj(figHand,'tag','seqField');
h.line_notes = findobj(figHand,'tag','notesField');
h.modify_box = findobj(figHand,'tag','modifyBox');

m.modify_box = get(h.modify_box,'value');

if m.modify_box == 1
    s = get(h.line,'string');
    v = get(h.line,'value');
    m.line = s{v};
else return
end

if strcmp(' ',m.line)
    set(h.line_full,'string','');
    set(h.rec_strain,'string','');
    set(h.donor_strain,'string','');
    set(h.n,'string','');
    set(h.seq,'string','');
    set(h.line_notes,'string','');
    return
end

s = fetch(mice.Lines & ['line="' m.line '"'],'*');

set(h.line_full,'string',s.line_full);
set(h.rec_strain,'string',s.rec_strain);
set(h.donor_strain,'string',s.donor_strain);
set(h.seq,'string',s.seq);
set(h.line_notes,'string',s.line_notes);

if num2str(s.n) == 'NaN'
    set(h.n,'string','');
else set(h.n,'string',s.n);
end

end
