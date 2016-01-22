function clearEntry(src,~)

figHand = get(src,'parent');

[h,m] = mice.GUIs.getUIData(figHand);
h.range_start = findobj(figHand,'tag','rangeStart');
h.range_end = findobj(figHand,'tag','rangeEnd');

set(h.animal_id,'String','');
set(h.other_id,'String','');
set(h.dob,'String','');
set(h.dow,'String','');
set(h.parent1,'String','');
set(h.parent2,'String','');
set(h.parent3,'String','');
set(h.rack,'String','');
set(h.row,'String','');
set(h.mouse_notes,'String','');
set(h.doa,'String','');
set(h.source,'String','');

if ~isempty(h.isViewMice)
    set(h.sex,'Value',1);
    set(h.color,'Value',1);
    set(h.ear_punch,'Value',1);
    set(h.owner,'Value',1);
    set(h.facility,'Value',1);
    set(h.room,'Value',1);
    set(h.line1,'Value',1);
    set(h.line2,'Value',1);
    set(h.line3,'Value',1);
    set(h.genotype1,'Value',1);
    set(h.genotype2,'Value',1);
    set(h.genotype3,'Value',1);
    set(h.range_start,'String','');
    set(h.range_end,'String','');
    return
end

set(h.sex,'Value',3);
set(h.color,'Value',4);
set(h.ear_punch,'Value',1);
set(h.owner,'Value',13);
set(h.facility,'Value',4);
set(h.room,'Value',6);
set(h.line1,'Value',1);
set(h.line2,'Value',1);
set(h.line3,'Value',1);
set(h.genotype1,'Value',7);
set(h.genotype2,'Value',7);
set(h.genotype3,'Value',7);

end
