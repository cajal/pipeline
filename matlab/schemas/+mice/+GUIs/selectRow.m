function selectRow(src,event)

if isempty(event.Indices)
    return
end

row = event.Indices(1);
rnames = get(src,'RowName');
dim = size(rnames,1);
rnames = cell(dim,1);
rnames{row} = '>';
set(src,'RowName',rnames);

figHand = get(src,'parent');

if ~isempty(findobj(figHand,'tag','rangeEnd'))
    return
end

[h,m] = mice.GUIs.getUIData(figHand);

set(h.animal_id,'string',m.new_mice{row,1});
set(h.other_id,'string',m.new_mice{row,2});
set(h.dob,'string',m.new_mice{row,3});
set(h.dow,'string',m.new_mice{row,4});



if ~isempty(h.isMice)
    set(h.rack,'string',m.new_mice{row,20});
    set(h.row,'string',m.new_mice{row,21});
    s = get(h.sex,'string');
    v = find(strcmp(m.new_mice{row,8},s));
    set(h.sex,'value',v);
    s = get(h.color,'string');
    v = find(strcmp(m.new_mice{row,9},s));
    set(h.color,'value',v);
    s = get(h.ear_punch,'string');
    v = find(strcmp(m.new_mice{row,10},s));
    set(h.ear_punch,'value',v);
    set(h.parent1,'string',m.new_mice{row,5});
    set(h.parent2,'string',m.new_mice{row,6});
    set(h.parent3,'string',m.new_mice{row,7});
    set(h.mouse_notes,'string',m.new_mice{row,22});
    s = get(h.line1,'string');
    v = find(strcmp(m.new_mice{row,11},s));
    set(h.line1,'value',v);
    s = get(h.genotype1,'string');
    v = find(strcmp(m.new_mice{row,12},s));
    set(h.genotype1,'value',v);
    s = get(h.line2,'string');
    v = find(strcmp(m.new_mice{row,13},s));
    set(h.line2,'value',v);
    s = get(h.genotype2,'string');
    v = find(strcmp(m.new_mice{row,14},s));
    set(h.genotype2,'value',v);
    s = get(h.line3,'string');
    v = find(strcmp(m.new_mice{row,15},s));
    set(h.line3,'value',v);
    s = get(h.genotype3,'string');
    v = find(strcmp(m.new_mice{row,16},s));
    set(h.genotype3,'value',v);
    s = get(h.owner,'string');
    v = find(strcmp(m.new_mice{row,17},s));
    set(h.owner,'value',v);
    s = get(h.facility,'string');
    v = find(strcmp(m.new_mice{row,18},s));
    set(h.facility,'value',v);
    s = get(h.room,'string');
    v = find(strcmp(m.new_mice{row,19},s));
    set(h.room,'value',v);

end

if ~isempty(h.isFounder)
    set(h.rack,'string',m.new_mice{row,14});
    set(h.row,'string',m.new_mice{row,15});
    set(h.source,'string',m.new_mice{row,16});
    set(h.mouse_notes,'string',m.new_mice{row,17});
    set(h.doa,'string',m.new_mice{row,5});
    s = get(h.sex,'string');
    v = find(strcmp(m.new_mice{row,6},s));
    set(h.sex,'value',v);
    s = get(h.color,'string');
    v = find(strcmp(m.new_mice{row,7},s));
    set(h.color,'value',v);
    s = get(h.ear_punch,'string');
    v = find(strcmp(m.new_mice{row,8},s));
    set(h.ear_punch,'value',v);
    s = get(h.line1,'string');
    v = find(strcmp(m.new_mice{row,9},s));
    set(h.line1,'value',v);
    s = get(h.genotype1,'string');
    v = find(strcmp(m.new_mice{row,10},s));
    set(h.genotype1,'value',v);
    s = get(h.owner,'string');
    v = find(strcmp(m.new_mice{row,11},s));
    set(h.owner,'value',v);
    s = get(h.facility,'string');
    v = find(strcmp(m.new_mice{row,12},s));
    set(h.facility,'value',v);
    s = get(h.room,'string');
    v = find(strcmp(m.new_mice{row,13},s));
    set(h.room,'value',v);
end

end
