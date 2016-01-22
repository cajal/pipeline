function minusCallback(src,~)

figHand = get(src,'parent');

[h,m] = mice.GUIs.getUIData(figHand);

if ~isempty(m.new_mice) 
    mouseTable = m.new_mice;
    for i = 1: size(m.new_mice,1)
        if strcmp(m.animal_id, m.new_mice{i,1})
            mouseTable(i,:) = [];
        end
    end
    set(h.new_mice,'Data',mouseTable)
end

% Clear previous error messages

h.errorMessage = findobj(figHand,'tag','errorMessage');
h.errorBox = findobj(figHand,'tag','errorBox');

delete(h.errorMessage);
delete(h.errorBox);

end