function printList(src,~)

figHand = get(src,'parent');

[h,m] = mice.GUIs.getUIData(figHand);

isempty(h.new_mice)

cnames = get(h.new_mice,'columnname');
cnames = cnames.';
data = cell(size(m.new_mice,1)+1,size(m.new_mice,2));
data(1,:) = cnames;
for i = 1:size(m.new_mice,1)
    data((i+1),:) = m.new_mice(i,:);
end

data = cell2dataset(data);
a = uigetdir;
export(data,'File',[a '/printMice.csv'],'delimiter',',');

end