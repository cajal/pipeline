function requestorCallback(src,~)

figHand = get(src,'parent');

h.requestor = findobj(figHand,'tag','requestorField');
h.request_table = findobj(figHand,'tag','requestTable');

s = get(h.requestor,'string');
v = get(h.requestor,'value');
m.requestor = s{v};

if strcmp('',m.requestor)
    set(h.request_table,'Data','');
    return
end

requests = fetch(mice.Requests & ['requestor="' m.requestor '"'],'*');
transfers = fetch(mice.Transfers & ['to_owner="' m.requestor '"']);
transferID = {};
for i = 1:size(transfers,1)
    transferID{i} = transfers(i).animal_id;
end

requestTable = {};
row = 0;
for i = 1:size(requests,1)
    transferCount = 0;
    id = {};
    for j = 1:length(transferID)
        if ~isempty(transferID{j})
            match = 1;
            if ~isempty(requests(i).line1)
                a = fetch(mice.Genotypes & ['animal_id=' num2str(transferID{j})] & ['line="' requests(i).line1 '"'] & ['genotype="' requests(i).genotype1 '"']);
                if isempty(a) && strcmp(requests(i).genotype1,'heterozygous')
                    a = fetch(mice.Genotypes & ['animal_id=' num2str(transferID{j})] & ['line="' requests(i).line1 '"'] & ['genotype="positive"']);
                end
                if isempty(a) && strcmp(requests(i).genotype1,'positive')
                    a = fetch(mice.Genotypes & ['animal_id=' num2str(transferID{j})] & ['line="' requests(i).line1 '"'] & ['genotype="heterozygous"']);
                    if isempty(a)
                        a = fetch(mice.Genotypes & ['animal_id=' num2str(transferID{j})] & ['line="' requests(i).line1 '"'] & ['genotype="homozygous"']);
                    end
                end
                if isempty(a)
                    match = 0;
                end 
            end
            if ~isempty(requests(i).line2)
                a = fetch(mice.Genotypes & ['animal_id=' num2str(transferID{j})] & ['line="' requests(i).line2 '"'] & ['genotype="' requests(i).genotype2 '"']);
                if isempty(a) && strcmp(requests(i).genotype2,'heterozygous')
                    a = fetch(mice.Genotypes & ['animal_id=' num2str(transferID{j})] & ['line="' requests(i).line1 '"'] & ['genotype="positive"']);
                end
                if isempty(a) && strcmp(requests(i).genotype2,'positive')
                    a = fetch(mice.Genotypes & ['animal_id=' num2str(transferID{j})] & ['line="' requests(i).line1 '"'] & ['genotype="heterozygous"']);
                    if isempty(a)
                        a = fetch(mice.Genotypes & ['animal_id=' num2str(transferID{j})] & ['line="' requests(i).line1 '"'] & ['genotype="homozygous"']);
                    end
                end
                if isempty(a)
                    match = 0;
                end
            end
            if ~isempty(requests(i).line3)
                a = fetch(mice.Genotypes & ['animal_id=' num2str(transferID{j})] & ['line="' requests(i).line3 '"'] & ['genotype="' requests(i).genotype3 '"']);
                if isempty(a) && strcmp(requests(i).genotype3,'heterozygous')
                    a = fetch(mice.Genotypes & ['animal_id=' num2str(transferID{j})] & ['line="' requests(i).line1 '"'] & ['genotype="positive"']);
                end
                if isempty(a) && strcmp(requests(i).genotype3,'positive')
                    a = fetch(mice.Genotypes & ['animal_id=' num2str(transferID{j})] & ['line="' requests(i).line1 '"'] & ['genotype="heterozygous"']);
                    if isempty(a)
                        a = fetch(mice.Genotypes & ['animal_id=' num2str(transferID{j})] & ['line="' requests(i).line1 '"'] & ['genotype="homozygous"']);
                    end
                end
                if isempty(a)
                    match = 0;
                end
            end
            a = fetch(mice.Genotypes & ['animal_id=' num2str(transferID{j})],'*');
            for k = 1:size(a,1)
                if ~strcmp(a(k).line,requests(i).line1) && ~strcmp(a(k).line,requests(i).line2) && ~strcmp(a(k).line,requests(i).line3) && ~strcmp(a(k).genotype,'negative')
                        match = 0;
                end
            end
            if match == 1 && datenum(requests(i).dor) < datenum(transfers(j).dot)
                if transferCount < requests(i).number_mice
                    transferCount = transferCount + 1;
                    id(transferCount) = transferID(j);
                    transferID{j} = [];
                end
            end
        end
    end
    count(i) = transferCount;
    idx{i} = id(:);
    if requests(i).number_mice ~= count(i)
        row = row + 1;
        requestTable{row,1} = false;
        requestTable{row,2} = requests(i).dor;
        requestTable{row,3} = requests(i).number_mice;
        requestTable{row,4} = count(i);
        requestTable{row,5} = requests(i).line1;
        requestTable{row,6} = requests(i).genotype1;
        requestTable{row,7} = requests(i).line2;
        requestTable{row,8} = requests(i).genotype2;
        requestTable{row,9} = requests(i).line3;
        requestTable{row,10} = requests(i).genotype3;
    end
end
set(h.request_table,'Data',requestTable);

% clear previous error messages

h.errorMessage = findobj(figHand,'tag','errorMessage');
h.errorBox = findobj(figHand,'tag','errorBox');

delete(h.errorMessage);
delete(h.errorBox);

end