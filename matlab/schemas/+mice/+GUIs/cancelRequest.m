function cancelRequest(src,~)

% Get UI data

figHand = get(src,'parent');

h.requestor = findobj(figHand,'tag','requestorField');
h.request_table = findobj(figHand,'tag','requestTable');

m.request_table = get(h.request_table,'Data');

s = get(h.requestor,'string');
v = get(h.requestor,'value');
m.requestor = s{v};

% clear previous error messages

h.errorMessage = findobj(figHand,'tag','errorMessage');
h.errorBox = findobj(figHand,'tag','errorBox');

delete(h.errorMessage);
delete(h.errorBox);

% Error Checking

errorCount = 0;
errorString = {};

% the selected user must have at least one active request

if isempty(m.request_table)
    errorCount = errorCount + 1;
    errorString{errorCount} = 'The selected user does not have any active requests.';
end

% There must be at least one request selected from the table

numSelected = 0;
for i = 1:size(m.request_table,1)
    if m.request_table{i,1} == true
        numSelected = numSelected + 1;
    end
end
if numSelected == 0;
    errorCount = errorCount + 1;
    errorString{errorCount} = 'Please select request(s) that you wish to cancel.';
end

% if there are errors, display them to the user

if ~isempty(errorString)
    h.errorMessage = uicontrol('style','text','String',['Cannot submit request due to the following errors: '], 'position', [50 470 300 29],'fontsize',14,'tag','errorMessage');
    h.errorBox = uicontrol('style','listbox','string',errorString,'tag','errorBox','position',[350 470 300 29]);
    return
end

% if there are no errors, modify the requested number of mice so that it is
% equal to the number already transferred (no more mice needed)

if isempty(errorString)
    for i = 1:size(m.request_table,1)
        request = fetch(mice.Requests & ['requestor="' m.requestor '"'] & ['dor="' m.request_table{i,2} '"'] & ['line1="' m.request_table{i,5} '"'] & ['genotype1="' m.request_table{i,6} '"'] & ['line2="' m.request_table{i,7} '"'] & ['genotype2="' m.request_table{i,8} '"'] & ['line3="' m.request_table{i,9} '"'] & ['genotype3="' m.request_table{i,10} '"']);
        if m.request_table{i,1} == true
            update(mice.Requests & ['request_idx=' num2str(request.request_idx)],'number_mice',m.request_table{i,4});
        end
    end
    mice.GUIs.requestorCallback(src);
end
end