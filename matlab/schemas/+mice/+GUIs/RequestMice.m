f = figure;
set(f, 'position',[0 0 706 500])

uicontrol('style','text','string','Requested By:','position',[50 420 110 29],'fontunits','normalized','fontsize',.4,'HorizontalAlignment','Right');
uicontrol('style','text','string','Date:','position',[50 385 110 29],'fontunits','normalized','fontsize',.4,'HorizontalAlignment','Right');
uicontrol('style','text','string','Number of Mice:','position',[50 350 110 29],'fontunits','normalized','fontsize',.4,'HorizontalAlignment','Right');
uicontrol('style','text','string','Age Range:','position',[50 315 110 29],'fontunits','normalized','fontsize',.4,'HorizontalAlignment','Right');
uicontrol('style','text','string','Comments:','position',[50 280 110 29],'fontunits','normalized','fontsize',.4,'HorizontalAlignment','Right');
uicontrol('style','text','string','Current Requests for Selected User','position',[5 250 696 16],'fontunits','normalized','fontsize',.8,'HorizontalAlignment','Center');
uicontrol('style','text','string','Line 1','position',[350 435 166 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Genotype 1','position',[521 435 135 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Line 2','position',[350 385 166 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Genotype 2','position',[521 385 135 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Line 3','position',[350 335 166 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Genotype 3','position',[521 335 135 16],'fontunits','normalized','fontsize',.8);

h.dor = uicontrol('style','edit','position',[160 385 106 30],'fontunits','normalized','fontsize',.4,'tag','dor');
h.number_mice = uicontrol('style','edit','position',[160 350 106 30],'fontunits','normalized','fontsize',.4,'tag','numberMice');
h.request_notes = uicontrol('style','edit','position',[160 280 496 30],'fontunits','normalized','fontsize',.4,'tag','requestNotes','HorizontalAlignment','Left');

s = getEnumValues(mice.Mice.table,'owner');
s = {'' s{:}};
v = find(strcmp('',s));
h.requestor = uicontrol('style','popupmenu','string',s,'value',v,'position',[160 415 106 35],'fontunits','normalized','fontsize',.4,'tag','requestorField','Callback',@mice.GUIs.requestorCallback);

s = getEnumValues(mice.Requests.table,'age');
s = {'' s{:}};
v = find(strcmp('',s));
h.age = uicontrol('style','popupmenu','string',s,'value',v,'position',[160 310 106 35],'fontunits','normalized','fontsize',.3,'tag','ageField');

s = getEnumValues(mice.Lines.table,'line');
s = {'' s{:}};
v = find(strcmp('',s));
h.line1 = uicontrol('style','popupmenu','string',s,'value',v,'position',[350 400 166 35],'fontunits','normalized','fontsize',.4,'tag','line1Field');
h.line2 = uicontrol('style','popupmenu','string',s,'value',v,'position',[350 350 166 35],'fontunits','normalized','fontsize',.4,'tag','line2Field');
h.line3 = uicontrol('style','popupmenu','string',s,'value',v,'position',[350 300 166 35],'fontunits','normalized','fontsize',.4,'tag','line3Field');

s = getEnumValues(mice.Genotypes.table,'genotype');
s = {'' s{:}};
v = find(strcmp('',s));
h.genotype1 = uicontrol('style','popupmenu','string',s,'value',v,'position',[521 400 135 35],'fontunits','normalized','fontsize',.4,'tag','genotype1Field');
h.genotype2 = uicontrol('style','popupmenu','string',s,'value',v,'position',[521 350 135 35],'fontunits','normalized','fontsize',.4,'tag','genotype2Field');
h.genotype3 = uicontrol('style','popupmenu','string',s,'value',v,'position',[521 300 135 35],'fontunits','normalized','fontsize',.4,'tag','genotype3Field');

cnames = {'','DOR','#Requested','#Filled','Line1','Genotype1','Line2','Genotype2','Line3','Genotype3'};
cformat = {'logical','char','char','char','char','char','char','char','char','char'};
cwidth = {22,'auto','auto',50,82,'auto',82,'auto',82,'auto'};
cedit = [true false false false false false false false false false];
h.request_table = uitable('position',[5 60 696 190],'ColumnName',cnames,'ColumnFormat',cformat,'columnwidth',cwidth,'tag','requestTable','RowName',[],'ColumnEditable',true);

h.submitRequest = uicontrol('style','pushbutton','string','Submit New Request','position',[353 10 256 50],'fontunits','normalized','fontsize',.3,'Callback',@mice.GUIs.submitRequest);
h.deleteRequest = uicontrol('style','pushbutton','string','Cancel Selected Request(s)','position',[97 10 256 50],'fontunits','normalized','fontsize',.3,'Callback',@mice.GUIs.cancelRequest);
