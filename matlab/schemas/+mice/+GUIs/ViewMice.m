f = figure;
set(f, 'position',[0 0 1404 750])

uicontrol('style','text','string','ID Range:','position',[50 671 75 29],'fontunits','normalized','fontsize',.4,'HorizontalAlignment','Right');
uicontrol('style','text','string','-','position',[175 671 10 29],'fontunits','normalized','fontsize',.4,'HorizontalAlignment','Center');

uicontrol('style','text','string','Parent1','position',[240 700 90 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Parent2','position',[335 700 90 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Sex','position',[430 700 106 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Color','position',[541 700 106 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Ear Punch','position',[652 700 106 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Line 1','position',[763 700 166 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Genotype 1','position',[934 700 135 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Line 2','position',[763 630 166 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Genotype 2','position',[934 630 135 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Line 3','position',[763 560 166 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Genotype 3','position',[934 560 135 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Owner','position',[50 630 106 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Facility','position',[161 630 106 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Room','position',[272 630 106 16],'fontunits','normalized','fontsize',.8);

h.range_start = uicontrol('style','edit','position',[125 671 50 30],'fontunits','normalized','fontsize',.4,'tag','rangeStart');
h.range_end = uicontrol('style','edit','position',[185 671 50 30],'fontunits','normalized','fontsize',.4,'tag','rangeEnd');

h.parent1 = uicontrol('style','edit','position',[240 671 90 30],'fontunits','normalized','fontsize',.4,'tag','parent1Field');
h.parent2 = uicontrol('style','edit','position',[335 671 90 30],'fontunits','normalized','fontsize',.4,'tag','parent2Field');

s = getEnumValues(mice.Mice.table,'sex');
s = {'' s{:}};
v = find(strcmp('',s));
h.sex = uicontrol('style','popupmenu','string',s,'value',v,'position',[430 661 106 35],'fontunits','normalized','fontsize',.4,'tag','sexField');

s = getEnumValues(mice.Mice.table,'color');
s = {'' s{:}};
v = find(strcmp('',s));
h.color = uicontrol('style','popupmenu','string',s,'value',v,'position',[541 661 106 35],'fontunits','normalized','fontsize',.4,'tag','colorField');

s = getEnumValues(mice.Mice.table,'ear_punch');
s = {'' s{:}};
v = find(strcmp('',s));
h.ear_punch = uicontrol('style','popupmenu','string',s,'value',v,'position',[652 661 106 35],'fontunits','normalized','fontsize',.4,'tag','earpunchField');

s = getEnumValues(mice.Mice.table,'owner');
s = {'' s{:}};
v = find(strcmp('',s));
h.owner = uicontrol('style','popupmenu','string',s,'value',v,'position',[50 590 106 35],'fontunits','normalized','fontsize',.4,'tag','ownerField');

s = getEnumValues(mice.Mice.table,'facility');
s = {'' s{:}};
v = find(strcmp('',s));
h.facility = uicontrol('style','popupmenu','string',s,'value',v,'position',[161 590 106 35],'fontunits','normalized','fontsize',.4,'tag','facilityField');

s = getEnumValues(mice.Mice.table,'room');
s = {'' s{:}};
v = find(strcmp('',s));
h.room = uicontrol('style','popupmenu','string',s,'value',v,'position',[272 590 106 35],'fontunits','normalized','fontsize',.4,'tag','roomField');

s = getEnumValues(mice.Lines.table,'line');
s = {'' s{:}};
v = find(strcmp('',s));
h.line1 = uicontrol('style','popupmenu','string',s,'value',v,'position',[763 660 166 35],'fontunits','normalized','fontsize',.4,'tag','line1Field');
h.line2 = uicontrol('style','popupmenu','string',s,'value',v,'position',[763 590 166 35],'fontunits','normalized','fontsize',.4,'tag','line2Field');
h.line3 = uicontrol('style','popupmenu','string',s,'value',v,'position',[763 520 166 35],'fontunits','normalized','fontsize',.4,'tag','line3Field');

s = getEnumValues(mice.Genotypes.table,'genotype');
s = {'' s{:}};
v = find(strcmp('',s));
h.genotype1 = uicontrol('style','popupmenu','string',s,'value',v,'position',[934 660 135 35],'fontunits','normalized','fontsize',.4,'tag','genotype1Field');
h.genotype2 = uicontrol('style','popupmenu','string',s,'value',v,'position',[934 590 135 35],'fontunits','normalized','fontsize',.4,'tag','genotype2Field');
h.genotype3 = uicontrol('style','popupmenu','string',s,'value',v,'position',[934 520 135 35],'fontunits','normalized','fontsize',.4,'tag','genotype3Field');

h.used = uicontrol('style','checkbox','string','Include Used/Euthanized Mice','position',[140 550 250 35],'fontunits','normalized','fontsize',.4,'tag','usedBox');
h.clear = uicontrol('style','pushbutton','string','Clear','position',[50 550 90 35],'fontunits','normalized','fontsize',.4,'Callback',@mice.GUIs.clearEntry);

cnames = {'ID','AltID','DOB','DOW','Parent1','Parent2','Parent3','Sex','Color','EarPunch','Line1','Genotype1','Line2','Genotype2','Line3','Genotype3','Owner','Facility','Room','Rack','Row','Notes'};
cformat = {'char','char','char','char','char','char','char','char','char','char','char','char','char','char','char','char','char','char','char','char','char','char'};
cwidth = {40,40,'auto','auto',50,50,50,40,40,60,'auto','auto','auto','auto','auto','auto','auto',50,40,40,40,'auto'};
h.new_mice = uitable('position',[50 60 1304 450],'RowName',' ','ColumnName',cnames,'ColumnFormat',cformat,'columnwidth',cwidth,'tag','miceTable','CellSelectionCallback',@mice.GUIs.selectRow);

h.find = uicontrol('style','pushbutton','position',[50 510 110 29],'fontunits','normalized','fontsize',.4,'string','Find Mice','HorizontalAlignment','Center','Callback',@mice.GUIs.findViewMice);

h.print = uicontrol('style','pushbutton','string','Print List','position',[524 10 256 50],'fontunits','normalized','fontsize',.35,'Callback',@mice.GUIs.printList,'tag','printListButton');