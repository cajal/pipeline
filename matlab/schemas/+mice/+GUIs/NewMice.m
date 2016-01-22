f = figure;
set(f, 'position',[0 0 1404 750])

uicontrol('style','text','string','Ear Tag #','position',[50 700 90 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Alternate ID','position',[145 700 90 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','DOB','position',[240 700 90 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','DOW','position',[335 700 90 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Father','position',[430 700 90 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Mother 1','position',[525 700 90 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Mother 2','position',[620 700 90 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Sex','position',[715 700 106 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Color','position',[826 700 106 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Ear Punch','position',[937 700 106 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Line 1','position',[1048 700 166 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Genotype 1','position',[1219 700 135 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Line 2','position',[1048 630 166 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Genotype 2','position',[1219 630 135 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Line 3','position',[1048 560 166 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Genotype 3','position',[1219 560 135 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Owner','position',[50 630 106 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Facility','position',[161 630 106 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Room','position',[272 630 106 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Rack','position',[383 630 90 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Row','position',[478 630 90 16],'fontunits','normalized','fontsize',.8);
uicontrol('style','text','string','Notes','position',[573 630 470 16],'fontunits','normalized','fontsize',.8);

h.animal_id = uicontrol('style','edit','position',[50 665 90 35],'fontunits','normalized','fontsize',.4,'tag','animalIDField');
h.other_id = uicontrol('style','edit','position',[145 665 90 35],'fontunits','normalized','fontsize',.4,'tag','otherIDField');
h.dob = uicontrol('style','edit','position',[240 665 90 35],'fontunits','normalized','fontsize',.4,'tag','dobField');
h.dow = uicontrol('style','edit','position',[335 665 90 35],'fontunits','normalized','fontsize',.4,'tag','dowField');
h.parent1 = uicontrol('style','edit','position',[430 665 90 35],'fontunits','normalized','fontsize',.4,'tag','parent1Field');
h.parent2 = uicontrol('style','edit','position',[525 665 90 35],'fontunits','normalized','fontsize',.4,'tag','parent2Field');
h.parent3 = uicontrol('style','edit','position',[620 665 90 35],'fontunits','normalized','fontsize',.4,'tag','parent3Field');

s = getEnumValues(mice.Mice.table,'sex');
v = find(strcmp('unknown',s));
h.sex = uicontrol('style','popupmenu','string',s,'value',v,'position',[715 660 106 35],'fontunits','normalized','fontsize',.4,'tag','sexField');

s = getEnumValues(mice.Mice.table,'color');
v = find(strcmp('unknown',s));
h.color = uicontrol('style','popupmenu','string',s,'value',v,'position',[826 660 106 35],'fontunits','normalized','fontsize',.4,'tag','colorField');

s = getEnumValues(mice.Mice.table,'ear_punch');
v = find(strcmp('None',s));
h.ear_punch = uicontrol('style','popupmenu','string',s,'value',v,'position',[937 660 106 35],'fontunits','normalized','fontsize',.4,'tag','earpunchField');

s = getEnumValues(mice.Mice.table,'owner');
v = find(strcmp('none',s));
h.owner = uicontrol('style','popupmenu','string',s,'value',v,'position',[50 590 106 35],'fontunits','normalized','fontsize',.4,'tag','ownerField');

s = getEnumValues(mice.Mice.table,'facility');
v = find(strcmp('unknown',s));
h.facility = uicontrol('style','popupmenu','string',s,'value',v,'position',[161 590 106 35],'fontunits','normalized','fontsize',.4,'tag','facilityField');

s = getEnumValues(mice.Mice.table,'room');
v = find(strcmp('unknown',s));
h.room = uicontrol('style','popupmenu','string',s,'value',v,'position',[272 590 106 35],'fontunits','normalized','fontsize',.4,'tag','roomField');

h.rack = uicontrol('style','edit','position',[383 595 90 35],'fontunits','normalized','fontsize',.4,'tag','rackField');
h.row = uicontrol('style','edit','position',[478 595 90 35],'fontunits','normalized','fontsize',.4,'tag','rowField');
h.mouse_notes = uicontrol('style','edit','position',[573 595 470 35],'fontunits','normalized','fontsize',.4,'HorizontalAlignment','left','tag','notesField'); 

s = getEnumValues(mice.Lines.table,'line');
s = {'' s{:}};
v = find(strcmp('',s));
h.line1 = uicontrol('style','popupmenu','string',s,'value',v,'position',[1048 660 166 35],'fontunits','normalized','fontsize',.4,'tag','line1Field');
h.line2 = uicontrol('style','popupmenu','string',s,'value',v,'position',[1048 590 166 35],'fontunits','normalized','fontsize',.4,'tag','line2Field');
h.line3 = uicontrol('style','popupmenu','string',s,'value',v,'position',[1048 520 166 35],'fontunits','normalized','fontsize',.4,'tag','line3Field');

s = getEnumValues(mice.Genotypes.table,'genotype');
v = find(strcmp('unknown',s));
h.genotype1 = uicontrol('style','popupmenu','string',s,'value',v,'position',[1219 660 135 35],'fontunits','normalized','fontsize',.4,'tag','genotype1Field');
h.genotype2 = uicontrol('style','popupmenu','string',s,'value',v,'position',[1219 590 135 35],'fontunits','normalized','fontsize',.4,'tag','genotype2Field');
h.genotype3 = uicontrol('style','popupmenu','string',s,'value',v,'position',[1219 520 135 35],'fontunits','normalized','fontsize',.4,'tag','genotype3Field');

last = fetch(mice.Mice);
number = size(last,1);
if number ~= 0 
    set(h.animal_id,'string',(last(number).animal_id + 1))
end

h.autopopulate = uicontrol('style','checkbox','string','Autopopulate','position',[140 550 150 35],'fontunits','normalized','fontsize',.4,'tag','autoBox');
h.clear = uicontrol('style','pushbutton','string','Clear','position',[50 550 90 35],'fontunits','normalized','fontsize',.4,'Callback',@mice.GUIs.clearEntry);

cnames = {'ID','Alt ID','DOB','DOW','Father','Mother1','Mother2','Sex','Color','Ear Punch','Line1','Genotype1','Line2','Genotype2','Line3','Genotype3','Owner','Facility','Room','Rack','Row','Notes'};
cformat = {'char','char','char','char','char','char','char','char','char','char','char','char','char','char','char','char','char','char','char','char','char','char'};
cwidth = {40,40,'auto','auto',40,50,50,40,40,60,'auto','auto','auto','auto','auto','auto','auto',50,40,40,40,'auto'};
h.new_mice = uitable('position',[50 60 1304 450],'RowName',' ','ColumnName',cnames,'ColumnFormat',cformat,'columnwidth',cwidth,'tag','miceTable','CellSelectionCallback',@mice.GUIs.selectRow);

h.add = uicontrol('style','pushbutton','string','+','position',[50 510 25 25],'fontunits','normalized','fontsize',.8,'backgroundcolor','g','Callback',@mice.GUIs.plusCallback);
h.delete = uicontrol('style','pushbutton','string','-','position',[75 510 25 25],'fontunits','normalized','fontsize',.8,'backgroundcolor','r','Callback',@mice.GUIs.minusCallback);

h.submit = uicontrol('style','pushbutton','string','Submit New Mice to Database','position',[524 10 256 50],'fontunits','normalized','fontsize',.35,'Callback',@mice.GUIs.submitMice,'tag','submitMiceButton');


