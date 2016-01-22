f = figure;
set(f, 'position',[0 0 900 600]);

uicontrol('style','text','string','Enter Mouse IDs:','position',[50 530 110 29],'fontunits','normalized','fontsize',.4,'HorizontalAlignment','Right');
uicontrol('style','text','string','Or Enter Range:','position',[50 500 110 29],'fontunits','normalized','fontsize',.4,'HorizontalAlignment','Right');
uicontrol('style','text','string','-','position',[210 500 10 29],'fontunits','normalized','fontsize',.4,'HorizontalAlignment','Center');
uicontrol('style','text','string','Transfer To:','position',[50 130 100 16],'fontunits','normalized','fontsize',.8,'HorizontalAlignment','Center');
uicontrol('style','text','string','Owner','position',[160 130 100 16],'fontunits','normalized','fontsize',.8,'HorizontalAlignment','Center');
uicontrol('style','text','string','Facility','position',[270 130 100 16],'fontunits','normalized','fontsize',.8,'HorizontalAlignment','Center');
uicontrol('style','text','string','Room','position',[380 130 100 16],'fontunits','normalized','fontsize',.8,'HorizontalAlignment','Center');
uicontrol('style','text','string','Rack','position',[490 130 100 16],'fontunits','normalized','fontsize',.8,'HorizontalAlignment','Center');
uicontrol('style','text','string','Row','position',[600 130 100 16],'fontunits','normalized','fontsize',.8,'HorizontalAlignment','Center');
uicontrol('style','text','string','Date','position',[710 130 100 16],'fontunits','normalized','fontsize',.8,'HorizontalAlignment','Center');

h.animal_id1 = uicontrol('style','edit','position',[160 530 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID1');
h.animal_id2 = uicontrol('style','edit','position',[210 530 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID2');
h.animal_id3 = uicontrol('style','edit','position',[260 530 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID3');
h.animal_id4 = uicontrol('style','edit','position',[310 530 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID4');
h.animal_id5 = uicontrol('style','edit','position',[360 530 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID5');
h.animal_id6 = uicontrol('style','edit','position',[410 530 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID6');
h.animal_id7 = uicontrol('style','edit','position',[460 530 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID7');
h.animal_id8 = uicontrol('style','edit','position',[510 530 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID8');
h.animal_id9 = uicontrol('style','edit','position',[560 530 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID9');
h.animal_id10 = uicontrol('style','edit','position',[610 530 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID10');
h.animal_id11 = uicontrol('style','edit','position',[660 530 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID11');
h.animal_id12 = uicontrol('style','edit','position',[710 530 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID12');
h.animal_id13 = uicontrol('style','edit','position',[760 530 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID13');
h.range_start = uicontrol('style','edit','position',[160 500 50 30],'fontunits','normalized','fontsize',.4,'tag','rangeStart');
h.range_end = uicontrol('style','edit','position',[220 500 50 30],'fontunits','normalized','fontsize',.4,'tag','rangeEnd');
h.find = uicontrol('style','pushbutton','position',[50 470 110 29],'fontunits','normalized','fontsize',.4,'string','Find Mice','HorizontalAlignment','Center','Callback',@mice.GUIs.findMice);

cnames = {'ID','Line 1','Genotype 1','Line 2','Genotype 2','Line 3','Genotype 3'};
cwidth = {75,100,125,100,125,100,125};
h.table = uitable('position',[50 160 800 310],'RowName',' ','ColumnName',cnames,'tag','miceTable','CellSelectionCallback',@mice.GUIs.selectRow,'ColumnWidth',cwidth);

owners = getEnumValues(mice.Mice.table,'owner');
owners = {'',owners{:}};
h.owner = uicontrol('style','popupmenu','string',owners,'position',[155 90 110 30],'fontunits','normalized','fontsize',.4,'tag','owner');

facilities = getEnumValues(mice.Mice.table,'facility');
facilities = {'',facilities{:}};
h.facility = uicontrol('style','popupmenu','string',facilities,'position',[265 90 110 30],'fontunits','normalized','fontsize',.4,'tag','facility');

rooms = getEnumValues(mice.Mice.table,'room');
rooms = {'',rooms{:}};
h.room = uicontrol('style','popupmenu','string',rooms,'position',[375 90 110 30],'fontunits','normalized','fontsize',.4,'tag','room');

h.rack = uicontrol('style','edit','position',[490 95 100 29],'fontunits','normalized','fontsize',.4,'HorizontalAlignment','Center','tag','rack');
h.row = uicontrol('style','edit','position',[600 95 100 29],'fontunits','normalized','fontsize',.4,'HorizontalAlignment','Center','tag','row');
h.transfer_date = uicontrol('style','edit','position',[710 95 100 29],'fontunits','normalized','fontsize',.4,'HorizontalAlignment','Center','tag','transferDate');

h.submitTransfer = uicontrol('style','pushbutton','string','Submit Transfer','position',[350 10 200 50],'fontunits','normalized','fontsize',.3,'Callback',@mice.GUIs.submitTransfer,'tag','submitTransferButton');