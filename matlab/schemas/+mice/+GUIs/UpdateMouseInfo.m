f = figure;
set(f, 'position',[0 0 1500 500]);

uicontrol('style','text','string','Enter Mouse IDs:','position',[50 430 110 29],'fontunits','normalized','fontsize',.4,'HorizontalAlignment','Right');
uicontrol('style','text','string','Or Enter Range:','position',[50 400 110 29],'fontunits','normalized','fontsize',.4,'HorizontalAlignment','Right');
uicontrol('style','text','string','-','position',[210 400 10 29],'fontunits','normalized','fontsize',.4,'HorizontalAlignment','Center');

h.animal_id1 = uicontrol('style','edit','position',[160 430 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID1');
h.animal_id2 = uicontrol('style','edit','position',[210 430 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID2');
h.animal_id3 = uicontrol('style','edit','position',[260 430 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID3');
h.animal_id4 = uicontrol('style','edit','position',[310 430 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID4');
h.animal_id5 = uicontrol('style','edit','position',[360 430 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID5');
h.animal_id6 = uicontrol('style','edit','position',[410 430 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID6');
h.animal_id7 = uicontrol('style','edit','position',[460 430 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID7');
h.animal_id8 = uicontrol('style','edit','position',[510 430 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID8');
h.animal_id9 = uicontrol('style','edit','position',[560 430 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID9');
h.animal_id10 = uicontrol('style','edit','position',[610 430 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID10');
h.animal_id11 = uicontrol('style','edit','position',[660 430 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID11');
h.animal_id12 = uicontrol('style','edit','position',[710 430 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID12');
h.animal_id13 = uicontrol('style','edit','position',[760 430 50 30],'fontunits','normalized','fontsize',.4,'tag','animalID13');
h.range_start = uicontrol('style','edit','position',[160 400 50 30],'fontunits','normalized','fontsize',.4,'tag','rangeStart');
h.range_end = uicontrol('style','edit','position',[220 400 50 30],'fontunits','normalized','fontsize',.4,'tag','rangeEnd');
h.find = uicontrol('style','pushbutton','position',[50 370 110 29],'fontunits','normalized','fontsize',.4,'string','Find Mice','HorizontalAlignment','Center','Callback',@mice.GUIs.findUpdateMice);

cnames = {'ID','DOB','DOW','Sex','Color','Ear Punch','Line 1','Genotype 1','Line 2','Genotype 2','Line 3','Genotype 3','Notes'};
genotypes = getEnumValues(mice.Genotypes.table,'genotype');
sexes = getEnumValues(mice.Mice.table,'sex');
colors = getEnumValues(mice.Mice.table,'color');
ears = getEnumValues(mice.Mice.table,'ear_punch');
cformat = {'char','char','char',sexes,colors,ears,'char',genotypes,'char',genotypes,'char',genotypes,'char'};
cwidth = {50, 75, 75, 100, 100, 100,100,125,100,125,100,125,190};
h.table = uitable('position',[50 60 1400 310],'RowName',' ','ColumnName',cnames,'tag','miceTable','CellSelectionCallback',@mice.GUIs.selectRow,'ColumnFormat',cformat,'ColumnEditable',[false true true true true true false true false true false true true],'ColumnWidth',cwidth);

h.submitUpdateMice = uicontrol('style','pushbutton','string','Update Mouse Info','position',[650 10 200 50],'fontunits','normalized','fontsize',.35,'Callback',@mice.GUIs.submitUpdateMice);