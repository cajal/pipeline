f = figure;
set(f,'position',[0 0 300 720]);

uicontrol('style','pushbutton','string','View Current Mice','fontunits','normalized','fontsize',.2,'position',[25 625 250 70],'Callback',@mice.GUIs.MenuViewMice);
uicontrol('style','pushbutton','fontunits','normalized','fontsize',.2,'position',[25 550 250 70],'string','Add New Mice (From Within Colony)','Callback',@mice.GUIs.MenuNewMice);
uicontrol('style','pushbutton','fontunits','normalized','fontsize',.2,'position',[25 475 250 70],'string','Add New Mice (From Other Source)','Callback',@mice.GUIs.MenuNewFounder);
uicontrol('style','pushbutton','fontunits','normalized','fontsize',.2,'position',[25 400 250 70],'string','Add/Modify Mouse Lines','Callback',@mice.GUIs.MenuNewLine);
uicontrol('style','pushbutton','fontunits','normalized','fontsize',.2,'position',[25 325 250 70],'string','Update Mouse Info','Callback',@mice.GUIs.MenuUpdateMouseInfo);
uicontrol('style','pushbutton','fontunits','normalized','fontsize',.2,'position',[25 250 250 70],'string','Transfer Mice','Callback',@mice.GUIs.MenuTransferMice);
uicontrol('style','pushbutton','fontunits','normalized','fontsize',.2,'position',[25 175 250 70],'string','Update Used/Euthanized Mice','Callback',@mice.GUIs.MenuUsedMice);
uicontrol('style','pushbutton','fontunits','normalized','fontsize',.2,'position',[25 100 250 70],'string','Request Mice','Callback',@mice.GUIs.MenuRequestMice);
uicontrol('style','pushbutton','fontunits','normalized','fontsize',.2,'position',[25 25 250 70],'string','View Current Requests','Callback',@mice.GUIs.MenuViewRequests);
