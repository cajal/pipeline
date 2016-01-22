function [h,m] = getUIData(FigHand)

% Get UI Handles

h.animal_id = findobj(FigHand,'tag','animalIDField');
h.other_id = findobj(FigHand,'tag','otherIDField');
h.dob = findobj(FigHand,'tag','dobField');
h.dow = findobj(FigHand,'tag','dowField');
h.parent1 = findobj(FigHand,'tag','parent1Field');
h.parent2 = findobj(FigHand,'tag','parent2Field');
h.parent3 = findobj(FigHand,'tag','parent3Field');
h.sex = findobj(FigHand,'tag','sexField');
h.color = findobj(FigHand,'tag','colorField');
h.ear_punch = findobj(FigHand,'tag','earpunchField');
h.owner = findobj(FigHand,'tag','ownerField');
h.facility = findobj(FigHand,'tag','facilityField');
h.room = findobj(FigHand,'tag','roomField');
h.rack = findobj(FigHand,'tag','rackField');
h.row = findobj(FigHand,'tag','rowField');
h.mouse_notes = findobj(FigHand,'tag','notesField');
h.line1 = findobj(FigHand,'tag','line1Field');
h.line2 = findobj(FigHand,'tag','line2Field');
h.line3 = findobj(FigHand,'tag','line3Field');
h.genotype1 = findobj(FigHand,'tag','genotype1Field');
h.genotype2 = findobj(FigHand,'tag','genotype2Field');
h.genotype3 = findobj(FigHand,'tag','genotype3Field');
h.new_mice = findobj(FigHand,'tag','miceTable');
h.doa = findobj(FigHand,'tag','doaField');
h.source = findobj(FigHand,'tag','sourceField');
h.isMice = findobj(FigHand,'tag','submitMiceButton');
h.isFounder = findobj(FigHand,'tag','submitFounderButton');
h.isViewMice = findobj(FigHand,'tag','printListButton');

% Get UI Data

m.animal_id = get(h.animal_id,'string');
m.other_id = get(h.other_id,'string');
m.dob = get(h.dob,'string');
m.dow = get(h.dow,'string');
m.rack = get(h.rack,'string');
m.row = get(h.row,'string');

v = get(h.sex,'value');
s = get(h.sex,'string');
m.sex = s{v};

v = get(h.color,'value');
s = get(h.color,'string');
m.color = s{v};

v = get(h.ear_punch,'value');
s = get(h.ear_punch,'string');
m.ear_punch = s{v};

v = get(h.owner,'value');
s = get(h.owner,'string');
m.owner = s{v};

v = get(h.facility,'value');
s = get(h.facility,'string');
m.facility = s{v};

v = get(h.room,'value');
s = get(h.room,'string');
m.room = s{v};

if ~isempty(h.isMice)
    m.parent1 = get(h.parent1,'string');
    m.parent2 = get(h.parent2,'string');
    m.parent3 = get(h.parent3,'string');
    v = get(h.line1,'value');
    s = get(h.line1,'string');
    m.line1 = s{v};
    v = get(h.line2,'value');
    s = get(h.line2,'string');
    m.line2 = s{v};
    v = get(h.line3,'value');
    s = get(h.line3,'string');
    m.line3 = s{v};
    v = get(h.genotype1,'value');
    s = get(h.genotype1,'string');
    m.genotype1 = s{v};
    v = get(h.genotype2,'value');
    s = get(h.genotype2,'string');
    m.genotype2 = s{v};
    v = get(h.genotype3,'value');
    s = get(h.genotype3,'string');
    m.genotype3 = s{v};
    m.mouse_notes = get(h.mouse_notes,'string');
end

if ~isempty(h.isFounder)
    m.doa = get(h.doa,'string');
    m.source = get(h.source,'string');
    m.founder_notes = get(h.mouse_notes,'string');
    v = get(h.line1,'value');
    s = get(h.line1,'string');
    m.line = s{v};
    v = get(h.genotype1,'value');
    s = get(h.genotype1,'string');
    m.genotype = s{v};
end

m.new_mice = get(h.new_mice,'Data');

clear v s 

end