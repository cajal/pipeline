%{
rf.Fluorophore (lookup) # calcium-sensitive indicators
fluorophore  : char(10)   # fluorophore short name
-----
notes = ''  :  varchar(2048)  
%}

classdef Fluorophore < dj.Relvar

    methods
        function fill(self)
            self.inserti({
                'OGB'       ''
                'GCaMP6s'   ''
                'GCaMP6m'   ''
                'GCaMP6f'   ''
                'TN-XXL'    ''
                'Twitch2B'  ''
                })
        end
    end
end



