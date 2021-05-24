%{
# stimulus protocol run during scan
-> experiment.Scan
---
protocol                    : varchar(255)                  # name of the protocol
scan_ts=CURRENT_TIMESTAMP   : timestamp                     # don't edit
%}

classdef ScanProtocol < dj.Manual
end