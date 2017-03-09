classdef FIFO < handle
    % FIFO -- a simple FIFO queue
    %
    % Usage:
    %   f = FIFO()
    %   f.push(item1)
    %   f.push(item2)
    %
    %   while f.length
    %       item = f.pop
    %       ...
    %   end
    
    
    properties(Access=private)
        buffer
        head
        tail
    end
    
    properties(Dependent)
        contents
    end
    
    methods
        
        function self = FIFO()
            self.reset
        end
        
        function reset(self)
            % clear the queue
            self.buffer = cell(0,1);
            self.head = 0;
            self.tail = 0;
        end
        
        function push(self, item)
            self.head = self.head + 1;
            if length(self.buffer) < self.head
                self.buffer{max(100, end*2),1} = [];
            end
            self.buffer{self.head} = item;
        end
        
        function item = pop(self)
            item = self.peek;
            self.tail = self.tail + 1;
            self.buffer{self.tail} = [];
            % truncate tail if too long
            if self.tail >= max(1000, self.head/2)
                self.buffer(1:self.tail) = [];
                self.head = self.head-self.tail;
                self.tail = 0;
            end
        end
        
        function item = peek(self)
            if self.isempty
                error 'queue is empty'
            else
                item = self.buffer(self.tail+1);
            end
        end
        
        function yes = isempty(self)
            yes = ~self.length;
        end
        
        function n = length(self)
            n = self.head - self.tail;
        end
        
        function data = get.contents(self)
            data = self.buffer(self.tail+1:self.head);
        end        
    end
end