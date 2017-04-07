classdef FIFO < handle
    % FIFO -- a simple FIFO queue with a circular fixed-size buffer
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
        buffer_size
        buffer
        head  % next position to insert
        count
    end
    
    properties(Dependent)
        contents
    end
    
    methods
        
        function self = FIFO(buffer_size)
            self.buffer_size = buffer_size;
            self.reset
        end
        
        function reset(self)
            self.buffer = cell(1, self.buffer_size);
            self.head = 1;
            self.count = 0;
        end
        
        function push(self, item)
            if self.length >= self.buffer_size
                error 'FIFO buffer is full'
            end
            self.buffer{self.head} = item;
            self.head = mod(self.head, self.buffer_size)+1;
            self.count = self.count + 1;
        end
        
        function item = pop(self)
            item = self.peek;
            self.buffer{mod(self.head-self.count-1, self.buffer_size)+1} = []; % free memory
            self.count = self.count - 1;
        end
        
        function item = peek(self)
            if self.isempty
                error 'queue is empty'
            else
                item = self.buffer{mod(self.head-self.count-1, self.buffer_size)+1};
            end
        end
        
        function yes = isempty(self)
            yes = ~self.count;
        end
        
        function n = length(self)
            n = self.count;
        end
        
        function data = get.contents(self)
            data = self.buffer(mod(self.head+(-self.count:-1)-1, self.buffer_size)+1);
        end
    end
    
    methods(Static)
        function test
            s = stimulus.core.FIFO(1024);
            objects = {1, [2 3], {2, 3}, 'four', struct('five', 5, 'six', 6)};
            
            s.push(objects{1})            
            s.push(objects{2})
            ob1 = s.pop();
            ob2 = s.pop();
            assert(isequal(ob1, objects{1}))
            assert(isequal(ob2, objects{2}))
            
            s.push(objects{3})            
            s.push(objects{4})
            ob3 = s.peek();
            ob4 = s.pop();
            assert(isequal(ob3, ob4));
        end
    end
end