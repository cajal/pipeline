% erase cluster around the center
function outmap = eraseCluster(map, center_x, center_y, radius)

outmap = map ;
[rows,cols] = size(map) ;
for ii=1:rows
    for jj=1:cols
        if (ii-center_y)^2 + (jj-center_x)^2 < radius^2
            outmap(ii,jj) = 0 ;
        end
    end
end