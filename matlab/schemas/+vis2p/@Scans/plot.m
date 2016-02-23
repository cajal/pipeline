function plot( obj )

n = length( obj );
nCols = ceil(sqrt(n)*1.2);
nRows = ceil( n/nCols );
if nRows==1
    nCols = n;
end
count = 0;
for key=enumerate( obj )
    count = count+1;
    subplot(nRows,nCols,count);
    
    [r,g] = take( Movies(key), 'red', 'green' );
    %r = r - 0.1*polyfitImage( r, 3, 25, 25 );  % equalize
    %g = g - 0.1*polyfitImage( g, 3, 25, 25 );  % equalize
    %r = r - prctile(r(:),5);
    %g = g - prctile(g(:),5);
    %g = min( 1, g/prctile(g(:),98) );
    %r = min( 1, r/prctile(r(:),98) );
    r = sqrt(max(0,r+10));  % anscombe transform
    g = sqrt(max(0,g+10));
    mx = 35;
    r = (r-min(r(:)))/(mx-min(r(:)));
    g = (g-min(g(:)))/(mx-min(g(:)));
    
    img = cat( 3, r, cat( 3, g, zeros( size(g ) ) ) );
    imshow( img );

    [masknum,x,y,r,red] = take( CircCells(key, 'masknum>1'), 'masknum', 'img_x', 'img_y', 'cell_radius', 'red_contrast' );
    for iCell = 1:length(x)
        if red(iCell)>1.3
            c = [0 0 0];
        else
            c = [1 1 1];
        end
        rectangle( 'Position', [x(iCell)-r(iCell) y(iCell)-r(iCell) 2*r(iCell) 2*r(iCell) ], 'Curvature', [1 1], 'EdgeColor', c );
        text( x(iCell),y(iCell), num2str(masknum(iCell)), 'Color', [0 0 1] );
        title( sprintf('Site %s-%03d', key.exp_date, key.scan_idx ) );
    end
end