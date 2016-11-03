% for each eye position, determine which circles it falls in
function circlemembership = fillcirclemembership(x, y, xrange, yrange, lut)

circlemembership = zeros(diff(yrange)+1,diff(xrange)+1) ;

for ii=1:length(x)
    if ~isnan(x(ii)) && ~isnan(y(ii))
      x0 = round(x(ii)-xrange(1)+1) ;
      y0 = round(y(ii)-yrange(1)+1) ;
      if (x0>0) && (y0>0) && (x0<diff(xrange)+1) && (y0<diff(yrange)+1)
        circles = lut.circlesforthispoint(x0,y0) ;
        for jj=1:length(circles)
            xc = circles{jj}.x ;
            yc = circles{jj}.y ;
            if (xc<=diff(xrange)+1) && (yc<=diff(yrange)+1)
              circlemembership(yc, xc)=circlemembership(yc, xc)+1 ;
            end ;
        end
      end
    end
end