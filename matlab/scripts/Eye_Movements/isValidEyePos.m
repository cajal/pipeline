% if eye positions are within the circle centered at the peak of membership
% matrix, then include it in the calculations of RF
function eyevalid = isValidEyePos(x, y, center, radius)

eyevalid = ((x-center(2)).^2 + (y-center(1)).^2) < radius^2 ;