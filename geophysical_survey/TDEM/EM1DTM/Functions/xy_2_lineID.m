function [lineID] = xy_2_lineID(x,y)
% Workspace to create figures and sort data per line
 %Output predicted data line by line 
 
lineID = zeros(length(x),1);
 
xs = x(1);
ys = y(1);

xm = xs;
ym = ys;

linenum    = 1;
linestart   = 1;
countline   = 1;
lineID(1) = linenum;

for jj = 2 : length(x)
    
    % Compute unit vector from last point to the next
    r1 = [x(jj) - x(jj-1) y(jj) - y(jj-1)];
    
    % Compute unit vector from last point to median point
    r2 = [x(jj-1) - xm y(jj-1) - ym];
    
    % Compute unit vector from next to first
    r3 = [x(jj) - xs y(jj) - ys];
    
    % Compute unit vector from first to median
    r4 = [xs - xm ys - ym];
    
    % Compute dot product between the two vectors
    % If the angle is larger than 60, then create a new line,
    % otherwise update the current trend line
    ang1 = abs( r1/norm(r1) * r2'/norm(r2) );
    ang2 = abs( r3/norm(r3) * r4'/norm(r4) );
    
    if ang1 < cosd(45) || ang2 < cosd(45) && jj > 2
        
        xs = x(jj);
        ys = y(jj);
        
        xm = xs;
        ym = ys;
        
        linenum   = linenum + 1;
        linestart  = jj;
        
        % Update line number in data structure for current station
        lineID(jj) = linenum;
        
    else
        
%         G = [ones(jj-linestart+1,1) X(linestart:jj)];
%         pline = (G' * G)\ (G' * Y(linestart:jj));

        xm = median(x(linestart:jj));
        ym = median(y(linestart:jj));
        
        % Update line number in data structure for current station
        lineID(jj) = linenum;
        
        countline = countline + 1;
        
%         figure(1); scatter(x,y,2); hold on
%         plot([x(jj) x(jj-1)],[y(jj) y(jj-1)],'r'); hold on
%         plot([x(jj-1) xm],[y(jj-1) ym],'g');hold on  
%         plot([x(jj) xs],[y(jj) ys]);  hold on
%         plot([xs xm],[ys ym],'m'); hold off
        
    end
    
    
end