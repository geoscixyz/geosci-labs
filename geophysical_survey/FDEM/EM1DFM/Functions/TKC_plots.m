% TKC DIGHEM
% Workspace to create figures and sort data per line
 %Output predicted data line by line 
clear all
close all

work_dir = 'C:\Users\dominiquef\Documents\Projects\UBC_Tli_Kwi_Cho\Modelling\Inversion\EM_Inversion\DIGHEM';

% Load data struture
load ([work_dir '\data']);
load ([work_dir '\pred']);

% Cycle through each station and assign line number to all stations fitting
% on a 1th-order polynomial

[nstn, IA, IC] = unique(data{1,9}(:,1));

X = data{1,9}(IA,2);
Y = data{1,9}(IA,3);

xs = X(1);
ys = Y(1);

xm = xs;
ym = ys;

linenum    = 1;
linestart   = 1;
countline   = 1;
data{1,9}((data{1,9}(:,1)==1),4) = linenum;

for jj = 2 : length(nstn)
    
    % Compute unit vector from last point to the next
    r1 = [X(jj) - X(jj-1) Y(jj) - Y(jj-1)];
    
    % Compute unit vector from last point to median point
    r2 = [X(jj-1) - xm Y(jj-1) - ym];
    
    % Compute unit vector from next to first
    r3 = [X(jj) - xs Y(jj) - ys];
    
    % Compute unit vector from first to median
    r4 = [xs - xm ys - ym];
    
    % Compute dot product between the two vectors
    % If the angle is larger than 60, then create a new line,
    % otherwise update the current trend line
    ang1 = abs( r1/norm(r1) * r2'/norm(r2) );
    ang2 = abs( r3/norm(r3) * r4'/norm(r4) );
    
    if ang1 < cosd(45) || ang2 < cosd(45) && jj > 2
        
        xs = X(jj);
        ys = Y(jj);
        
        xm = xs;
        ym = ys;
        
        linenum   = linenum + 1;
        linestart  = jj;
        
        % Update line number in data structure for current station
        data{1,9}((data{1,9}(:,1)==nstn(jj)),4) = linenum;
        
    else
        
%         G = [ones(jj-linestart+1,1) X(linestart:jj)];
%         pline = (G' * G)\ (G' * Y(linestart:jj));

        xm = median(X(linestart:jj));
        ym = median(Y(linestart:jj));
        
        % Update line number in data structure for current station
        data{1,9}((data{1,9}(:,1)==nstn(jj)),4) = linenum;
        
        countline = countline + 1;
        
        figure(1); scatter(X,Y,2); hold on
        plot([X(jj) X(jj-1)],[Y(jj) Y(jj-1)],'r'); hold on
        plot([X(jj-1) xm],[Y(jj-1) ym],'g');hold on  
        plot([X(jj) xs],[Y(jj) ys]);  hold on
        plot([xs xm],[ys ym],'m'); hold off
        
    end
    
    
end

% Now output all data points with format
% [X Y Z Line Freq Real(d) Imag(d) Real(pred) Imag(pred)]
fid1 = fopen([work_dir '\Data__vs_pred_56k.dat'],'w');
fid2 = fopen([work_dir '\Data__vs_pred_7k.dat'],'w');
fid3 = fopen([work_dir '\Data__vs_pred_900.dat'],'w');

for ii = 1 : size(data{9},1)

    if data{3}(ii) == 56000

        fprintf(fid1,'%f %f %i %i %f %f %f %f\n', data{9}(ii,2), data{9}(ii,3),...
         data{9}(ii,4),data{3}(ii),data{7}(ii,1),data{7}(ii,2),pred{7}(ii,1),pred{7}(ii,2));

    elseif data{3}(ii) == 7200

        fprintf(fid2,'%f %f %i %i %f %f %f %f\n', data{9}(ii,2), data{9}(ii,3),...
         data{9}(ii,4),data{3}(ii),data{7}(ii,1),data{7}(ii,2),pred{7}(ii,1),pred{7}(ii,2));

    else
        fprintf(fid3,'%f %f %i %i %f %f %f %f\n', data{9}(ii,2), data{9}(ii,3),...
         data{9}(ii,4),data{3}(ii),data{7}(ii,1),data{7}(ii,2),pred{7}(ii,1),pred{7}(ii,2));

    end

end

fclose('all');

