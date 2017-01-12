function [indx] = Filter_xy(x,y,radius)
%function Filter_xy
%Decimate point set from 2D distance to neighbour
%Input
% XYZd: [X(:) Y(:) Z(:) data(:,:)]
%
%
%Written: December 4th, 2014
%By: D. Fournier

%Set search radius
% radius=50;

% Plot initial point location
% figure; scatter(XYZd(:,1),XYZd(:,2));title('Before sorting');hold on

% Get initial number of entries
nstn = length(x);

if length(x)~=length(y)
    
    fprintf('x-coordinates should be the same size as y. Verify!\n')
    return
    
end
% Initialize mask filter  
indx = ones(nstn,1);

progress = -1;
tic 
for ii = 1 : nstn

    if indx(ii)==1

        % Compute distance from closest point
        r = ( (x(ii) - x).^2 +...
            (y(ii) - y).^2 ) .^0.5;

        % Only keep the curretn points and neighbours at distance r+
        indx(r <= radius) = 0;
        indx(ii) = 1;

    end
    
    d_iter = floor(ii/nstn*100);
    if  d_iter > progress

        fprintf('Computed %i pct of data in %8.5f sec\n',d_iter,toc)
        progress = d_iter;

    end

end

% scatter(XYZd_out(:,1),XYZd_out(:,2),'r*');title('After sorting');
