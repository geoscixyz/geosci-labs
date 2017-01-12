function [nullcell,topocell,ztopo_n] = topocheck(xn,yn,zn,topo)
% [nullcell,topocell,toponode] = topocheck(Xn,Yn,Zn,topo)
% Create active cell matrix from discretized topography and topocell for
% all cells intersected by the toposurface
%
% Inputs//
% xn, yn, zn: 1-D vector for the X, Y and Z location of all nodes in mesh
% topo: Topography array 3-by-points [x(:) y(:) z(:)]
%
% Output//
% nullcell: 1D vector, in UBC format, of binary values 1 (active) or
% 0(inactive) depending if below or above topography
% 
% topocell: 1D vector, in UBC format, of binary values 1 (yes) or 0 (no)
% if cell is intersected by topo surface
% 

%% FOR DEV
% clear all
% close all
% 
% work_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\Projects\Research\Modelling\Topo_adjust';
% 
% meshfile = 'Mesh_5m.msh';
% topofile = 'Gaussian.topo';

%% SCRIPT START HERE
nxn = length(xn);
nyn = length(yn);
nzn = length(zn);

nx = nxn-1;
ny = nyn-1;
nz = nzn-1;

[Xn,Yn] = ndgrid(xn,yn);
% Xn = Xn(:);
% Yn = Yn(:);

% Create topo surface
F = scatteredInterpolant(topo(:,1) , topo(:,2) , topo(:,3),'linear' ,'linear');
ztopo_n = F( Xn,Yn);

% Look at 8 corner of each cell and form a logical matrices 
% depending on their location with respect to topography
% below=1 , above = 0;
nullcell = ones(nz,nx,ny);
topocell = zeros(nz,nx,ny);

% Keep track of progress
fprintf('Topocheck Calculations\n');
progress = 0;
tic
for ii = 1 : nx
    
    for jj = 1 : ny
        
        count = 1;
        flag = 0;
        while count <= nz && flag == 0;

            if sum((ztopo_n(ii:ii+1,jj:jj+1) < zn(count)) +...
                    (ztopo_n(ii:ii+1,jj:jj+1) < zn(count+1)) ) > 0
                
                nullcell(count,ii,jj) = 0;    
                count = count + 1;
                
            else
                
                topocell(count,ii,jj) = 1;
                flag = 1;
                
            end
            
        end
        
                % Prompt iteration and time message
        d_iter = floor(ii*jj/(nx*ny)*20);

        if  d_iter > progress

            fprintf('Computed %i pct of data in %8.5f sec\n',d_iter*5,toc)

            progress = d_iter;        
            tic

        end
        
    end
    
end
    
nullcell = nullcell(:);
topocell = topocell(:);
