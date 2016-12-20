function [Wx, Wy, Wz] = make_EM1DFM_grad(work_dir,meshfile)
% Create derivative matrices for given 3D mesh using a centered difference
% sheme.
%
% INPUT
% work_dir: Directory to find mesh file
% meshfile: Mesh file in UBC3D format
%
% OUTPUT
% Wx, Wy, Wz, Ws: Derivative matrices of size mcell-by-mcell, with entries
% of [-V/(2*dx(n-1)) ... 0 ... V/(2*dx(n+1))] exept at boundaries.

%% FOR DEV ONLY
% clear all
% close all
% 
% dx = [2 2 2];
% dy = [5 6 7 8];
% dz = [9 10 11 12 13];
% 
% nx = 3;
% ny = 4;
% nz = 5;

%% SCRIPT STARTS HERE
mesh = get_UBC_mesh([work_dir '\' meshfile]);
nx = mesh(1,1); %size(X,1);    %number of cell in X
ny = mesh(1,2); %size(X,2);    %number of cell in Y
nz = mesh(1,3); %size(X,3);    %number of cell in Z

mcell = nx*ny*nz;

% Cell size array
dx = mesh(3,1:nx);
dy = mesh(4,1:ny);
dz = mesh(5,1:nz);

% Create diagonal matrices for dimensions
dX = spdiags(dx',0,nx,nx);
dY = spdiags(dy',0,ny,ny);
dZ = spdiags(dz',0,nz,nz);

Dx = kron(kron(speye(ny),dX),speye(nz));
Dy = kron(kron(dY,speye(nx)),speye(nz));
Dz = kron(kron(speye(ny),speye(nx)),dZ);

V = Dx .* Dy .* Dz;

%% Create primary divergence (face to center)
% First and last column of Dz and Dy are set to 0, since A = 0 on BC
ddx = @(n) spdiags (ones (n+1,1)*[-1,1],[0,1],n+1,n+1);

% Create diagonal matrices for dimensions
dX = spdiags(1./(dx)',0,nx,nx);
dY = spdiags(1./(dy)',0,ny,ny);
dZ = spdiags(1./(dz)',0,nz,nz);

d_dx = dX * ddx(nx-1); 
d_dx(end,end-1)= dX(end,end);

Wx = V * kron(kron(speye(ny), d_dx),speye(nz));

d_dy = dY * ddx(ny-1); 
d_dy(end,end-1)= dY(end,end);

Wy = V * kron(kron(dY * d_dy,speye(nx)),speye(nz));

d_dz = dZ * ddx(nz-1); 
d_dz(end,end-1)= dZ(end,end);

Wz = V * kron(kron(speye(ny),speye(nx)),dZ * d_dz);