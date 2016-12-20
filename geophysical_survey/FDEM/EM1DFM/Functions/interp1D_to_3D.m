function [m3D] = interp1D_to_3D(m1D,P,W,indx)
% Function interp1D_to_3D(m1D,P,W)
% Script to propagate a 1D inverted model onto the full 3D mesh
% Requires to run the function make_EM1D_Q_3D and make_EM1D_P_3D matrix
% ahead of time.
% Computes the interpolation via a sparse matrix notation, where the matrix
% P: index of cells to grab value from
% W: inverse distance weights for averaging

% Pre-allocate model
m3D = ones(size(m1D))*1e-8;

% Number of interpolated points
ncol = size(P,2);

% Only select the cells that are interpolated
% indx = P(:,1)~=0;

mkron = kron(ones(1,ncol),m1D);
    
temp = sum(mkron(P).*W,2);

m3D(indx) = temp;