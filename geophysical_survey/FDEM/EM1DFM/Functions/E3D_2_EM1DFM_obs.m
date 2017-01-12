% Function EM1DFM_LC_driver
% 
% 
% 
% Last update: August 23, 2015
% D Fournier
% fourndo@gmail.com


clear all
close all

addpath '.\Functions';

work_dir    = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\UBC_GIF\em_examples\geophysical_survey\FDEM';

dsep = '\';

%% Load input file
obsfile = 'E3D_Sphere.obs';
predfile = 'Broadband\WholeSpace2_dpred0.txt';
topofile = 'Topo.topo';

topo = read_UBC_topo([work_dir dsep topofile]);

radius = 0;
%% Reformat data in array structure
% Last entry specifies the minimum distance between points

% % Load pred file in E3D format
[trx, d] = load_E3D_obs([work_dir dsep obsfile]);
[~, prim] = load_E3D_pred([work_dir dsep predfile]);

H0true = 0.00017543534291434681 * pi ;

% Convert to ppm
d(:,end-3) = (d(:,end-3) - prim(:,end-1)) / H0true * 1e+6;
d(:,end-2:end) = d(:,end-2:end) / H0true * 1e+6;

data = convert_E3D_2_EM1D(trx,d,topo, radius);

%% Write to file
writeem1dfmobs(work_dir,'EM1DFM.obs',data,'')