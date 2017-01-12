% Function EM1DFM_FWR_driver
% 
% Script to forward model EM1DFM data. 
% 
% Data is loaded from a UBC-1D format and store in a data structure. See
% sub-function [load_EM1DFM_obs] for datails on the structure.
% 
% INPUT:
% work_dir: Location of the input files
% EM1DFM_LC.inp: input file (see file for description of lines)
%
% OUTPUT:
% Inv_PRED_iter1.pre: Data matrix for plotting -->
%
% Last update: October 27, 2015
% D Fournier
% fourndo@gmail.com


clear all
close all

addpath '.\Functions';

work_dir    = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Google Drive\Tli_Kwi_Cho\Modelling\Inversion\EM\DIGHEM\1D';

dsep = '\';

%% Load input file
[meshfile,obsfile,topofile,nullfile,m_con,m_sus] = EM1DFM_FWR_read_inp([work_dir dsep 'EM1DFM_FWR.inp']);

%% Load models and parameters
[xn,yn,zn] = read_UBC_mesh([work_dir dsep meshfile]);

[Zn,Xn,Yn] = ndgrid(zn,xn,yn);

mcell = (length(zn)-1)*(length(xn)-1)*(length(yn)-1);


% Create or load reference cond model
if ischar(m_con)==1
    
    m_con = load([work_dir dsep m_con]);

else
    
    m_con = ones(mcell,1)*m_con;

end

% Create or load reference cond model
if ischar(m_sus)==1
    
    m_sus = load([work_dir dsep m_sus]);
    
else
    
    m_sus = ones(mcell,1)*m_sus;
    
end


%% Load topography
if ~isempty(topofile) && isempty(nullfile)
    
    topo = read_UBC_topo([work_dir dsep topofile]);
    [nullcell,temp,temp] = topocheck(xn,yn,zn,topo);
    save([work_dir dsep 'nullcell.dat'],'-ascii','nullcell');

elseif isempty(topofile) && ~isempty(nullfile)
    
    nullcell = load([work_dir dsep nullfile]);
else
    
    nullcell = ones(mcell,1);
    
end
%% Reformat data in array structure
data = load_EM1DFM_obs(work_dir,obsfile);
xyz = [data{9}(1:3:end,1:2) -data{1}(1:3:end,3)];


%% Change uncertainties
% indx = data{3}(:)==56000;
% data{8}(indx,1) = abs(data{7}(indx,1))*0.07 + 5;
% data{8}(indx,2) = abs(data{7}(indx,2))*0.07 + 5;
% 
% indx = data{3}(:)==7200;
% data{8}(indx,1) = abs(data{7}(indx,1))*0.07 + 2;
% data{8}(indx,2) = abs(data{7}(indx,2))*0.07 + 2;
% 
% indx = data{3}(:)==900;
% data{8}(indx,1) = abs(data{7}(indx,1))*0.07 + 0.5;
% data{8}(indx,2) = abs(data{7}(indx,2))*0.07 + 0.5;

%% Downsample data


%% Write lines of data
% Assign line number to data
% lineID = xy_2_lineID(xyz(:,1),xyz(:,2));
% line = unique(lineID);
% 
% for ii = 1 : length(line);
%     
%     for jj = 1 : size(data,2)
%         
%         subdata{jj} = data{jj}(lineID==line(ii),:);
%         
%     end
%     
%     writeem1dfmobs(work_dir,['DIGHEM_line' num2str(line(ii)) '.obs'],subdata,'')
% 
% end

%% Create Querry (Q) matrix and interpolation (P) for 1D to 3D mesh
% Search for interpolation matrix. If not in directory, then calculate
fprintf('Creating Interpolation matrix\n')
fprintf('This may take a while\n')
Q = make_EM1D_Q_3D(work_dir,meshfile,nullcell,xyz);

%% RUN 1D FWR Calculations
       
dpred = run_EM1DFM_fwr(work_dir,meshfile,obsfile,m_con,m_sus,Q);

pred = [dpred(:,1:3) data{3} dpred(:,5:6) data{8}(:,1:2)];
save([work_dir dsep 'FWR_pred_final.dat'],'-ascii','pred');
