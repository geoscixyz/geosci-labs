% Function EM1DFM_LC_driver
% 
% Script for a Lateraly Constrained 1D inversion of frequency domain EM
% data. The program uses EM1DFM as central solver for the invidivual
% inversions. Each 1D sounding is inverted for a single beta iteration,
% then merged onto a 3D mesh. The merged conductivity [susceptibility] is
% used as a reference model for the subsequent iterations.
% 
% Data is loaded from a UBC-1D format and store in a data structure. See
% sub-function [load_EM1DFM_obs] for datails on the structure.
% 
% INPUT:
% work_dir: Location of the input files
% EM1DFM_LC.inp: input file (see file for description of lines)
%
% OUTPUT:
% Inv_MOD_iterX.con: 3D conductivity model
% Inv_PRED_iter1.pre: Data matrix for plotting -->
%
% 
% Last update: November 23, 2016
% D Fournier
% fourndo@gmail.com


clear all
close all

addpath '.\Functions';

work_dir    = '.';
dsep = '\';

% dataHead = [1,2,14,13,3,4,5,6,7,8]; % Data header for [x,y,xOffset,height,
% ds_r = 100; % Downsampling radius
%% Load input file
[meshfile,obsfile,topofile,nullfile,m_con,con_ref,m_sus,sus_ref,alpha_con,alpha_sus,beta,cooling,target,bounds,mtype,interp_n,interp_r,interp_s] = EM1DFM_read_inp([work_dir dsep 'EM1DFM_LC.inp']);

%% Load models and parameters
[xn,yn,zn] = read_UBC_mesh([work_dir dsep meshfile]);

[Zn,Xn,Yn] = ndgrid(zn,xn,yn);

mcell = (length(zn)-1)*(length(xn)-1)*(length(yn)-1);


% Create or load reference cond model
if ischar(m_con)==1
    
    m_con = load([work_dir dsep m_con]);
    HSflag = 1;
elseif isempty(m_con)

    % If m_con is empty, then start with HS
    m_con = ones(mcell,1)*1e-8;
    HSflag = 0;
    
else
    
    m_con = ones(mcell,1)*m_con;
    HSflag = 1;
end

% Create or load starting cond model
if ischar(con_ref)==1
    
    con_ref = load([work_dir dsep con_ref]);
    
else
    
    con_ref = ones(mcell,1)*con_ref;
    
end

% Create or load reference cond model
if ischar(m_sus)==1
    
    m_sus = load([work_dir dsep m_sus]);
    
else
    
    m_sus = ones(mcell,1)*m_sus;
    
end

% Create or load starting cond model
if ischar(sus_ref)==1
    
    sus_ref = load([work_dir dsep sus_ref]);
    
else
    
    sus_ref = ones(mcell,1)*sus_ref;
    
end



%% Reformat data in array structure
% Last entry specifies the minimum distance between points
freqin = [875 4920 33000];% 5001 901]; %DIGHEM

% Load data matrix and reformat to structure
% dataMat = load([work_dir '\data']);

% rawdata:  Matlab array for data
% frequin:  Unique frequencies
% radius:   Minimum distance between points. Used to filter out data.
% limits:   xmin,xmax,ymin,ymax limit extent
% indx:     index for the following colums [x,y,dx,dz,freq1_I,freq1_Q,...]
% [data,xyz] = rawdata_2_EM1DFM(dataMat.data,freqin,ds_r,[],dataHead);

% Write out the same data to EM1DFM data format
% writeem1dfmobs(work_dir,'EM1DFM_data.obs',data,'')

% % Load obs file in EM1DFM format
data = load_EM1DFM_obs(work_dir,obsfile);
[stnID,IA,IC] = unique(data{9}(:,3));

xy = data{9}(IA,1:2);
xy(:,1) = xy(:,1) - data{1}(IA,1);
xyz = [xy -data{1}(IA,3)];
% data{7} = abs(data{7});

nstn = size(xyz,1);

% % Create beta vector
beta = ones(nstn,1)*beta;
%% Change uncertainties
freqs = unique(data{3}(:));
floor = ones(length(freqs))*10;%1:length(freqs);
pct = ones(length(freqs))*.1;

for ii = 1:length(freqs)
    indx = data{3}(:)==freqs(ii);
    data{8}(indx,1) = abs(data{7}(indx,1)) * pct(ii) + floor(ii);
    data{8}(indx,2) = abs(data{7}(indx,2)) * pct(ii) + floor(ii);
end

data{7}(:,2)=abs(data{7}(:,2));
%% Downsample data
% figure;
% scatter(xyz(:,1),xyz(:,2),3);hold on
% 
% indx = Filter_xy(xyz(:,1),xyz(:,2),25);
% 
% scatter(xyz(indx==1,1),xyz(indx==1,2),3,'ro');
% xyz = xyz(indx==1,:);
% 
% axis equal
% % Kron the index by 3
% indx = kron(indx,ones(3,1));
% for ii = 1 : size(data,2)
%     
%     data{ii} = data{ii}(indx==1,:);
%     
% end

% % Write data out X Y Z Freq In Quad Uncert_In Uncert_Quad
data_out = [data{9}(:,1:3) data{3} data{7}(:,1:2) data{8}(:,1:2)];
save([work_dir '\Inv_Data_XYZ.dat'],'-ascii','data_out');
writeem1dfmobs(work_dir,'EM1DFM_data.obs',data,'')

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

%% Create Querry (Q) matrix and interpolation (P) for 1D to 3D mesh
% Search for interpolation matrix. If not in directory, then calculate
fprintf('Creating Interpolation matrix\n')
fprintf('This may take a while\n')
Q = make_EM1D_Q_3D(work_dir,meshfile,nullcell,xyz);

[P,W,indx] = make_EM1D_P_3D(work_dir,meshfile,Q,interp_n,interp_r,interp_s,alpha_con(2),alpha_con(3));


%% RUN 1D INVERSION

% Pre-set average misfit
phid_all = [99999 99999];
phid = ones(size(Q,1),1)*99999;
pred = zeros(size(data{7},1),8);

% Count iterations
ii = 1;
max_iter = 12;

% set(figure, 'Position', [25 50 1800 900])

while ( ii <= max_iter || phid_all(end) <= phid_all(end-1) ) &&...
        phid_all(end) > size(data{7},1)*2

    % Leave cell weights to unity (can be manually changed)
    w=ones(mcell,1);
    
    % Run the inversions
    [m_con1D,m_sus1D,m_misfit,phid,phim,beta,pred,bHSpace] = run_EM1DFM_inv(work_dir,meshfile,data,m_con,con_ref,m_sus,sus_ref,alpha_con,alpha_sus,w,Q,mtype,1,phid,beta,cooling,target,'both',ii,pred(:,5:6),xyz,HSflag);
        
    m_con = interp1D_to_3D(m_con1D,P,W,indx);
    m_sus = interp1D_to_3D(m_sus1D,P,W,indx);
    m_misfit = interp1D_to_3D(m_misfit,P,W,indx);
    
    con_ref = m_con;
    
    if HSflag == 0 && ii==1
        
        bHSpace = interp1D_to_3D(bHSpace,P,W,indx);
       save([work_dir '\Bestfitting_HS.dat'],'-ascii','bHSpace');
       
    end

    % Apply bounds
%     m_con(m_con<=1e-5) = 1e-5;

    m_con(nullcell==0) = 1e-8;
    m_con(m_con==0) = 1e-8;
    m_sus(nullcell==0) = -100;

    save([work_dir dsep 'Inv_MOD_iter' num2str(ii) '.con'],'-ascii','m_con');
    save([work_dir dsep 'Inv_PRED_iter' num2str(ii) '.pre'],'-ascii','pred');
    
    % Save model susceptibility model (mode 3 only)
    if mtype == 3
        save([work_dir dsep 'Inv_MOD_iter' num2str(ii) '.sus'],'-ascii','m_sus')
    end
    
    % Save misfit map
%     save([work_dir '\Misfit.con'],'-ascii','m_misfit')
    
    % Compute final data misfit
%     phid_all(ii) = sum( ((data{7}(:,1) - pred(:,5)) ./ (data{8}(:,1))).^2 +...
%         ( (data{7}(:,2) - pred(:,6)) ./ (data{8}(:,2)) ).^2 );
    
    phid_all(ii) = sum(phid);

    
    
    ii = ii + 1;


end

% arg = gtext('Would you like to run the forward model on the final interpolated model? Y:yes or N:no\n');
% 
% if strcmp(arg,'Y')
%         
% dpred = run_EM1DFM_fwr(work_dir,meshfile,obsfile,m_con,m_sus,Q);
% 
% pred = [dpred(:,1:3) data{3} dpred(:,5:6) data{8}(:,1:2)];
% save([work_dir dsep 'FWR_pred_final.dat'],'-ascii','pred');
% end
