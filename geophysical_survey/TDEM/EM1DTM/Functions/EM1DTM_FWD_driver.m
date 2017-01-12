% FOR DEV ONLY
% Temporary function to format data to GIFTool format
% MUST BE REFORMATED FOR EVERY FILE

clear all
close all

addpath     'C:\Users\dominiquef.MIRAGEOSCIENCE\Dropbox\Master\FUNC_LIB';
work_dir    = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Google Drive\Tli_Kwi_Cho\Modelling\Inversion\EM\VTEM\1D\New_Data';
% work_dir    ='C:\Users\dominiquef.MIRAGEOSCIENCE\Google Drive\Tli_Kwi_Cho\Modelling\Inversion\EM\AeroTEMinv\1D';
meshfile    = 'UBC_mesh_small.msh';
topofile    = 'CDED_076c05_NAD27.topo';
condfile    = 'Inv_1D_DIGHEM_Block_2Ohm.con';

%% Reformat data in array structure
% Last entry specifies the minimum distance between points
% sort_EM1DTM will downsample the data using a minimum distance
% as specified by the last input

load([work_dir '\VTEM_data_DF'])
load([work_dir '\VTEM_xyz_DF'])
load([work_dir '\VTEM_tc_DF'])
load([work_dir '\VTEM_Waveform'])

% Set limits of observations or leave empty
% limits = [];

radius = 13;

nT = 1e-12;
% NI = 250*8;
A = pi*radius^2;


data = (data) * nT * A;


%% AeroTEM
% load([work_dir '\ATEM_data_DF'])
% load([work_dir '\ATEM_xyz_DF'])
% load([work_dir '\ATEM_tc_DF'])
% load([work_dir '\ATEM_Waveform'])
% 
% % Number of time channels to skip
% early_tc = 0;
% late_tc = 0;
% radius = 2.5;
% 
% nT = 1e-09;
% % A = pi*radius^2;
% NI = 250*8;
% data = (data) * nT / NI;

%%
limits(1) = 556300;
limits(2) = 558100;

% DO-18
% limits(3) = 7134380;
% limits(4) = 7134450;

% DO-27
limits(3) = 7133580;
limits(4) = 7133610;

% Filter data by location and distance
[data,xyz] = sort_EM1DTM(data,xyz,limits,10,'all');

% data = abs(data);

%% Reformat data units


%% Write EM1DFME data to file
% obsfile = 'DIGHEM_TKC_ALL.obs';
% writeem1dfmobs(inv_dir,obsfile,data,'')


%% Make nullcell from topo and mesh
% [xn,yn,zn]=read_UBC_mesh([work_dir '\' meshfile]);
% [Zn,Xn,Yn] = ndgrid(zn,xn,yn);
% 
% % Load topography
% topo = read_UBC_topo([work_dir '\' topofile]);
% 
% % Create discretize topogrphy
% [nullcell,tcellID,ztopo_n] = topocheck(Xn,Yn,Zn,topo);
% save([work_dir '\nullcell.dat'],'-ascii','nullcell');
load([work_dir '\nullcell.dat']);
% load([work_dir '\index']);

%% Create Querry (Q) matrix  for 1D to 3D mesh
% Q = make_EM1D_Q_3D(work_dir,meshfile,nullcell,xyz);
% save([work_dir '\Q'],'Q')
load([work_dir '\Q']);
    
%% Create interpolation (P) matrix
nnodes  = 10; % Number of nearest neighbours to interpolate with
% P = make_EM1D_P_3D(work_dir,meshfile,Q,nnodes,5,150);
% save([work_dir '\P'],'P')
% load([work_dir '\P']);


%% RUN 1D INVERSION

%% Create transmiter loop (Seogi's func)
xc = 0.;
yc = 0.;

[x, y] = circfun(xc, yc, radius, 24);
txloc = [x(:), y(:)];
rxloc = [0 0];

% Input#2
% txheight = height of TX (negative above the surface)
ft = 0.3048;
txheight = xyz(:,4);
rxheight = xyz(:,4);

% Create inform background models for now
% or load conductivity model
% m_con = nullcell; 
% m_con(m_con==1) = 1e-4;
% m_con(m_con==0) = 1e-4;
m_con = load([work_dir '\' condfile]);

models_in{1,1} = m_con;

% Input#6
% wf = waveform, can be 'STEPOFF' for stepoff or a number for RAMP or
% two-column for a discretized waveform (time in second)
% ta = 5.5*1e-4;
% tb = 1.1*1e-3;
% twave = linspace(0., tb, 2^7+1) ;
% wfval = [trifun(twave, ta, tb); 0 ];
% 
% twave = [twave 0.0026];
% 
% waveform = [twave(:) , wfval(:) ]; 

ntc = length(tc);
    
%% Setup inversion
mcell = length(nullcell);
% Pre-allocate for output model
m_out3D = ones(mcell,1) * 1e-8;

% % Create derivative matrices
% [Wx, Wy, Wz] = make_EM1DFM_grad(work_dir,meshfile);

% Pre-set average misfit
phid_avg = 99999;
beta = ones(size(Q,1),1);
phid = ones(size(Q,1),1)*99999;

% ndata = numel(data);
% noisefun = @(a, t)(1-exp(-a*t));
% floormax = 1.5/(NI);
flr = abs(min(abs(data),2))*0.25;%

sd = flr;

% Set target misfit

%Leave all weights to 1 for now
%     save([work_dir '\w_iter' num2str(ii) '.con'],'-ascii','w')

% Run the inversions
pred = run_EM1DTM_fwd(work_dir,meshfile,data,Q,txloc,txheight,rxloc,rxheight,sd,'dbzdt',tc, waveform , models_in);

save([work_dir '\PRED_1D_DIGHEM_TD.dat'],'-ascii','pred');

%% Plot obs vs pred
figure; 

pos_data = data; pos_data(data<0) = NaN;
neg_data = data; neg_data(data>0) = NaN;

count = 1;
for ii = 1 : 2 : ntc
subplot(2,ceil(ntc/4),count)    
plot(xyz(:,1) , (pos_data(:,ii)) , ':','LineWidth',2) ; hold on
plot(xyz(:,1) , (neg_data(:,ii)) , 'r:','LineWidth',2) ; hold on
plot(xyz(:,1) , -(pred(:,ii)) ,'k','LineWidth',2) ; hold on
title(['\bfT' num2str(ii) ' :' num2str(tc(ii)*1e6) ' \mu s'])
xlim([limits(1) limits(2)])
grid on
count = count+1;
xlabel('Easting (m)')
end


legend('VTEM (+)dB/dt','VTEM (-)dB/dt','DIGHEM FWR model')  
% figure; 
% for ii = 7 : 13
%     
% semilogy(xyz(data(:,ii)>0,1) , abs(data(data(:,ii)>0,ii)) , 'b:') ; hold on
% semilogy(xyz(data(:,ii)<0,1) , abs(data(data(:,ii)<0,ii)) , 'r:') ; hold on
% semilogy(xyz(:,1) , abs(pred(:,ii)) * NI,'k') ; hold on
% 
% end
%%
% figure;
% loglog(tc,-pred(94,:),'r');hold on
% loglog(tc,abs( data(94,:)));

