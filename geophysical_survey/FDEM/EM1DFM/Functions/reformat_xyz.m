clear all
close all

addpath '.\Functions';

work_dir    = 'D:\Egnyte\Private\stanislawah\Projects\CMIC\Moidelling\DIGHEM_1D_inversion\3858_EM1DFM_TestLines\3858_EM1DFM\Data';

dsep = '\';
[meshfile,obsfile,topofile,nullfile,m_con,con_ref,m_sus,sus_ref,alpha,beta,cooling,target,bounds,mtype,interp_n,interp_r] = EM1DFM_read_inp([work_dir '\EM1DFM_LC.inp']);

data = load_EM1DFM_obs(work_dir,obsfile);
obsfile = 'Obs_L12490-L2570_filt_15m.obs';
writeem1dfmobs(work_dir,obsfile,data,'')


% Write data out X Y Z Freq In Quad Uncert_In Uncert_Quad
data_out = [data{9}(:,1:3) data{3} data{7}(:,1:2) data{8}(:,1:2)];
save([work_dir '\Inv_Data_XYZ_ds10.dat'],'-ascii','data_out');
