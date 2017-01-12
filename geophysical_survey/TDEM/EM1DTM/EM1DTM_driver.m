% FOR DEV ONLY
% Temporary function to format data to GIFTool format
% MUST BE REFORMATED FOR EVERY FILE

clear all
close all

addpath '.\Functions';

work_dir    = '.';

dsep = '\';

%% Load input file
[meshfile,obsfile,topofile,nullfile,m_con,con_ref,alpha_con,beta,cooling,target,bounds,mtype,interp_n,interp_r,interp_s] = EM1DTM_read_inp([work_dir dsep 'EM1DTM_LC.inp']);

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




%% Load data in UBC format

[trx,d] = read_E3D_obs([work_dir dsep obsfile]);
data = convert_E3D_2_EM1D(trx,d,topo);
write_EM1DTM_obs([work_dir dsep 'EM1DTM.obs'],data,[])

% data = read_EM1DTM_obs([work_dir dsep obsfile]);

tc = data{5}{1}{1}{5}(:,1);

%load([work_dir '\dtbs'])
%data = dtbs; clear dtbs

nstn = size(data{1},1);
% Create beta vector
beta = ones(nstn,1)*beta;



%% VTEM SURVEY
% NEED TO WRITE THE I/O FOR TIME-DOMAINE DATA
% load([work_dir '\VTEM_data_DF'])
% load([work_dir '\VTEM_xyz_DF'])
% load([work_dir '\VTEM_tc_DF'])
% load([work_dir '\VTEM_Waveform'])
% 
% Load XYZ data
% dfile = load([work_dir dsep obsfile]);
% indx = dfile(:,3) > 0;
% data = dfile(indx,4:end);
% xyz = dfile(indx,1:3);
% tc = [120 141 167 198 234 281 339 406 484 573 682 818 974 1151 1370 1641 1953 2307 2745 3286 3911 4620 5495 6578]*1e-6;
% % 
% % % % Number of time channels to skip
% % % early_tc = 0;
% % % late_tc = 1;
% radius = 13;
% 
% pT = 1e-12;
% A = pi*radius^2;
% data = (data)*pT*A;
% 
% floor = repmat(std(data),size(data,1),1);
% pc_err= 0.05;
% write_h3d_obs([work_dir dsep 'VTEM_FLIN_h3d.dat'],data,xyz,tc,radius,floor,pc_err)


%% Load E3D-TD data
% [trx,d] = read_E3D_obs([work_dir dsep 'VTEM_FLIN_h3d.dat']);

% Convert E3D format to data matrix (assume Z only for now)
% data = convert_E3D_2_EM1D(trx,d,topo);


%% AEROTEM SURVEY
% load([work_dir '\ATEM_data_DF'])
% load([work_dir '\ATEM_xyz_DF'])
% load([work_dir '\ATEM_tc_DF'])
% load([work_dir '\ATEM_Waveform'])
% 
% % Number of time channels to skip
% early_tc = 0;
% late_tc = 0;
% radius = 2.5;
% % 
% nT = 1e-09;
% A = pi*radius^2;
% NI = 250*8;
% %data = -(data) * nT / NI;
% 
% % Change uncertainties
% for ii = 1 : size(data{5},1)
%     
%     data{5}{ii}{1}{7}(:,1) = data{5}{ii}{1}{7}(:,1) * nT * A / NI;
%     data{5}{ii}{1}{7}(:,2) = abs(data{5}{ii}{1}{7}(:,1))*0.1+1e-15 ;
%     
% end

%% Apply correction

% data = data(:, (1+early_tc) : (end - late_tc) );
% tc = tc((1+early_tc) : (end - late_tc));
% ntc = length(tc);
% 
% std_data = std(data,1);

%% STACK DATA

%% OR DOWNSAMPLE ALONG LINE


nstn = size(data{1},1);
xyz = [data{1}{:}];
xyz = reshape(xyz,3,nstn)';
% 
% [data,xyz] = sort_EM1DTM(data,xyz,[],20,[],[]);
% 
% %% Get line numbers
% lineID = xy_2_lineID(xyz(:,1),xyz(:,2));
% line = unique(lineID);


%% Write EM1DTM data to file
% obsfile = 'VTEM_FLIN_flt20m.obs';
% write_EM1DTM_obs([work_dir dsep obsfile],data,[])

%% Write to H3D data format
% write_h3d_obs([work_dir '\VTEM_AVG_h3d.obs'],data,xyz,tc,radius,uncert,0.05);

%% Plot all data lines
% count = 1;
% 
% for ii = 1 : length(line)
%     set(figure(ceil(ii/4)+1), 'Position', [0 0 2000 1000])
%     subplot(4,1,count)
%     for jj = 1 : 2 : ntc
%         
%         semilogy(xyz(lineID==line(ii),1),data(lineID==line(ii),jj),:); hold on
%         
%     end
%     
%     count = count+1;
%     
%     if count==5
%         
%         count =1;
%         
%     end
%     
%     
% end


%% Create interpolation (P) matrix
nnodes  = 8; % Number of nearest neighbours to interpolate with
% P = make_EM1D_P_3D(work_dir,meshfile,Q,nnodes,15,100,1,2);
% save([work_dir '\P'],'P')
% load([work_dir '\P']);

fprintf('Creating Interpolation matrix\n')
fprintf('This may take a while\n')
Q = make_EM1D_Q_3D(work_dir,meshfile,nullcell,xyz);

[P,W,indx] = make_EM1D_P_3D(work_dir,meshfile,Q,interp_n,interp_r,interp_s,alpha_con(2),alpha_con(3));


%% RUN 1D INVERSION

%% Create transmiter loop (Seogi's func)
% Create database from raw file
% dtbs = [];
% ntc = length(tc);
% 
% for ii = 1 : size(data,1)
%     
%     dtbs{1,1}{ii,1} = [xyz(ii,1) xyz(ii,2) -xyz(ii,4)];
%     
%     [x, y] = circfun(xyz(ii,1), xyz(ii,2), 2.5, 24);
%     
%     dtbs{1,2}{ii,1}(1) = 24;
%     for jj = 1 : 24
%         
%         dtbs{1,2}{ii,1}(end+1:end+2) = [x(jj) y(jj)];
%         
%     end
%     
%     dtbs{1,3}{ii,1} = 'em1dtm.wf';
%     dtbs{1,4}{ii,1} = [1,3];
%     dtbs{1,5}{ii,1}{1}{1} = 1;
%     dtbs{1,5}{ii,1}{1}{2} = [xyz(ii,1) xyz(ii,2) -xyz(ii,4)];
%     dtbs{1,5}{ii,1}{1}{3} = 'z';
%     dtbs{1,5}{ii,1}{1}{4} = [ntc 3];
%     dtbs{1,5}{ii,1}{1}{5} = [tc' ones(ntc,1)];
%     
%     for jj = 1 : ntc
%         
%         dtbs{1,5}{ii,1}{1}{6}(jj) = 'v';
%         
%     end
%     
%     
%     dtbs{1,5}{ii,1}{1}{7} = [data(ii,:)' abs(data(ii,:)'*0.1 + 1e-5 )];
%     
%     
% end
    
    

% txloc = [x(:), y(:)];
% rxloc = [0 0];

% wf = waveform, can be 'STEPOFF' for stepoff or a number for RAMP or
% two-column for a discretized waveform (time in second)
% ta = 5.5*1e-4;
% tb = 1.1*1e-3;
% twave = linspace(0., tb, 2^7+1) ;
% wfval = [trifun(twave, ta, tb); 0 ];

% twave = [twave 0.0026];

% waveform = [twave(:) , wfval(:) ]; 

% Input#7
% tc = vector for time channels 

%% Run FWR Calculations
%pred = run_EM1DTM_fwd(work_dir,meshfile,data,m_con,Q);


    
%% Setup inversion
% Pre-set average misfit
phid_all = [99999 99999];
phid = ones(size(Q,1),1)*99999;

% Count iterations
ii = 1;
max_iter = 15;

% set(figure, 'Position', [25 50 1800 900])

while  phid_all(end) < phid_all(end-1) || ii < 10%ii <= max_iter 

    % Leave cell weights to unity (can be manually changed)
    w=ones(mcell,1);
    
    % Run the inversions
    [m_con1D,m_misfit,phid,phim,beta,pred,bHSpace] = run_EM1DTM_inv(work_dir,meshfile,data,m_con,con_ref,alpha_con([1,4]),w,Q,phid,beta,cooling,target,ii,HSflag);
        
    m_con = interp1D_to_3D(m_con1D,P,W,indx);
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

    save([work_dir dsep 'Inv_MOD_iter' num2str(ii) '.con'],'-ascii','m_con');

    write_EM1DTM_obs([work_dir dsep 'Inv_PRED_iter' num2str(ii) '.pre'],pred,[])
        
    % Save misfit map
%     save([work_dir '\Misfit.con'],'-ascii','m_misfit')
    
    % Compute final data misfit
%     phid_all(ii) = sum( ((data{7}(:,1) - pred(:,5)) ./ (data{8}(:,1))).^2 +...
%         ( (data{7}(:,2) - pred(:,6)) ./ (data{8}(:,2)) ).^2 );
    
    phid_all(ii) = sum(phid);

    
    
    ii = ii + 1;

    
 
end

ii = ii - 1;
%% Output obs data back to E3D predicted format

% Load EM1D obs file
%tc = d{1}{1}(:,4);
convert_EM1D_2_E3D_pred([work_dir dsep 'EM1DTM.obs'],tc,[work_dir dsep 'Obs1D_2_E3Dpre.pre']);

convert_EM1D_2_E3D_pred([work_dir dsep 'Inv_PRED_iter' num2str(10) '.pre'],tc,[work_dir dsep 'Pre1D_2_E3Dpre.pre']); 

%% Plot time channels
dobs = load([work_dir dsep 'Obs1D_2_E3Dpre.pre']);
dpre = load([work_dir dsep 'Pre1D_2_E3Dpre.pre']);

% Find non-zero time channels
index = zeros(1,length(tc));
for ii = 1 : length(tc)
    
    % Only plot 
    if sum(~isnan(dobs(dobs(:,4)==tc(ii),end)))~=0
        
        index(ii) = 1;
        
    end
    
end

ntimes = sum(index);
index = find(index);

set(figure(1), 'Position', [25 50 1800 900])
xmin = min(dobs(:,1));
xmax = max(dobs(:,1));

ymin = min(dobs(:,2));
ymax = max(dobs(:,2));

dx = 10;
dy = 10;

x = xmin:dx:xmax;
y = ymin:dy:ymax;
[Y,X] = ndgrid(y,x);

Y = flipud(Y);




for ii = 1 : ntimes
    
    subdata = dobs( dobs(:,4) == tc(index(ii)),: );
    subpred = dpre( dpre(:,4) == tc(index(ii)),: );
    
    
%     
%     F_o = scatteredInterpolant(subdata(:,1),subdata(:,2),abs(log10(subdata(:,end))),'linear');
%     F_p = scatteredInterpolant(subpred(:,1),subpred(:,2),abs(log10(subpred(:,end))),'linear');
%     
%     data_2D = F_o(Y,X);
%     pred_2D = F_p(Y,X);
    
    data_2D = griddata(subdata(:,2),subdata(:,1),((subdata(:,end))),Y,X);
    pred_2D = griddata(subpred(:,2),subpred(:,1),((subpred(:,end))),Y,X);
    
    subplot(2,ntimes,ii)
    h = imagesc(x,y,data_2D);
    set(h,'alphadata',~isnan(data_2D));
    colormap(jet)
    title(['\bf T:' num2str(tc(index(ii)))])
    caxis([min(data_2D(:)) max(data_2D(:))])
    axis equal tight
    set(gca,'YTickLabel',[])
    
    subplot(2,ntimes,ii+ntimes)
    h = imagesc(x,y,pred_2D);
    set(h,'alphadata',~isnan(pred_2D));
    colormap(jet)
    title(['\bf Pred:' num2str(tc(index(ii)))])
    caxis([min(data_2D(:)) max(data_2D(:))])
    axis equal tight
    set(gca,'YTickLabel',[])
end
%% Plot obs vs pred interpolated in 2D

for ii = 1 : 5 : ntc-2
    set(figure, 'Position', [0 0 2000 1000])
    if ii == 1
        % Set coordinates for plot
        xmin = min(xyz(:,1));
        xmax = max(xyz(:,1));

        ymin = min(xyz(:,2));
        ymax = max(xyz(:,2));

        dx = 10;
        dy = 10;

        x = xmin:dx:xmax;
        y = ymin:dy:ymax;
        [Y,X] = ndgrid(y,x);

        Y = flipud(Y);
    end
    
    F_d = TriScatteredInterp(xyz(:,2),xyz(:,1),log10(data(:,ii)),'linear');
    F_p = TriScatteredInterp(xyz(:,2),xyz(:,1),log10(-pred(:,ii)),'linear');
    
    data_2D = F_d(Y,X);
    pred_2D = F_p(Y,X);
    
    subplot(1,3,1)
    imagesc(y,x,data_2D);
    title(['\bf LOG Observed Time: ' num2str(tc(ii))]);
%     caxis([min(data(:,ii)) max(data(:,ii))]);
    colorbar
    
    subplot(1,3,2)
    imagesc(y,x,pred_2D);
    title(['\bf LOG Predicted : ' num2str(tc(ii))]);
%     caxis([min(data(:,ii)) max(data(:,ii))]);
    colorbar
    
    subplot(1,3,3)
    imagesc(y,x,data_2D-pred_2D);
    title(['\bf LOG Residual Time: ' num2str(tc(ii))]);
%     caxis([min(data(:,ii)) max(data(:,ii))]);
    colorbar
    
end   
