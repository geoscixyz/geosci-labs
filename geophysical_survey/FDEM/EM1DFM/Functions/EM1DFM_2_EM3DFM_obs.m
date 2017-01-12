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

work_dir    = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Google Drive\Tli_Kwi_Cho\Modelling\Inversion\EM\DIGHEM\1D';

dsep = '\';

%% Load input file
obsfile = 'DIGHEM_TKC_1D_FULL.obs';
topofile = 'CDED_076c05_NAD27_LAKE_BATHY.topo';

%% Reformat data in array structure
% Last entry specifies the minimum distance between points
% freqin = [56000 7200 900];% 5001 901]; %DIGHEM
% % % freqin = [396 1773 3247 8220 39880 132700]; % RESOLVE
% limits(2,1:2) = [557800 7134100];
% limits(1,1:2) = [556800 7133200];
% % 
% % % Load raw for TKC
% [data,xyz] = rawdata_2_EM1DFM([work_dir '\DIGHEM_data'],freqin,1,limits);
% 
% % Load obs file in EM1DFM format
data = load_EM1DFM_obs(work_dir,obsfile);
xyz = [data{9}(1:3:end,1:2) -data{1}(1:3:end,3)];
% data{7} = abs(data{7});

nstn = size(xyz,1);


%% Change uncertainties
indx = data{3}(:)==56000;
data{8}(indx,1) = 8;%abs(data{7}(indx,1))*0.1 + 1;
data{8}(indx,2) = 8;%abs(data{7}(indx,2))*0.1 + 1;

indx = data{3}(:)==7200;
data{8}(indx,1) = 5;%abs(data{7}(indx,1))*0.1 + 1;
data{8}(indx,2) = 5;%abs(data{7}(indx,2))*0.1 + 1;

indx = data{3}(:)==900;
data{8}(indx,1) = 1;
data{8}(indx,2) = 1;%abs(data{7}(indx,2))*0.1 + 1;



%% Write lines of data
%Assign line number to data
lineID = xy_2_lineID(xyz(:,1),xyz(:,2));
line = unique(lineID);

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
 
topo = read_UBC_topo([work_dir dsep topofile]);

%Create topo surface
T = scatteredInterpolant(topo(:,1),topo(:,2),topo(:,3));

% Evaluate Z of stations
z = T(xyz(:,1),xyz(:,2)) + xyz(:,3);

%% Create matrices of tx and data
% tx(LOOP) [ X Y Z rad phi, theta, nrx]
% data [Freq X Y Z Ex UncEx ... Hz UncHz iHz UnciHz]
tx = [];
rx = [];
for ii = 1:size(data{1},1)
    
    indx = data{9}(ii,3);
    
    % Create transmitter loop
    tx(ii,:) = [xyz(indx,1:2) z(indx) 1 0 0 1];
    
    % Move receiver location depending on line orientation and frequency
    P1 = xyz(indx,1:2);
    if indx == nstn
        
        P1 = xyz(indx-1,1:2);
        P2 = xyz(indx,1:2);
        
    elseif lineID(indx)~=lineID(indx+1)

        P2 = xyz(indx+2,1:2);
        
    else
        
        P2 = xyz(indx+1,1:2);
        
    end
    
    % Compute unit directions
    dl = (P2 - P1) / norm(P2 - P1);
    
    % Offset the receiver
    rxLoc = xyz(indx,1:2) + dl*data{1}(ii,1);
    
    rx(ii,:) = [data{3}(ii) rxLoc z(indx) NaN(1,20) data{7}(ii,1) data{8}(ii,1) -data{7}(ii,2) data{8}(ii,2)];
end

%% Filter for negative in-phases
% indx = ones(size(rx,1),1);
% 
% for ii = 1 : size(rx,1)
%     
%     if rx(ii,25) <=0
%         
%         rx(ii,25) = NaN;
%             
%     end
%     
%     if rx(ii,27) >=0
%         
%         rx(ii,27) = NaN;
%             
%     end
%     
%     if isnan(rx(ii,[25 27]))
%         
%         indx(ii) = 0;
%         
%     end
%     
%     
% end
% 
% tx = tx(indx==1,:);
% 
% 
% rx = rx(indx==1,:);



%% Write E3D observation

write_e3d_obs([work_dir dsep 'DIGHEM_TKC_E3D_ppm.dat'],tx,rx)