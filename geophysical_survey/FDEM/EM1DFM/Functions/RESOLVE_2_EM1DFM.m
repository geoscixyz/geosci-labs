% function [data_out,stnxyz] = rawdata_2_EM1DFM(rawdata,freqin,radius,limits)
% DEV ONLY
% Temporary function to format data to GIFTool format
% INPUT:
% rawdata:  Matlab array for data
% frequin:  Unique frequencies
% radius:   Minimum distance between points. Used to filter out data.
%
% OUTPUT:
% data:     Cell array of data with format
%
% {1} tr_specs(5 x ndatum) transmitter parameters [dx dy -dz mom_t axis]
% {2} rc_specs(5 x ndatum) receiver parameters [dx dy -dz mom_r axis]
% {3} freq    (1 x ndatum) vector of frequency for each datum
% {4} ontype  (1 x ndatum) Type of nomr
% {5} octype  (1 x ndatum) 'b' for both, 'i' for in-phase or 'q' for quadrature
% {6} utype   (1 x ndatum) 'v' for absolute uncertainty or 'p' for percentage
% {7} obs     (2 x ndatum)  or (1 x ndatum) depending on octype
% {8} uncert  (2 x ndatum)  or (1 x ndatum) depending on octype
% (9) station (3x ndatum)  Station ID#, X ,Y coordinate 
% MUST BE REFORMATED FOR EVERY INPUT FILE TYPE
% 
% Written by: D.Fournier
% Last update: 2014-02-27

%% FOR DEV ONLY
clear all
close all

work_dir= 'C:\Users\dominiquef.MIRAGEOSCIENCE\ownCloud\Research\Bookpurnong\Data';
freqin = [396 1773 3247 8220 39880 132700];
rawdata= [work_dir '\Raw_XY_Data.csv'];
radius = 10;

%% SCRIPT STARTS HERE
struct_in = importdata(rawdata);

data = struct_in.data;

% Number of observation stations
nstn = size(data,1);

% Number of frequencies
nfreq = length(freqin); % Hard coded ... file specific

% Total number of data
ndat = nstn * nfreq;



% Re-format to have all frequencies on single column and assign a code if
% transmitter dipole is oriented (ot) along X(1) , Y(2) or Z(3) axis
% Memory allocation
tx    = zeros(ndat,3); % transmitters offset [dx dy -z mom_t axis]
rx    = zeros(ndat,3); % receivers [dx dy -z mom_r axis]
freq        = zeros(ndat,1);
ontype      = ones(ndat,1);
octype      = [];    %'b' 'i' or 'q';
utype       = [];    %'v' or 'p';
obs         = zeros(ndat,2);
uncert      = zeros(ndat,2);
stn_num = zeros(ndat,1); % Keep track of station number

count = 1;
for ii = 1 : nstn
    
    for jj = 1 : nfreq
        

        tx(count,1:4)   = [7.91 0 -data(ii,6) 1];    

        
        rx(count,1:4)   = [0 0 -data(ii,6) 1];
        
        % Last two frequencies are co-axial (x=1), and only have inphase
        if jj == 3
            
            rx(count,5)     = 1;
            tx(count,5)     = 1;
            octype{count}   = 'i';
            obs(count,1:2)    = [data(ii,jj*2+5) data(ii,jj*2+6)];
            freq(count,1)   = freqin(jj);
        % First three frequencies are co-planar (z=3)
        else
           
            rx(count,5)= 3;
            tx(count,5)= 3;
            octype{count,1}  = 'b';
            obs(count,1:2)   = [data(ii,jj*2+5) data(ii,jj*2+6)];
            freq(count,1)         = freqin(jj);
            
        end
            
        
        ontype(count,1)      = 1;
        utype{count,1}       = 'v';
        stn_num(count,1:3)      = [data(ii,1) data(ii,2) ii];
        
        count = count + 1;
        
  
        
    end   
    
end
pct = 0.05;
flr_in   = 1;%0.05*std(obs(:,1));
flr_quad = 1;%0.05*std(obs(:,2));

uncert(:,1) = pct*abs(obs(:,1)) + flr_in;
uncert(:,2) = pct*abs(obs(:,2)) + flr_quad;

% uncert(freq(:,1)==900,1:2)=1;
% uncert(freq(:,1)==7200,1:2)=1;
% uncert(freq(:,1)==56000,1:2)=4;
%% Filter data if radius~=0
figure; scatter(stn_num(:,1),stn_num(:,2));title('Before sorting');hold on

if radius~=0
    
    [~,index] = unique(stn_num(:,3),'stable');
    stnxyz = stn_num(index,:);
    mask = ones(nstn,1);
    
    
    for ii = 1 : size(stnxyz,1)
        
        if mask(ii)==1
    %         temp = zeros(length(rc_specs),1); temp(ii:ii+nfreq-1)=1;
%             temp = stn_num(:,1)==stn(ii,1);
            r = ( (stnxyz(ii,1) - stnxyz(:,1)).^2 +...
                (stnxyz(ii,2) - stnxyz(:,2)).^2 ) .^0.5;

            % Only keep the curretn points and neighbours at distance r+
            mask(r <= radius) = 0;
            mask(ii) = 1;
            
        end

    end
    
    
    % Impose global XY limits
    if isempty(limits)==0

        temp = stnxyz(:,1) >= limits(1,1) & stnxyz(:,1) <= limits(2,1) &...
            stnxyz(:,2) >= limits(1,2) & stnxyz(:,2) <= limits(2,2);

    end
    
    mask = mask .* temp;

    stnxyz = stnxyz(mask==1,:);

    % Unpack lofical index to all data
    logic = stnxyz(:,3);
    mask = zeros(ndat,1);
    
    for ii = 1 : length(logic)
                
        mask(stn_num(:,3)==logic(ii)) = 1;
        
    end
    
end


    
% stnxyz = stnxyz(mask==1,:);
%%
rx= rx(mask==1,:);
tx= tx(mask==1,:);
freq    = freq(mask==1);
ontype  = ontype(mask==1);   
octype  = octype(mask==1);   
utype   = utype(mask==1);
obs     = obs(mask==1,:);
uncert  = uncert(mask==1,:);
stn_num = stn_num(mask==1,:);        
        
%% Create output data array       
data_out{1} = tx;
data_out{2} = rx;
data_out{3} = freq;
data_out{4} = ontype;
data_out{5} = octype;
data_out{6} = utype;
data_out{7} = obs;
% data_out{7}(data_out{7}(:,1)<0,1) = 0 ;
% data_out{7}(data_out{7}(:,2)<0,2) = 0 ;
data_out{8} = uncert;
data_out{9} = stn_num;

scatter(stn_num(:,1),stn_num(:,2),'r*');title('After sorting')