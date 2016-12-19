function data_out = convert_E3D_2_EM1D(trx,d,topo, radius)
% Use data from E3D file format and convert to 1D data structure
%
% INPUT
% tx = cell-array of transmitter information
% d = cell-array of data information
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


%% FOR DEV ONLY
% clear all
% close all
% work_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Google Drive\Tli_Kwi_Cho\Modelling\Inversion\EM\VTEM\1D';
% obsfile = 'VTEM_TKC_Sub_RAW.obs';

%% SCRIPT STARTS HERE
%% Assume Z-loop only
% nseg = 24; % Number transmitter segments
% nsnds =  % number of soundings

% Create topography surface to extract dz
F = scatteredInterpolant(topo(:,1),topo(:,2),topo(:,3));

data = [];

freqin = unique(d(:,1));

% Number of frequencies
nfreq = length(freqin); % Hard coded ... file specific

% Total number of data
ndat = size(trx,1);

nstn = ndat;


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

% Keep track of new unique stations
stnID = 1;

for ii = 1 : ndat
        
    zloc = F(trx(ii,1:2)) - trx(ii,3);
    dxy = trx(ii,1:2) - d(ii,2:3);

%         if jj > 1
    tx(ii,1:4)   = [dxy zloc 1];
%         else
%             tx(ii,1:4)   = [d(ii,2:3) zloc 1];    
%         end

    rx(ii,1:4)   = [0 0 zloc 1];

    % Last two frequencies are co-axial (x=1), and only have inphase
%         if jj == 4 || jj == 5
%             
%             rx(ii,5)     = 1;
%             tx(ii,5)     = 1;
%             octype{ii}   = 'i';
%             obs(ii,1)    = data(ii,jj+12);
%             freq(ii,1)   = freqin(jj);
    % First three frequencies are co-planar (z=3)
%         else

    rx(ii,5)= 3;
    tx(ii,5)= 3;
    octype{ii,1}  = 'b';
    obs(ii,1:2)   = [d(ii,end-3) d(ii,end-1)];
    freq(ii,1)         = d(ii,1);

%         end


    ontype(ii,1)      = 1;
    utype{ii,1}       = 'v';
    uncert(ii,:)     = [d(ii,end-2) d(ii,end)];
    
    if ii > 1
        
        if any(trx(ii,1:2) ~= trx(ii-1,1:2))
            
            stnID = stnID+1;
            
        end
        
    end
    stn_num(ii,1:3)      = [trx(ii,1:2) stnID];

end
pct = 0.05;
flr_in   = 1;%0.05*std(obs(:,1));
flr_quad = 1;%0.05*std(obs(:,2));

% uncert(:,1) = pct*abs(obs(:,1)) + flr_in;
% uncert(:,2) = pct*abs(obs(:,2)) + flr_quad;

% uncert(freq(:,1)==900,1:2)=1;
% uncert(freq(:,1)==7200,1:2)=1;
% uncert(freq(:,1)==56000,1:2)=4;
%% Filter data if radius~=0
figure; scatter(stn_num(:,1),stn_num(:,2));title('Before sorting');hold on


    
[~,index] = unique(stn_num(:,3),'stable');
stnxyz = stn_num(index,:);
mask = ones(nstn,1);
    
if radius~=0    
    
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



