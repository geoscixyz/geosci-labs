% GIF tools script for the conversion of 
%
% Read in data from the ASCII file (TKC_DIGHEM_ASCII.dat) and set up a 
% GIFtools project : DIGHEMproj.mat for the DIGHEM data. 
%
% To launch GUI with the project...
%  clear all
%  load DICHEMproj
%  GIFtools(obj)
%
% 
% Base script by: Lindsey Heagy (April 26, 2014)
% Adapted by: Dominique Fournier
% Last update: September 18th, 2014


clear all
close all
clc

% path to GIFtools
addpath 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIFtools'
GIFtools_addPath

%% USER INPUT PARAMETERS

work_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Google Drive\Tli_Kwi_Cho\Modelling\In';

obsfile = 'DIGHEM_line10.obs';
topofile = 'CDED_076c05_NAD27.topo';


% Name of project to save
% svtitle = '.\EMproj.mat';


%% Start GIFTools Project
% EMproj = GIFproject();


%% Create data structure
% d_E3D = FEMdata(EMproj);

% Path to DIGHEM ASCII FILE
% datahandle = '\..\..\Datasets\DIGHEM\TKC_DIGHEM_ASCII.dat';
% FEMdata.importGeosoftXYZ(EMproj,[EMproj.getWorkDir,datahandle]);
% clear datahandle

%% Sort Data
% grab FEM data object 
% item = EMproj.getItem(1);

%% Load topo and grid for referencing Z on em1d data
% TOPOdata(EMproj);

% topo = EMproj.getItem(2); 
% topo.importXYZ([work_dir '\' topofile]);

% grid Topo and get topo at our measurement points
gridTopo = TriScatteredInterp( topo.getData(:,1), topo.getData(:,2), topo.getData(:,3) );
% Topo     = gridTopo([item.getData(:,'X_NAD27') item.getData(:,'Y_NAD27')]);

%% Load EM1D obs or pred
% Will have to run it in a LOOP if you have multiple seperated 1D files
%
% Data cell-arrays
% data{1} : transmitters offset [dx dy -dz mom_t axis]  axis--> x=1, y=2, z=3
% data{2} : receivers [dx dy -dz mom_r axis]  axis--> x=1, y=2, z=3
% data{3} : frequency 
% data{4} : ontype
% data{5} : octype = 'b' both, 'i' in-phase only, 'q' quadrature only
% data{6} : utype = 'v' absolute value of uncertainty, 'p' percentage
% data{7} : OBS:    col(1): Real , col(2): Imag
% data{8} : Uncert: col(1): Real , col(2): Imag
% data{9} : [sounding # , X , Y]

% All observations in file are appended in order they are encountered
% transmitters may be repeated several times since 1D data count over
% sounding and frequencies

data = load_EM1DFM_obs(work_dir,obsfile);

%% Reformat data in UTM
ndat = size(data{1},1);
freq = data{3}(:);

tx_spec = zeros(ndat,6); % [ X Y Z radius thete phi ]
rx_loc = zeros(ndat,3);

%[Hx_R uncert Hx_I uncert ... Hz_R uncert Hz_I uncert] 
% Set to NaN to differentiate with real 0
obs = zeros(ndat,12); 
obs(obs==0) = NaN;

for ii = 1 : size(data{1},1)
    
    tx_spec(ii,1:2) = data{9}(ii,1:2) + data{1}(ii,1:2);
    tx_spec(ii,3) = gridTopo(tx_spec(ii,1:2)) - data{1}(ii,3);
    tx_spec(ii,4) = data{1}(ii,4);
    
    % Set orientation of the LOOP [ angle from vertical, azimuth ]
    if data{1}(ii,5)==1 %If normal on x
        
        tx_spec(ii,5:6) = [90 90];
        
    elseif data{1}(ii,5)==2 %If normal on y
        
        tx_spec(ii,5:6) = [90 0];
        
    elseif data{1}(ii,5)==3 % Normal on z
    
        tx_spec(ii,5:6) = [0 0];
        
    end
    
    
    
    rx_loc(ii,1:2) = data{9}(ii,1:2) + data{2}(ii,1:2);
    rx_loc(ii,3) = gridTopo(rx_loc(ii,1:2)) - data{2}(ii,3);
    
    % Determine component (Hx(1:4), Hy(5:8), Hz(9:12))
    index = 4 * ( data{2}(ii,5) - 1 ) + 1;
    
    % Detect the type of data (Im,Real) and 
    if strcmp(data{5}{ii},'b')==1 %Both imaginary and real
        
        
        
        obs(ii,index) = data{7}(ii,1);
        obs(ii,index+2) = data{7}(ii,2);
        
        % Look at type of uncertainties if they exist
        if isempty(data{8}) == 0 
            
            if strcmp(data{6}{ii},'v')==1
        
                obs(ii,index+1) = abs( data{8}(ii,1) * data{7}(ii,1) );
                obs(ii,index+3) = abs( data{8}(ii,2) * data{7}(ii,2) );
                
            else
                
                obs(ii,index+1) = data{8}(ii,1);
                obs(ii,index+3) = data{8}(ii,2);
                
            end
            
        end
        
    elseif strcmp(data{5}{ii},'q')==1 %Only imaginary
                
        obs(ii,index+2) = data{7}(ii,2);
        
        % Look at type of uncertainties if they exist
        if isempty(data{8}) == 0 
            
            if strcmp(data{6}{ii},'v')==1

                obs(ii,index+3) = abs( data{8}(ii,2) * data{7}(ii,2) );
                
            else

                obs(ii,index+3) = data{8}(ii,2);
                
            end
            
        end
        
    elseif strcmp(data{5}{ii},'i')==1 %Both imaginary and real
        
        
        
        obs(ii,index) = data{7}(ii,1);

        
        % Look at type of uncertainties if they exist
        if isempty(data{8}) == 0 
            
            if strcmp(data{6}{ii},'v')==1
        
                obs(ii,index+1) = abs( data{8}(ii,1) * data{7}(ii,1) );

                
            else
                
                obs(ii,index+1) = data{8}(ii,1);

                
            end
            
        end
        
    end
         
        
    
    
end

%% Populate FEM object

% You technically have everything to create a FEM object after you loaded
% all the 1D files in a loop
%
% Use tx_spec, freq, rx_loc and obs
%
% d_E3D.setData(1,'FREQUENCY') = 2;



