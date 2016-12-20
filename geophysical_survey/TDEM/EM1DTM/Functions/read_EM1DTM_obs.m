function data = read_EM1DTM_obs(obsfile)
% Load observation file for UBC EM1DFM
%
% INPUT
% obsfile : Observation file in UBC-EM1DFM format
% 
% OUTPUT: data structure
% data{1}{tx} : transmitters location [x, y, -dz]
% data{2}{tx} : transmitter loop [nsegs, x1, y1, x2, y2, ..., xn, yn ]
% data{3}{tx} : waveform
% data{4}{tx} : receivers specs [nrx, dtype (1: usec, 2: msec, 3: sec]
% data{5}{tx}{rx}{1} : receivers moment
% data{5}{tx}{rx}{2} : receivers x, y, -dz] 
% data{5}{tx}{rx}{3} : axis {x | y | z}
% data{5}{tx}{rx}{4} : ntimes, ontype ontype = 1: uV, 2: mV, 3: V, 4: nT, 5: uT, 6: mT
% data{5}{tx}{rx}{5} : times, #sweep
% data{5}{tx}{rx}{6} : utype (p: %, v: absolute)
% data{5}{tx}{rx}{7} : obs, uncert


%% FOR DEV ONLY
% clear all
% close all
% work_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Google Drive\Tli_Kwi_Cho\Modelling\Inversion\EM\VTEM\1D';
% obsfile = 'VTEM_TKC_Sub_RAW.obs';

%% SCRIPT STARTS HERE
%% Load Horizontal dipole experiment

fid=fopen(obsfile,'rt');    

line=fgets(fid); %gets next line
nsnds = str2num(line); % number of soundings

data = [];
% Loop over all the soundings
count = 1;
for tx = 1:nsnds
    % Read next line
    temp = str2num(fgets(fid));
    
    data{1}{tx,1}  = temp;    % X,Y,Z coordinate of transmiters
   
    data{2}{tx,1}    = str2num(fgets(fid)); % Tx loop locations
    
    data{3}{tx,1}      = fgets(fid);   % Waveform file
    
    data{4}{tx,1}  = str2num(fgets(fid)); % nrx, dtype
    
    temp = strsplit( strtrim( fgets( fid ) ), ' ' );
    
    for rx = 1 : data{4}{tx}(1)
        
        data{5}{tx,1}{rx}{1}    = str2num(temp{1}); % receivers [mom_r, x, y, -dz]
        data{5}{tx,1}{rx}{2}    = [str2num(temp{2}) str2num(temp{3}) str2num(temp{4})];
        data{5}{tx,1}{rx}{3}    = temp{5};
        data{5}{tx,1}{rx}{4}    = [str2num(temp{6}) str2num(temp{7})];
        
        for tt = 1 : data{5}{tx}{rx}{4}(1)
            
            temp = strsplit( strtrim( fgets( fid ) ), ' ' );
            
            data{5}{tx,1}{rx}{5}(tt,:)    = [str2num(temp{1}) str2num(temp{2})];
            data{5}{tx,1}{rx}{6}(tt)      = temp(4);
            data{5}{tx,1}{rx}{7}(tt,:)    = [str2num(temp{3}) str2num(temp{5})];
            
        end
        
    end
    
end

fclose(fid);

