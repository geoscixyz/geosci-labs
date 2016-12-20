function data = convert_E3D_2_EM1D(trx,d,topo)
% Use data from E3D file format and convert to 1D data structure
%
% INPUT
% tx = cell-array of transmitter information
% d = cell-array of data information
% 
% OUTPUT: data structure
% data{1}{tx} : transmitters location [x, y, -dz]
% data{2}{tx} : transmitter loop [nsegs, x1, y1, x2, y2, ..., xn, yn ]
% data{3}{tx} : waveform
% data{4}{tx} : receivers specs [nrx, dtype (1: usec, 2: msec, 3: sec]
% data{5}{tx}{rx}{1} : receivers moment
% data{5}{tx}{rx}{2} : rx[x, y, -dz] 
% data{5}{tx}{rx}{3} : axis {x | y | z}
% data{5}{tx}{rx}{4} : ntimes, ontype = 1: uV, 2: mV, 3: V, 4: nT, 5: uT, 6: mT
% data{5}{tx}{rx}{5} : times, #sweep
% data{5}{tx}{rx}{6} : utype (p: %, v: absolute)
% data{5}{tx}{rx}{7} : obs, uncert


%% FOR DEV ONLY
% clear all
% close all
% work_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Google Drive\Tli_Kwi_Cho\Modelling\Inversion\EM\VTEM\1D';
% obsfile = 'VTEM_TKC_Sub_RAW.obs';

%% SCRIPT STARTS HERE
%% Assume Z-loop only
nseg = 24; % Number transmitter segments
nsnds = size(trx,1); % number of soundings

if ~isempty(topo)
    % Create topography surface to extract dz
    F = scatteredInterpolant(topo(:,1),topo(:,2),topo(:,3));
end

data = [];

% Loop over all the soundings
for tx = 1:nsnds
    
    % If no topo, then assume flat 0 offset
    if ~isempty(topo)
        % Query the topo at the trx location    
        ztopo = F(trx{tx}(1),trx{tx}(2));

        data{1}{tx,1}  = [trx{tx}(1:2) ztopo - trx{tx}(3)];    % X,Y,Z coordinate of transmiters
        
    else
        
         data{1}{tx,1}  = [trx{tx}(1:2) 0.0];    % X,Y,Z coordinate of transmiters
         
    end
    % Generate loop segments
    [x, y] = circfun(0, 0, trx{tx}(4), nseg);
    data{2}{tx,1}    = nseg; % Tx loop locations
    
    for jj = 1 : nseg
        
        data{2}{tx,1}(end+1:end+2) = [x(jj) y(jj)];
        
    end
      
    data{3}{tx,1}      = 'em1dtm.wf';   % Waveform file
    
    data{4}{tx,1}  = [size(d{tx},1) 3]; % nrx, dtype always in sec.
    
    for rx = 1 : data{4}{tx}(1)
        
        
        
        data{5}{tx,1}{rx}{1}    = 1; % Transmitter moment always unity
        
        if ~isempty(topo)
            ztopo = F(d{tx}{rx}(1,1),d{tx}{rx}(1,2));
            data{5}{tx,1}{rx}{2}    = [trx{tx}(1:2)-d{tx}{rx}(1,1:2) ztopo-d{tx}{rx}(1,3)]; % receivers [mom_r, x, y, -dz]
        else
            
            data{5}{tx,1}{rx}{2}    = [trx{tx}(1:2)-d{tx}{rx}(1,1:2) 0]; % receivers [mom_r, x, y, -dz]
            
        end
            
        data{5}{tx,1}{rx}{3}    = 'z'; % assume z-component only
        data{5}{tx,1}{rx}{4}    = [size(d{tx}{rx},1) 3];
        
        for tt = 1 : data{5}{tx}{rx}{4}(1)
            
            data{5}{tx,1}{rx}{5}(tt,:)    = [d{tx}{rx}(tt,4) 1];
            data{5}{tx,1}{rx}{6}{tt}      = 'v';
            data{5}{tx,1}{rx}{7}(tt,:)    = [d{tx}{rx}(tt,21:22)];
            
        end
        
    end
    
end




