function write_EM1DTM_obs(obsfile,data,index)
% WRITEEM1DFMOBS(rootname,data,varargin)
% Creates observation file in UBC format for the 1D inversion of frequency
% domain airborn data. 
% 
% INPUTS:
%
% Data matrix
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


%% Write obs for all stations\
nstn = size(data{1},1);

fid = fopen(obsfile,'wt');
fprintf(fid,'%i\n',nstn);
   
%std_floor = 0.01*max(abs(data),[],1);

for ii = 1 : nstn

    
    fprintf(fid,'%f %f %f\n',data{1}{ii}(1:3));
    
    % Write transmitter loop segments
    fprintf(fid,'%i ',data{2}{ii}(1));
    for jj = 2 : size(data{2}{ii},2)
        
        fprintf(fid,'%12.8e ',data{2}{ii}(jj));
        
    end
    fprintf(fid,'%f ',data{1}{ii}(3));
    
    fprintf(fid,'\n');
    
    % Write waveform file
    fprintf(fid,'%s \n',strtrim(data{3}{ii}));
    
    
    % Write number of receivers and data units flag
    fprintf(fid,'%i %i \n',data{4}{ii}(:));
    
    % For each receiver, write data
    for jj = 1 : data{4}{ii}(1)
        
        fprintf(fid,'%i ',data{5}{ii}{jj}{1});
        fprintf(fid,'%f %f %f ',data{5}{ii}{jj}{2}(:));
        fprintf(fid,'%s ',data{5}{ii}{jj}{3});
        fprintf(fid,'%i %i \n',data{5}{ii}{jj}{4}(:));
        
        for kk = 1 : data{5}{ii}{jj}{4}(1)
            fprintf(fid,'%e %i ',data{5}{ii}{jj}{5}(kk,:));
            fprintf(fid,'%e ',data{5}{ii}{jj}{7}(kk,1));
            fprintf(fid,'%s ',data{5}{ii}{jj}{6}{kk});
            fprintf(fid,'%e \n',data{5}{ii}{jj}{7}(kk,2));
        end
        
    end

%     fprintf(fid,'\n');
    
end

fclose(fid);