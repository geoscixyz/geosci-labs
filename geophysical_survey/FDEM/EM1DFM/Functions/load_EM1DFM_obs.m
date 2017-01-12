function data = load_EM1DFM_obs(work_dir,obsfile)
% Load observation file for UBC EM1DFM
%
% INPUT
% work_dir: directory of file
% obsfile : Observation file in UBC-EM1DFM format
% 
% OUTPUT: data structure
% data{1} : transmitters offset [dx dy -dz mom_t axis]  axis--> x=1, y=2, z=3
% data{2} : receivers [dx dy -dz mom_r axis]  axis--> x=1, y=2, z=3
% data{3} : frequency of observation
% data{4} : ontype
% data{5} : octype = 'b' both, 'i' in-phase only, 'q' quadrature only
% data{6} : utype = 'v' absolute value of uncertainty, 'p' percentage
% data{7} : datum (1): in-phase , datum(2): quadrature
% data{8} : uncertainty (1): in-phase , uncertainty(2): quadrature
% data{9} : [sounding # , X , Y]


%% FOR DEV ONLY
% clear all
% close all
% work_dir = 'C:\Users\dominiquef\Dropbox\DIGHEM\Processed_data\1DEMInversions\ALL_DF';
% obsfile = 'DIGHEM_TKC_ALL.obs';

%% SCRIPT STARTS HERE
%% Load Horizontal dipole experiment
utype = [];
uncert = [];
file = [work_dir '\' obsfile];
fid=fopen(file,'rt');    

line=strtrim(fgets(fid)); %gets next line
nsnds = str2num(line); % number of soundings

% Pre-allocate memory
Xsnd  = ones(nsnds,1);
Ysnd  = ones(nsnds,1);
nfreq = ones(nsnds,1);

% Loop over all the soundings
count = 1;
for ss = 1:nsnds
    % Read next line
    temp = str2num(fgets(fid));
    Xsnd(ss) = temp(1);    % X coordinate of transmiters
    Ysnd(ss) = temp(2);    % X coordinate of transmiters
    nfreq(ss) = temp(3);   % Number of frequencies per soundings
    
%     if ss==1
%         
%         freq = zeros(1,nfreq(ss));
% 
%         rx = zeros(nsnd,nfreq(ss),1,1,6); %Receiver array
% 
%     end
    
    % Loop over all frequencies
    for ff = 1:nfreq(ss)

        line=strtrim(fgets(fid)); %gets next line
        temp = str2num(line);

        freq(count,1) = temp(1);
        ntx = temp(2);

        % Loop over all transmitters
        % Pre-allocate memory
        tx_dpm = ones(ntx,1);       %Dipole moment of transmitter
        Z_tx = ones(ntx,1);         %Z location (negative up)

        for tt = 1:ntx

            line=strtrim(fgets(fid)); %gets next line
            temp = regexp(line,'\s+','split');
            temp = temp(1:end);
            
            % transmitters offset [dx dy -z mom_t axis]
            tx(count,4) = str2num(temp{1}); 
            tx(count,3) = str2num(temp{2});
            
            
            if strcmp(temp{3},'x')==1
                
               tx(count,5) = 1; 
               
            elseif strcmp(temp{3},'y')==1
                
               tx(count,5) = 2;  
               
            else
                
               tx(count,5) = 3; 
                
            end
                
%             tx_dpm(tt) = str2num(temp{1});  
%             Z_tx(tt) = str2num(temp{2});     
            nrx = str2num(temp{4});

            % Loop over all receivers
            % Pre-allocate memory
                        
            for rr = 1:nrx
                
                line=strtrim(fgets(fid)); %gets next line
                temp = regexp(line,'\s+','split');
                temp = temp(1:end);
                
                % transmitters offset [dx dy -z mom_t axis]
                tx(count,1) = str2num(temp{2});
                tx(count,2) = str2num(temp{3});
                rx(count,3) = str2num(temp{4});
                rx(count,4) = str2num(temp{1});
            
                if strcmp(temp{5},'x')==1
                
                    rx(count,5) = 1; 

                elseif strcmp(temp{5},'y')==1

                    rx(count,5) = 2;  

                else

                    rx(count,5) = 3; 

                end
            

                ontype(count,1)      = str2num(temp{6});

                octype{count,1} = temp{7};
                
                if strcmp(octype{count},'b')==1

                    obs(count,1) = str2num(temp{8});   % inphase
                    obs(count,2) = str2num(temp{9});   % quadrature                   
                    
                    % Load uncertainty if exists
                    if length(temp)> 9
                    utype{count,1} = temp{10};    
                    uncert(count,1) = str2num(temp{11});   % inphase
                    uncert(count,2) = str2num(temp{12});   % quadrature
                    end
                    
                else 
                    
                    if strcmp(octype{count},'i')==1
                        
                        obs(count,1) = str2num(temp{8});   % inphase
                        
                        % Load uncertainty if exists
                        if length(temp)> 8
                            utype{count,1} = temp{9};    
                            uncert(count,1) = str2num(temp{10});   % inphase
                        end
                        
                    else
                        
                        obs(count,2) = str2num(temp{8});   % quadrature
                        % Load uncertainty if exists
                        if length(temp)> 8
                            utype{count,1} = temp{9};    
                            uncert(count,2) = str2num(temp{10});
                        end
                        
                    end
                    

                    
                end
                
                stn_num(count,1:3) = [Xsnd(ss) Ysnd(ss) ss];
                
                count = count+1;

            end

        end
    end
end

fclose(fid);
%% Create output data array       
data{1} = tx;
data{2} = rx;
data{3} = freq;
data{4} = ontype;
data{5} = octype;
data{6} = utype;
data{7} = obs;
data{8} = uncert;
data{9} = stn_num;
