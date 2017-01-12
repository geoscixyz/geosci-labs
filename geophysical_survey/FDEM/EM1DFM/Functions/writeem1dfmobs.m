function writeem1dfmobs(work_dir,filename,data ,varargin)
% WRITEEM1DFMOBS(rootname,data,varargin)
% Creates observation file in UBC format for the 1D inversion of frequency
% domain airborn data. 
% 
% INPUTS:
%
% tr_specs(5 x ndatum) transmitter parameters [dx dy -z mom_t axis]
% rc_specs(5 x ndatum) receiver parameters [x y -z mom_r axis stn_num]
% freq    (1 x ndatum) vector of frequency for each datum
% ontype  (1 x ndatum) Type of nomr
% octype  (1 x ndatum) 'b' for both, 'i' for in-phase or 'q' for quadrature
% utype   (1 x ndatum) 'v' for absolute uncertainty or 'p' for percentage
% obs     (2 x ndatum)  or (1 x ndatum) depending on octype
% uncert  (2 x ndatum)  or (1 x ndatum) depending on octype
% 
% 
% Original script: Lindsey Heagy
% Last modified by: D Fournier (2014-02-27)
%
% DO TO LIST:
% Modify script to look for common receiver/frequency pairs

% fdir        = './';
% 
% for pair = 1:2:numel(varargin)
%     optname = varargin{pair};
%     if ~ischar(optname)
%         error('varargin:type','Invalid input, must be a string');
%     end
%     
%     switch upper(optname)
%         case 'DIR'
%             fdir = varargin{pair+1};
%     end
%     disp('Currently ignoring varargin, just using default');
% 
%     % PARSE VARARGIN
% end

% tr_specs= data{1};
% rc_specs= data{2};
% freq    = data{3} ;
% ontype  = data{4} ;
% octype  = data{5} ;
% utype   = data{6} ;
% obs     = data{7} ;
% uncert  = data{8} ;
% snds    = data{9} ;


% Extract data location
% nsnds = length(unique(data{9}(:,1)));
% nfreq = length(data{9}(:,1)) / nstn;


%% Write obs
[l,index] = unique(data{9}(:,3));

nsnds = length(index);

% Cycle through the data and create obsfile, layer file and run inversion
    fid = fopen([work_dir '\' filename],'wt');
    fprintf(fid,'%i\n',nsnds);
    
for ii = 1 : nsnds
    
    logic = data{9}(:,3) == data{9}(index(ii),3);
       
    freq    = data{3}(logic);
    tx      = data{1}(logic,:);
    rx      = data{2}(logic,:);
    ontype  = data{4}(logic);
    octype  = data{5}(logic);
    utype   = data{6}(logic);
    obs     = data{7}(logic,:);
    uncert  = data{8}(logic,:);
    X       = data{9}(logic,1);
    Y       = data{9}(logic,2);
    nfreq   = sum(logic);
    

    fprintf(fid, '%f  %f  %i\n', X(1) , Y(1), nfreq); 
           
    
    for ff = 1 : nfreq
        
        if tx(ff,5) == 1
            
            ot = 'x';
            
        elseif tx(ff,5) == 2
            
            ot = 'y';
            
        elseif tx(ff,5) == 3
            
            ot = 'z'; 
            
        end
        
        if rx(ff,5) == 1
            
            or = 'x';
            
        elseif rx(ff,5) == 2
            
            or = 'y';
            
        elseif rx(ff,5) == 3
            
            or = 'z'; 
            
        end        
        
        fprintf(fid, '%i  %i\n', freq(ff), 1);
        fprintf(fid, '%f  %f  %s  %i\n', tx(ff,4), tx(ff,3), ot, 1);

        if strcmp(octype{ff},'b')==1
            fprintf(fid, '%f  %f  %f  %f  %s  %i  %s  %12.4e  %12.4e  %s  %12.4e  %12.4e\n',...
                    rx(ff,4), tx(ff,1), tx(ff,2),...
                    rx(ff,3) ,or, ontype(ff), octype{ff},...
                    obs(ff,1),obs(ff,2),utype{ff},uncert(ff,1),...
                    uncert(ff,2));

        else

            fprintf(fid, '%f  %f  %f  %f  %s  %i  %s  %12.4e  %s  %12.4e  \n',...
                    rx(ff,4), tx(ff,1), tx(ff,2),...
                    rx(ff,3) ,or, ontype(ff), octype{ff},...
                    obs(ff,1),utype{ff},uncert(ff,1));

        end
           
        
        
    end
    
    
end
fclose(fid);