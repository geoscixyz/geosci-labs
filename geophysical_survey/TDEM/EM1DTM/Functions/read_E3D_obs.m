function [trx,d] = read_E3D_obs(obsfile)
% Function [tx,data] = load_E3D_obs(work_dir,obsfile)
% Takes a UBC-E3D time-domain obsfile and create two matrices
% tx(LOOP) [ X Y Z rad phi, theta, nrx]
% data [X Y Z Time Ex UncEx ... Hz UncHz iHz UnciHz]

fid = fopen(obsfile,'r');

% ndata = size(dwndata.data,1);
line = fgets(fid);

while isempty(regexp(line,'N_TRX','match'))
    
    line = fgets(fid);
    
end

temp = regexp(line,'\s','split');

ntrx = str2num(temp{2});

d = [];
trx = [];

count = 1;
count_tx = 0;
while line~=-1
    
    if isempty(regexp(line,'TRX_LOOP','match'))==0
        
        line = fgets(fid);
        while isempty(str2num(line))==1
            
            line = fgets(fid);
            
        end
        
        temp = str2num(line);
        count_tx = count_tx+1;
        trx{count_tx,1} = temp;      
        
    end
    
        
    if isempty(regexp(line,'N_RECV','match'))==0
        nrecv =regexp(line,'\s\d*','match');
        nrecv = str2num(nrecv{1});
        
        count_rx = 0;
        
        while count_rx < nrecv
            
            count_rx = count_rx+1;
            
            line = fgets(fid);
            temp = regexp(strtrim(line),'\s\d*','match');
            nt = str2num(temp{1});
            
            
            
            
            for ii = 1 : nt
                
                line = fgets(fid);
                temp = str2num(line);

                d{count_tx}{count_rx}(ii,1:22) = temp;

                
            end
            
            
            
        end
        
    end
    
    line = fgets(fid);
    
    
end

fclose(fid);

