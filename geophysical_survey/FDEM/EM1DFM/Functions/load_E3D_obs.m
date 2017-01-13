function [trx,d] = load_E3D_obs(obsfile)
% Function [tx,data] = load_E3D_obs(work_dir,obsfile)
% Takes a UBC-E3D obsfile and create two matrices
% tx(LOOP) [ X Y Z rad phi, theta, nrx]
% data [Freq X Y Z Ex UncEx ... Hz UncHz iHz UnciHz]

fid = fopen(obsfile,'r');

% ndata = size(dwndata.data,1);
line = fgets(fid);
d = zeros(1,28);
trx = zeros(1,6);
count = 1;
count_trx = 0;
while line~=-1
    
    if isempty(regexp(line,'TRX_LOOP','match'))==0
        
        line = fgets(fid);
        while isempty(str2num(line))==1
            
            line = fgets(fid);
            
        end
        
        temp = str2num(line);
        count_trx = count_trx+1;
        trx(count_trx,1:length(temp)) = temp;
        nrx(count_trx) = 0;
        
        
    end
    
    if isempty(regexp(line,'FREQUENCY','match'))==0
        
        temp = regexp(line,'\s','split');
        freq = str2double(temp{2});
        
    end
        
    if isempty(regexp(line,'N_RECV','match'))==0
        nrecv =regexp(line,'\s\d*','match');
        nrecv = str2num(nrecv{1});
        count_recv = 0;
        while count_recv < nrecv
            line = fgets(fid);
            
            if isempty(str2num(line))==0
                d(count,1) = freq;
                temp = str2num(line);
                d(count,2:length(temp)+1) = temp;
                nrx(count_trx) = nrx(count_trx) +1;
                count = count+1;
                count_recv = count_recv+1;
            end
        end
        
    end
    
    line = fgets(fid);
    
    
end

fclose(fid);

trx = [trx nrx'];
