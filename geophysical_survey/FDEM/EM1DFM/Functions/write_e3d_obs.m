function write_e3d_obs(filename,tx,data)
% Function write_e3d_obs(filename,freq,tx,rx)
% Write obsservation file for E3Dinv
% TO FINISH DESCRIPTION...

fid = fopen(filename,'w');

fprintf(fid,'! Exported from write_e3d_obs: D.Fournier\n');
fprintf(fid,'IGNORE NaN\n\n');
fprintf(fid,'N_TRX %i\n', size(tx,1) );

for jj = 1 : size(tx,1)
    
    
    fprintf(fid,'TRX_LOOP\n');
    fprintf(fid,'%12.8e %12.8e %12.8e %12.8e %12.8e %12.8e\n',tx(jj,1:end-1));
        
    
    fprintf(fid,'FREQUENCY %12.8e \n',data(jj,1));   
        
%         fprintf(fid,'FREQUENCY %12.8e\n',data(nrcv(jj),1));
    fprintf(fid,'N_RECV %i\n',1);

    fprintf(fid,'%12.8e %12.8e %12.8e ',data(jj,2:4));

    for ll = 1 : 20

        fprintf(fid,'NaN ');

    end

    if isnan(data(jj,25))

        fprintf(fid,'NaN NaN ');

    else

        fprintf(fid,'%12.8e %12.8e ',data(jj,25), data(jj,26));

    end

    if isnan(data(jj,27))

        fprintf(fid,'NaN NaN ');

    else

        fprintf(fid,'%12.8e %12.8e ',data(jj,27), data(jj,28));

    end

    fprintf(fid,'\n');
            
        
    
end

fclose all
