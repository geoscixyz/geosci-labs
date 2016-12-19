function write_h3d_obs(outfile,data,xyz,tc,radius,uncert,pc_err)
% Function writes observed data from VTEM 1D obs

ndv = 99999;
% Create obs file
fid = fopen(outfile,'w');

ndata = size(data,1);

%% Write transmiter header
fprintf(fid,'IGNORE %i\n\n',ndv);
fprintf(fid,'N_TRX %i\n\n',ndata);

%% Write data
for ii = 1 : ndata
    
    sub_tc = find(isnan(data(ii,:))==0);
    ntc = length(sub_tc);
    
    
    fprintf(fid,'TRX_LOOP\n');
    fprintf(fid,'%12.3f %12.3f %12.3f %12.3f %12.3f %12.3f\n',xyz(ii,1),xyz(ii,2),xyz(ii,3),radius,0.0,0.0);
    fprintf(fid,'\nN_RECV %i\n', 1);
    fprintf(fid,'N_TIME %i\n',ntc);


    for jj = 1 : ntc;
               
        
        % If no data for a specific receiver/time, post no data

            
            % Assign error on magnitude of field instead of components
            % individually
%             magdB = ( Atlas.dBx(index(ii),time_out(jj))^2 +...
%                 Atlas.dBy(index(ii),time_out(jj))^2 +...
%                 Atlas.dBz(index(ii),time_out(jj))^2 )^ 0.5;
            
            err_dBz = abs(data(ii,sub_tc(jj)))*pc_err + pc_err*uncert(ii,sub_tc(jj));

            fprintf(fid,'%12.3f %12.3f %12.3f %12.8e %i %i %i %i %i %i %i %i %i %i %i %i %12.8e %12.8e %12.8e %12.8e %12.8e %12.8e ',...
                xyz(ii,1),xyz(ii,2),xyz(ii,3),tc(sub_tc(jj)),...
                ndv,ndv,ndv,ndv,ndv,ndv,ndv,ndv,ndv,ndv,ndv,ndv,...
                ndv,ndv,ndv,ndv,data(ii,sub_tc(jj)),err_dBz);

            % Write dB/dt for input components in coil
%             for kk = 1:3
% 
%                 if kk==1 && sum(coil==kk)~=0
% 
%                     fprintf(fid,'%12.8e %12.8e ',...
%                     Atlas.dBx(index(ii),t_out(jj)),err_dBx);
% 
%                 elseif kk==2 && sum(coil==kk)~=0
% 
%                     fprintf(fid,'%12.8e %12.8e ',...
%                     Atlas.dBy(index(ii),t_out(jj)),err_dBy);
% 
%                 elseif kk==3 && sum(coil==kk)~=0
% 
%                     fprintf(fid,'%12.8e %12.8e ',...
%                     Atlas.dBz(index(ii),t_out(jj)),err_dBz);
% 
%                 else
% 
%                     fprintf(fid,'%i %i ', ndv,ndv);
% 
%                 end
% 
%             end
        
            fprintf(fid,'\n');

    
    end
    
        fprintf(fid,'\n');
end

fclose(fid);