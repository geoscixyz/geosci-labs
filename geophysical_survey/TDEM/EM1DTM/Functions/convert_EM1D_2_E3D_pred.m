function convert_EM1D_2_E3D_pred(obsfile,tc,outfile)
% function convert_dbs_2_mat(data,axis)
% Convert a E3D-TD file format to a 2D matrix. Only reformat the data for a
% single component at the time
%
% INPUT:
% obsfile   : EM1DTM obsfile file
% tc        : Time channels expected by E3D file
% outfile   : Name of the outputed predicted E3D file
%
% OUTPUT:
% outfile

% Load EM1D obs file
data = read_EM1DTM_obs(obsfile);

fid = fopen(outfile,'w');

% Loop through all the stations and write file
for ii = 1 : size(data{5},1)
    
    for jj = 1 : size(data{5}{ii},1)
        
       tin = data{5}{ii}{jj}{5}(:,1);
       
       % Write to file
       for kk = 1 : length(tc)
           
           index = tc(kk) == tin;
           if sum(index) == 1
               
               fprintf(fid,'%e\t%e\t%e\t%e\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\t%e\n',...
                   data{1}{ii}(1),data{1}{ii}(2),...
                   data{1}{ii}(3), tc(kk), data{5}{ii}{jj}{7}(index,1)); 
               
           else
               
               fprintf(fid,'%e\t%e\t%e\t%e\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\n',...
                   data{1}{ii}(1),data{1}{ii}(2),...
                   data{1}{ii}(3), tc(kk));
               
           end
            
       end
       
       fprintf(fid,'\n');
       
    end
    
end

fclose all;
    
    
