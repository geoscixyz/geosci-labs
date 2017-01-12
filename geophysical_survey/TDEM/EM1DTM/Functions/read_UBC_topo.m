function [topo]=read_UBC_topo(topofile)
% Function [topo]=read_topo(work_dir,topofile)
% Open topofile in UBC format
% Returns a 3-by-n array
% X,Y,Z



% Load topography - UBC style
fprintf('Loading Topofile ... please standby\n')

if isempty(topofile)==1

    topo = zeros(1,3);
else
    
    % Load header
    fid = fopen(topofile,'r');
    line = fgets(fid);

    nnodes = str2double(line);
    topo = zeros(nnodes,3);
    for ii = 1 : nnodes


        topo(ii,:) = str2num(fgets(fid));

    end
    
    fclose(fid);
    fprintf('Completed!\n')

end

