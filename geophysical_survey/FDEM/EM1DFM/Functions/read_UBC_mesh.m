function [xn,yn,zn]=read_UBC_mesh(meshfile)
% Read UBC mesh file and create nodal discretization
% Works for the condenced version (20 * 3) --> [20 20 20] 
fid=fopen(meshfile,'rt');


% Go through the mesh file and extract mesh information
for ii=1:5
    
    line=fgets(fid);
    
    % First line: number of cells in i, j, k 
    if ii==1
        
        mesh(1,:) = str2num(line);
    
        
    % Second line: origin coordinate (X,Y,Z)
    elseif ii==2
        
        mesh(2,:) = str2num(line);
    
    % Other lines for the dX, dY ,dZ
    else
        
        % Split the line
         var = regexp((line),'\s*','split');
        vec = zeros(1,mesh(1,ii-2));
        
        count = 1;
        for jj = 1 : length(var)

            if isempty(regexp(var{jj},'[*]','match'))==1 && ~isempty(var{jj})

                vec(count) = str2double(var{jj});
                count = count + 1;
            elseif ~isempty(var{jj})

                temp = regexp(var{jj},'*','split');
                dl = str2double(temp(2));
                ndx = str2double(temp(1));

                vec(count:count+ndx-1) = ones(1,ndx) * dl;
                count = count+ndx;
            end

        end
        
        
        if ii == 3
            
            dx = vec;
            
        elseif ii==4
            
            dy = vec;
            
        else
            
            dz = vec;
            
        end
        
        
    end
         
    
end

fclose(fid);

xn = [mesh(2,1) mesh(2,1) + cumsum(dx)]; 
yn = [mesh(2,2) mesh(2,2) + cumsum(dy)]; 
zn = [mesh(2,3) (mesh(2,3) - cumsum(dz))];