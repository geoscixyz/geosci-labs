function [pred] = run_EM1DTM_fwd(work_dir,meshfile,data,m_con,Q)

% ORIGINAL CODE FROM: SEOGI KANG
% ADAPTED BY: Dom Fournier (2014-03-21)
% 
root_dir = pwd; % get the absolute path of this file
% oldFolder = cd(thisfile_path); % get into modeling directory


%% Load 3D mesh
[xn,yn,zn]=read_UBC_mesh([work_dir '\' meshfile]);

nx = length(xn) - 1 ;
ny = length(yn) - 1 ;
nz = length(zn) - 1 ;

dz = zn(1:end-1) - zn(2:end);
%% Pre-allocate for inversion output
nsnds = size(data,1);

% Pre-allocate results
pred = zeros(size(data,1),size(data,2)); % same size as data and sd



%% Run all the stations in a loop
for ii = 1 : nsnds
    
    data_sub= [];
    for jj = 1 : size(data,2)
        
        data_sub{jj}   = data{jj}(ii);
        
    end
    
       
    % Write observation file
    write_EM1DTM_obs(work_dir,'em1dtm.obs',data_sub,[])
    
    
    %% write inp file
    fid = fopen([work_dir,'\','em1dtmfwd.in'],'wt');
    fprintf(fid,'em1dtm.obs\n');
    fprintf(fid,'inimodel.con\n');
    fprintf(fid,'DEFAULT\n');
    fprintf(fid,'DEFAULT\n');
    fprintf(fid,'NO\n');
    fclose(fid);


    %%
    % Create layer conductivity, susc and weights
    m_con = reshape(m_con,nz,nx,ny);
    
    dz_layer = dz(Q(ii,3):end)';
    condin = m_con(Q(ii,3):end,Q(ii,1),Q(ii,2));
    
    ncells = length(dz_layer);
    
    fid1 = fopen([work_dir '\inimodel.con'],'wt');
    fprintf(fid1,'%i\n',ncells+1);
    
    
    
    for jj = 1 : ncells
        
        % If first iteration then right out best fitting half-space
       
        fprintf(fid1,'%12.4f\t%12.8e\n',dz_layer(jj),condin(jj));
        
        
    end
    
    fprintf(fid1,'%12.4f\t%12.8e\n',0.0,condin(jj));
    fclose(fid1);
    

    %% run code
    cd(work_dir);
    system('em1dtmfwd');
    cd(root_dir);
    
    % read pred
    fid = fopen([work_dir,'\em1dtmfwd.out'],'r');
    tline = fgetl(fid);
    tline = fgetl(fid);
    tline = fgetl(fid);
    
    for p = 1:nrx
        rxline = fgetl(fid);
        temp = regexp(rxline,'\s*','split');
        npred = str2num(temp{end-1});
        
        for q = 1:npred
            tline = fgetl(fid);
            temp = sscanf(tline,'%f');
            pred(ii,q) = temp(3);
        end
        
        % Move time channels to proper slot
        pred(ii,nnztc==1) = pred(ii,1:npred);
        pred(ii,nnztc==0) = nan;
%         if strcmp(rxline(end),'4') % nT for B field
%             pred(ii,1:npred) = pred(ii,1:npred); % go back to SI
%         end
    end
    fclose(fid);

%     cd(oldFolder); % get beck to previous directory
end
