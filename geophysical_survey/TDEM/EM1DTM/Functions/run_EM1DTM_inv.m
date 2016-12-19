function [m_con1D, m_misfit, phid, phim, beta,pred,bHSpace] = run_EM1DTM_inv(work_dir,meshfile,data,m_start,m_ref,alpha,w,Q,phid,beta,cooling,target,iter,HSflag)
%
% 
% ORIGINAL CODE FROM: SEOGI KANG
% ADAPTED BY: Dom Fournier (2014-03-21)
% 
home_dir = pwd; % get the absolute path of this file
% oldFolder = cd(thisfile_path); % get into modeling directory

% internal parameters
huberekblom = [1000  2 0.0001  2 0.0001]; % Huber and Ekblom parameters
tol = []; % convergence test
hankle = []; % Hankle transform parameter
fourier = []; % Fourier transform parameter
output = 2; % amount of output

pred = [];
%% Load 3D mesh
[xn,yn,zn] = read_UBC_mesh([work_dir '\' meshfile]);
nx = length(xn)-1;
ny = length(yn)-1;
nz = length(zn)-1;

% Vertical discretization
dz = zn(1:end-1) - zn(2:end);

%% Re-shape model
% Reshape conductivity and susceptibility models
m_start = reshape(m_start,nz,nx,ny);
m_ref = reshape(m_ref,nz,nx,ny);

w = reshape(w,nz,nx,ny);



%% Pre-allocate for inversion output
nstn = size(data{1},1);

% Create final 1D model result matrix. At most nz cells from 3D mesh
% Create final 1D model result matrix. At most nz cells from 3D mesh
m_con1D = ones(nz,nx,ny)*1e-8;
m_misfit = ones(nz,nx,ny)*1e-8;
bHSpace     = ones(nz,nx,ny)*1e-8;

% Pre-allocate results
itern   = zeros(nstn,1);
phim    = zeros(nstn,1);
phi     = zeros(nstn,1);


%% Run all the stations in a loop
% pooljob = parpool(3);
inv_dir = [work_dir '\Workspace'];
mkdir(inv_dir);
% system(['copy ' work_dir '\em1dtm.wf ' inv_dir]);
    
for ii = 1 : nstn
    
    % Change work_dir to workspace
    
    
    data_sub= [];
    for jj = 1 : size(data,2)
        
        data_sub{jj}   = data{jj}(ii);
        
    end
    
       
    % Write observation file
    write_EM1DTM_obs([inv_dir '\em1dtm.obs'],data_sub,[])
    
    %% write input file
    fid = fopen([inv_dir,'\','em1dtm.in'],'wt');
    fprintf(fid,'em1dtm\n');
    fprintf(fid,'em1dtm.obs\n');
    
%     if iter==1
%         fprintf(fid,'inimodel.con\n');
%         fprintf(fid,'refmodel.con\n');        
%     else
        fprintf(fid,'inimodel.con\n');
        fprintf(fid,'refmodel.con\n');
%     end
    fprintf(fid,'NONE\n');
    fprintf(fid,'NONE\n');
    fprintf(fid,'%f  %f  %f  %f  %f\n',huberekblom);
    fprintf(fid,'%f  %f\n',alpha);
    
    fprintf(fid,'1  ! inv type\n');
    fprintf(fid,'%9.5e  ! inv parameters\n',beta(ii));
    fprintf(fid,'1\n');

    if isempty(tol)
        fprintf(fid,'DEFAULT\n');
    else
        fprintf(fid,'%f\n',tol);
    end
    if isempty(hankle)
        fprintf(fid,'DEFAULT\n');
    else
        fprintf(fid,'%d\n',hankle);
    end
    if isempty(fourier)
        fprintf(fid,'DEFAULT\n');
    else
        fprintf(fid,'%d\n',fourier);
    end
    fprintf(fid,'%d\n',output);
    fclose(fid);

    %%
    % Create layer conductivity, susc and weights
    dz_layer = dz(Q(ii,3):end)';
    cond = m_start(Q(ii,3):end,Q(ii,1),Q(ii,2));
    cond_ref = m_ref(Q(ii,3):end,Q(ii,1),Q(ii,2));
    wght = w(Q(ii,3):end,Q(ii,1),Q(ii,2));
    
    ncells = length(dz_layer);
    
    % Check if phid has reached misfit, if no then continue
    if phid(ii) <= target;
        
        beta(ii) = beta(ii);
        fprintf('##\nStation %i has reached the target misfit\n##\n',ii)
        fprintf('##\nKeep current beta and re-invert\n##\n')
    
    else
        
        beta(ii) = beta(ii)*cooling; 
          
    end
    
    % Write reference and starting model
    fid1 = fopen([inv_dir '\inimodel.con'],'wt');
    fprintf(fid1,'%i\n',ncells+1);
    
    fid2 = fopen([inv_dir '\refmodel.con'],'wt');
    fprintf(fid2,'%i\n',ncells+1);
    
    
    for jj = 1 : ncells
        

        fprintf(fid2,'%12.4f%12.4e\t\n',dz_layer(jj),cond_ref(jj));
        
        if iter==1 && HSflag == 0
            
            fprintf(fid1,'%12.4f\n',dz_layer(jj));
           
        else
        
            fprintf(fid1,'%12.4f\t%12.8e\n',dz_layer(jj),cond(jj));
        
        end
        
        
    end
    
    if iter==1 && HSflag == 0
        fprintf(fid1,'%12.4f\n',0.0);
    else
        fprintf(fid1,'%12.4f\t%12.8e\n',0.0,cond(jj));
    end
    fclose(fid1);
    
    fprintf(fid2,'%12.4f\t%12.8e\n',0.0,cond_ref(jj));
    fclose(fid2);

    %% run code
    cd(inv_dir);
    fprintf('Sounding %i / %i\n',ii,nstn);
    [status,cmdout] = system('em1dtm');

    cd(home_dir);
    
    
    fid = fopen([inv_dir,'\em1dtm.con'],'r');
    tline = fgetl(fid);
    nlayer = str2num(tline);
    
    model = zeros(nlayer,1);

    for p = 1:nlayer
        tline = fgetl(fid);
        temp = sscanf(tline,'%f');
        model(p) = temp(2);
    end
    fclose(fid);

    % Project back to 3D mesh
%     m_con1D{ii} = model(1:end-1);
    m_con1D(Q(ii,3):end,Q(ii,1),Q(ii,2)) = model(1:end-1);
    
    % Copy top cell all the way up the mesh to avoid 
    % interpolating air cells later
    m_con1D(1:Q(ii,3),Q(ii,1),Q(ii,2)) = model(1);
    
    % read pred
    pred_out = read_EM1DTM_pre([inv_dir '\em1dtm.prd']);

    for jj = 1 : size(data,2)
        
        pred{jj}{ii,1}   = pred_out{jj}{:};
        
    end
    % Add uncertainties and location from original obs
    pred{5}{ii}{1}{6} = data{5}{ii}{1}{6};
    pred{5}{ii}{1}{2} = data{5}{ii}{1}{2};
    pred{5}{ii}{1}{7}(:,2) = data{5}{ii}{1}{7}(:,2);
    
    % read em1dtm.out
    fid = fopen([inv_dir,'\em1dtm.out'],'r');
    line = fgets(fid);
    while line~=-1
        
        iteration = regexp(line,'Iteration','match');
        
        if isempty(iteration) == 0
            
            invout      = regexp(line,'(?<=\=)[a-zA-Z_0-9+.- ][^,]*','match');
            phid(ii)    = str2num(invout{1});
            betafromfile(ii)    = str2num(invout{2});
            phim(ii)    = str2num(invout{3});            
            % Print to screen result
            fprintf('         phid = %g, phim = %g, beta = %g\n',phid(ii),phim(ii),betafromfile(ii));

        end
        
        temp = regexp(line,'Best-fitting','match');
        if isempty(temp) == 0
            
            invout      = regexp(line,'\d*\.?\d*','match');
            bHSpace(:,Q(ii,1),Q(ii,2)) = str2double(invout{1})*10^(-str2double(invout{2}));
            
        end
        
        line = fgets(fid);
        
    end
    fclose(fid);

    m_misfit(:,Q(ii,1),Q(ii,2)) =  phid(ii);
%     cd(oldFolder); % get beck to previous directory
%     system(['rmdir /S /Q ' inv_dir]);
end
% delete(pooljob);
m_con1D = reshape(m_con1D,nz*nx*ny,1);

m_misfit = reshape(m_misfit,nz*nx*ny,1);
bHSpace  = reshape(bHSpace,nz*nx*ny,1);
end