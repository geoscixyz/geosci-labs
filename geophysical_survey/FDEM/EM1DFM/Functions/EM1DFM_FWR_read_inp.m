function [meshfile,obsfile,topofile,nullfile,m_con,m_sus] = EM1DFM_FWR_read_inp(inputfile)
% Function [meshfile,obsfile,topofile,nullcell,m_con,con_ref,m_sus,sus_ref,alphas,beta,cooling,target,bounds,mtype,interp_n,interp_r] = EM1DFM_read_inp([work_dir '\EM1DFM_LC.inp']); = MAG3D_read_inp('inputfile')
% Read input file for inversion

fid = fopen(inputfile,'r');
line = fgets(fid);

fprintf('Start reading input file\n')
count = 1;
while count < 6
    
    arg = regexp(line,'!','split');
    
    if count == 1
        
        meshfile = strtrim(arg{1});
        fprintf('Mesh file: \t\t\t %s\n',meshfile);
        
    elseif count == 2
        
        obsfile = strtrim(arg{1});
        fprintf('Observations: \t\t %s\n',obsfile);
    
    elseif count == 3
        
        topo = strtrim(arg{1});
        
        if ~isempty(regexp(topo,'FILE','match'))
            
            temp = regexp(topo,'\s','split');
            topofile = strtrim(temp{2});
            fprintf('Topography file: \t\t %s\n',topofile);
            nullfile = [];
        elseif ~isempty(regexp(topo,'NULL','match'))
            
            topofile =[];
            nullfile = 'nullcell.dat';
            fprintf('Nullcell file: \t\t %s\n',nullfile);
            
        else
            
            topofile = [];
            nullfile = [];
            
        end
        
    elseif count == 4
        
        m_con = strtrim(arg{1});
        
        if ~isempty(regexp(m_con,'VALUE','match'))
            
            temp = regexp(m_con,'\s','split');
            m_con = str2num(temp{2});
            fprintf('Conductivty model: \t\t %e\n',m_con);
            
        else
            
            temp = regexp(m_con,'\s','split');
            m_con = temp{2};
            fprintf('Conductivty model: \t\t %s\n',temp{2});
            
        end
        
        
            
    elseif count == 5
        
        m_sus = strtrim(arg{1});
        
        if ~isempty(regexp(m_sus,'VALUE','match'))
            
            temp = regexp(m_sus,'\s','split');
            m_sus = str2num(temp{2});
            fprintf('Susceptibility model: \t\t %e\n',m_sus);
            
        else
            
            temp = regexp(m_sus,'\s','split');
            m_sus = temp{2};
            fprintf('Susceptibility model: \t\t %s\n',temp{2});
            
        end
        
    end
        
        
    
    
    line = fgets(fid);
    count = count +1;
    
end
       
fclose(fid);