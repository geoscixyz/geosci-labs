function [meshfile,obsfile,topofile,nullfile,m_con,con_ref,m_sus,sus_ref,alpha_con,alpha_sus,beta,cooling,target,bounds,mtype,interp_n,interp_r,interp_s] = EM1DFM_read_inp(inputfile)
% Function [meshfile,obsfile,topofile,nullcell,m_con,con_ref,m_sus,sus_ref,alphas,beta,cooling,target,bounds,mtype,interp_n,interp_r] = EM1DFM_read_inp([work_dir '\EM1DFM_LC.inp']); = MAG3D_read_inp('inputfile')
% Read input file for inversion

fid = fopen(inputfile,'r');
line = fgets(fid);

fprintf('Start reading input file\n')
count = 1;
while count < 15
    
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
            fprintf('Start Cond model: \t\t %e\n',m_con);
            
        elseif ~isempty(regexp(m_con,'HS','match'))
            
%             temp = regexp(m_con,'\s','split');
            m_con = [];
            fprintf('Start with best-fitting Half-Space model\n');
            
        else
            
            fprintf('Start Cond model: \t\t %s\n',m_con);
            
        end
        
        
    elseif count == 5
        
        con_ref = strtrim(arg{1});
        
        if isempty(regexp(con_ref,'VALUE','match'))==0
            
            temp = regexp(con_ref,'\s','split');
            con_ref = str2num(temp{2});
            fprintf('Ref Cond model: \t\t %e\n',con_ref);
            
        else
            
            fprintf('Ref Cond model: \t\t %s\n',con_ref);
            
        end
        
    elseif count == 6
        
        m_sus = strtrim(arg{1});
        
        if ~isempty(regexp(m_sus,'VALUE','match'))
            
            temp = regexp(m_sus,'\s','split');
            m_sus = str2num(temp{2});
            fprintf('Start Susc model: \t\t %e\n',m_sus);
            
        else
            
            fprintf('Start Susc model: \t\t %s\n',m_sus);
            
        end
        
        
    elseif count == 7
        
        sus_ref = strtrim(arg{1});
        
        if isempty(regexp(sus_ref,'VALUE','match'))==0
            
            temp = regexp(sus_ref,'\s','split');
            sus_ref = str2num(temp{2});
            fprintf('Ref Susc model: \t\t %e\n',sus_ref);
            
        else
            
            fprintf('Ref Susc model: \t\t %s\n',sus_ref);
            
        end
        
    
        
    elseif count == 8
        
        target = str2num(arg{1});
        
        if isempty(target)==1
            
            target = 1.0;
            
        end
        
        fprintf('Target chi factor: \t %f\n',target);
        
    elseif count == 9
        
        alpha_con = str2num(arg{1});
        
        if isempty(alpha_con)==1 || length(alpha_con)~=4
            
            fprintf('Error in input file at line %i\n',count);
            fprintf('Requires four numerical values (e.g.--> 0.001 1 1 1\n')
            break
            
        end
        
        fprintf('Alpha conductivity values: \t\t %4.1e %4.1e %4.1e %4.1e\n',alpha_con);
        
    elseif count == 10
        
        alpha_sus = str2num(arg{1});
        
        if isempty(alpha_sus)==1 || length(alpha_sus)~=4
            
            fprintf('Error in input file at line %i\n',count);
            fprintf('Requires four numerical values (e.g.--> 0.001 1 1 1\n')
            break
            
        end
        
        fprintf('Alpha susc values: \t\t %4.1e %4.1e %4.1e %4.1e\n',alpha_sus);
        
    elseif count == 11
        
       temp = str2num(arg{1});   
       
       if isempty(temp) || length(temp)~=2
            
          fprintf('Error in input file at line %i\n',count);
            fprintf('Requires two numerical values (e.g.--> Beta_start cooling\n')
            break 
          
       else
           
          beta = temp(1);
          cooling = temp(2);
          fprintf('Starting Beta: \t\t %f\n',beta); 
          fprintf('Beta Cooling schedule: \t\t %f\n',cooling);
          
       end
    
   elseif count == 12
        
        temp = strtrim(arg{1});
        
        if isempty(regexp(temp ,'VALUE','match'))==0
            
            temp = regexp(temp,'\s','split');
            bounds(1) = str2num(temp{2});
            bounds(2) = str2num(temp{3});
            fprintf('Bounds: [ %f %f] \n',bounds(1),bounds(2));
            
        else
            
            fprintf('Default bounds: [1e-8 100]\n');
            bounds =[1e-8 100];
            
        end
        
   
    elseif count == 13
        
        mtype = str2num(arg{1});
        
        fprintf('Invert: ')
        if mtype == 1
        fprintf('Conductivity only\n');
        elseif mtype == 2
        fprintf('Susceptibility only\n');    
        elseif mtype == 3
        fprintf('Conductivity and Susceptibility (positive)\n');    
        else
        fprintf('Conductivity and Susceptibility\n');    
        end
        
    elseif count == 14

        temp = str2num(arg{1});
        
        if isempty(temp) || length(temp)~=3
            
          fprintf('Error in input file at line %i\n',count);
            fprintf('Requires three numerical values (e.g.--> N_Stations Max_dist\n')
            break 
          
        else
           
           interp_n = temp(1);
           interp_r = temp(2);
           interp_s = temp(3);
           
          fprintf('Nb interp stations: \t\t %i\n',interp_n); 
          fprintf('Max Interp Distance (m): \t\t %f\n',interp_r);
          fprintf('Smoothing Factor: \t\t %f\n',interp_s);
          
       end
    
    end
    
    line = fgets(fid);
    count = count +1;
    
end
       
fclose(fid);