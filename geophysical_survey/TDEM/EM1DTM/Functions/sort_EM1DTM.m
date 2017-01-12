function [data_sort,xyz_sort] = sort_EM1DTM(data,xyz,limits,filter_r,tflag,pflag)
% [data_out] = sort_EM1DTM(work_dir,rawdata.mat,radius)
%
% INPUT:
% rawdata:  Matlab array for data
% radius:   Minimum distance between points. Used to filter out data.
%
% OUTPUT:
% data:     Cell array of data with format
% 
% Written by: D.Fournier
% Last update: 2014-03-21

%% FOR DEV ONLY
% clear all
% close all
% 
% work_dir= 'C:\Users\dominiquef\Dropbox\DIGHEM\Codes\Test';
% freqin = [56000 7200 900 5000 900];
% rawdata= 'DIGHEM_data';
% radius = 100;

%% SCRIPT STARTS HERE
% Total number of data
ndat = size(xyz,1);

X = xyz(:,1);
Y = xyz(:,2);

mask = zeros(ndat,1);

if isempty(limits)==1
    
    mask(:) = 1;
    
else
    
    xmax = limits(2);
    xmin = limits(1);

    ymax = limits(4);
    ymin = limits(3);

    mask(xyz(:,1) > xmin & xyz(:,1) < xmax &...
        xyz(:,2) > ymin & xyz(:,2) < ymax) = 1 ;

end
figure; scatter(X,Y);title('Before sorting');hold on

for ii = 1 : ndat
    
    if mask(ii)==1
        
        data_sub= [];
        for jj = 1 : size(data,2)

            data_sub{jj}   = data{jj}(ii);

        end
    
%         temp = zeros(length(rc_specs),1); temp(ii:ii+nfreq-1)=1;
%             temp = stn_num(:,1)==stn(ii,1);
        r = ( (X(ii) - X(:)).^2 +...
            (Y(ii) - Y(:)).^2 ) .^0.5;

        % Only keep the curretn points and neighbours at distance r+
        mask(r <= filter_r) = 0;
        mask(ii) = 1;
        
        %% Filter for positivity and time channels
        % Only keep data specified by (tflag) and (dtype)
        index = ones(size(data_sub{5}{1}{1}{5},1),1);

        % Check how many time channels are greater than tflag
        if ~isempty(tflag)

            index(data_sub{5}{1}{1}{5}(:,1) > tflag) = 0;

        end

        % Check how many datum are negative
        if ~isempty(pflag)

            index(data_sub{5}{1}{1}{7}(:,1) < 0) = 0;

        end
        
        % Convert to logical
        index = index == 1;
        
        % If not enough data, skip the station
        if sum(index) < 2

            mask(ii) = 0;
        
        % Otherwise remove the bad data and update the database 
        else
            
            data{5}{ii}{1}{4}(1) = sum(index);
            data{5}{ii}{1}{5} = data{5}{ii}{1}{5}(index,:);
            data{5}{ii}{1}{6} = data{5}{ii}{1}{6}(index);
            data{5}{ii}{1}{7} = data{5}{ii}{1}{7}(index,:);
            
        end
% 
%         % Select data around and stack
%         index = r < interp_r;
% 
%         % Extract data around location
%         stck = data(index,:);
% 
%         % Pre-allocate matrix
%         avgV = zeros(1,ntc);
% 
%         % Computed weighted average
%         for jj = 1 : ntc
% 
%              % Only keep positive data for each time channel
%             %          stck(:,jj) = abs(stck(:,jj));
%              select = stck(:,jj)>0;
% 
%              std_sort(count,jj) = std(stck(:,jj));
% 
%              mu = mean(stck(:,jj));
%              w = 1./abs(stck(select,jj) - mu );
%              avgV(jj) = sum(w.*stck(select,jj)) / sum(w);
% 
% 
% %              semilogx(tc(jj),mu,'ro'); hold on
% %              semilogx(tc(jj),stck(:,jj),'*','MarkerSize',2); hold on
% 
%         end
% 
%         data_sort(count,:) = avgV;
%         xyz_sort(count,:) = xyz(ii,:);
%         
%         count = count+1;
        
    end

end

data_sort=[];

for ii = 1 : size(data,2)
    data_sort{ii}   = data{ii}(mask==1);
end

xyz_sort    = xyz(mask==1,:);


figure(1)
scatter(xyz_sort(:,1),xyz_sort(:,2),'r*');title('After sorting')