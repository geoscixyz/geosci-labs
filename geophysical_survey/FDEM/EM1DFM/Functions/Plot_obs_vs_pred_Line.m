% Function: plot_obs_vs_pred_Line
% Plot observed vs predicted data from EM1D inversion along line
% Line number is infered from the x,y location of each sounding, assuming
% that the data is ordered as it was surveyed.
% 
% Script assumes that each frequency has the same number of soundings. Will
% need more work to figure out a way to drop this assumption.
% 
% INPUTS:
% work_dir: directory for the files
% obsfile: Observed data matrix
% prefile: Predicted data matrix
% linefile(optional) : list of line names used for labeling
%
% Last update: August 23, 2015
% D Fournier
% fourndo@gmail.com

clear all
close all

addpath '.\Functions';

%%
work_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Desktop\3858_EM1DFM\Data';

obsfile = 'Inv_Data_XYZ_ds15m.dat';

predfile = 'Inv_PRED_iter3.pre';

linefile = 'line_all.txt';

dsep = '\';

%% Load data

data = load([work_dir '\' obsfile]);
pred = load([work_dir '\' predfile]);
freq = unique(data(:,4));
nfreq = length(freq);

%% Find survey lines
% Grab one of the block frequency

indx = data(:,4)==freq(1);
x = data(indx,1);
y = data(indx,2);
lineID = xy_2_lineID(x,y);

[LL,LI,LA] = unique(lineID);
nlines = length(LL);

% If input a linefile, create lookup table for name of lines
if ~isempty(linefile)
    
    fid = fopen([work_dir dsep linefile],'r');
    line = fgets(fid);
    count = 0;
    
    while line~=-1
        
        count = count + 1;       
        linename{count} = line ;
        line = fgets(fid);
        
    end
    
end
 
   
% Check if input linefile has the same length as the line detection
if size(linename,2)~= nlines
    
    fprintf('Script detected %i lines, but input line file has %i. Please verify\n',nlines,size(linename,2));
    
else
    
    % Plot the survey lines for reference.
    figure;
    for ii = 1 : nlines
        
        scatter( x(lineID==ii) , y(lineID==ii) ,1 ); hold on
        text( x(LI(ii)) , y(LI(ii)) , linename{ii} , 'HorizontalAlignment', 'center' )

    end
    title('Line ID detected')
    colormap(jet);
    axis equal
    
end
%% Plot each lines, four per figures
count = 4;
for jj = 1 : nlines


set(figure, 'Position', [25 50 1800 900])


% Grab all the stations and frequency for current line
indx = lineID == jj;
indx = kron(indx,ones(nfreq,1))==1;


d = data(indx,:);
p = pred(indx,:);


for ii = 1 : nfreq
    
    % Plot Real on left side
    subplot(3,2,(ii-1)*2+1)
%     errorbar( d( d(:,4) == freq(ii) , 5) ,...
%         d( d(:,4) == freq(ii) , 7), 'k-'); hold on
    
    plot( d( d(:,4) == freq(ii) , 5),'k-', 'LineWidth',2); hold on

    plot( p( p(:,4) == freq(ii) , 5) ,'r--', 'LineWidth',2);
    axis tight

    % Plot real on Imaginary side
    subplot(3,2,(ii-1)*2+2)

%     errorbar( d( d(:,4) == freq(ii) , 6) ,...
%         d( d(:,4) == freq(ii) , 8), 'k-'); hold on
    
    plot( d( d(:,4) == freq(ii) , 6),'k-', 'LineWidth',2); hold on

    plot( p( p(:,4) == freq(ii) , 6), 'r--', 'LineWidth',2);
    axis tight
    
end
% d = data(indx,1);
subplot(3,2,1); title('\bf Real'); ylabel(['\bf ' num2str(freq(1)) ' Hz']);
subplot(3,2,2); title('\bf Imaginary')
subplot(3,2,3);ylabel(['\bf ' num2str(freq(2)) ' Hz']);
subplot(3,2,5);ylabel(['\bf ' num2str(freq(3)) ' Hz']); xlabel('Station')
subplot(3,2,6);xlabel('Station')

axes('Position',[0.45 .85 .1 .1]);
set(gca,'Visible','off');
text(0.3,0.9,['Line:' linename{jj}],'FontSize',12);

end



