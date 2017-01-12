% Function: plot_obs_vs_pred_2D
% Plot 2D interpolated observed vs predicted data from EM1D inversion
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

%% INPUT FILES
% work_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\ownCloud\Research\Maysam\DIGHEM';
work_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\UBC_GIF\em_examples\geophysical_survey\FDEM';
% Data file: format->  [x, y, z, freq, H_I, H_Q, Uncert_I, Uncert_Q]
obsfile = 'Inv_Data_XYZ.dat';

% Predicted file: format-> [x, y, z, freq, H_I, H_Q, Uncert_I, Uncert_Q]
predfile = 'Inv_PRED_iter4.pre';

%% Load data
set(figure(1), 'Position', [25 50 1800 900])
set(figure(2), 'Position', [25 50 1800 900])
data = load([work_dir '\' obsfile]);
pred = load([work_dir '\' predfile]);

freq = unique(data(:,4));
nfreq = length(freq);
for ii = 1 : nfreq
    
    % Grab data for single frequency
    indx = data(:,4) == freq(ii);
    sub_data = data(indx,:);
    
    indx = pred(:,4) == freq(ii);
    sub_pred = pred(indx,:);

    
    if ii == 1
        %% Set coordinates for plot
        xmin = min(sub_data(:,1));
        xmax = max(sub_data(:,1));

        ymin = min(sub_data(:,2));
        ymax = max(sub_data(:,2));

        dx = 10;
        dy = 10;

        x = xmin:dx:xmax;
        y = ymin:dy:ymax;
        [Y,X] = ndgrid(y,x);

        Y = flipud(Y);
    end
    
%     F_I = TriScatteredInterp(sub_pred(:,2),sub_pred(:,1),sub_pred(:,6),'natural');
%     F_R = TriScatteredInterp(sub_pred(:,2),sub_pred(:,1),sub_pred(:,5),'natural');
    
    % Grid data and plot
    pred_I = griddata(sub_pred(:,2),sub_pred(:,1),sub_pred(:,6),Y,X);
    pred_R = griddata(sub_pred(:,2),sub_pred(:,1),sub_pred(:,5),Y,X);
    
%     F_I = TriScatteredInterp(sub_data(:,2),sub_data(:,1),sub_data(:,6),'natural');
%     F_R = TriScatteredInterp(sub_data(:,2),sub_data(:,1),sub_data(:,5),'natural');
    
    data_I = griddata(sub_data(:,2),sub_data(:,1),sub_data(:,6),Y,X);
    data_R = griddata(sub_data(:,2),sub_data(:,1),sub_data(:,5),Y,X);
    
%     U_I = TriScatteredInterp(sub_data(:,2),sub_data(:,1),sub_data(:,8),'natural');
%     U_R = TriScatteredInterp(sub_data(:,2),sub_data(:,1),sub_data(:,7),'natural');
    
    uncert_I = griddata(sub_data(:,2),sub_data(:,1),sub_data(:,8),Y,X);
    uncert_R = griddata(sub_data(:,2),sub_data(:,1),sub_data(:,7),Y,X);
    
    
    figure(1)
    subplot(4,nfreq,(ii-1)+1)
    h = imagesc(x,y,data_R);
    set(h,'alphadata',~isnan(data_R));
    title(['\bf' num2str(freq(ii)) 'Real'])
    caxis([min(sub_data(:,5)) max(sub_data(:,5))])
    colorbar
    colormap(jet)
    axis equal tight
    grid on
    
    subplot(4,nfreq,nfreq+1+(ii-1))
    h=imagesc(x,y,pred_R);
    set(h,'alphadata',~isnan(data_R));
    title('\bfPredicted Real')
    caxis([min(sub_data(:,5)) max(sub_data(:,5))])
    colorbar
    axis equal tight
    grid on
    
%     figure(2)
%     subplot(4,nfreq,(ii-1)*2+1)
%     h = imagesc(x,y,data_R);
%     set(h,'alphadata',~isnan(data_R));
%     title(['\bf' num2str(freq(ii)) 'Real'])
%     caxis([min(sub_data(:,5)) max(sub_data(:,5))])
%     colorbar
%     colormap(jet)
%     axis equal tight
%     grid on
%     
%     subplot(2,nfreq,(ii-1)*2+nfreq+1)
%     h=imagesc(x,y,( data_R-pred_R ) ./ uncert_R );
%     set(h,'alphadata',~isnan(data_R));
%     title('\bfNormalized Residual')
%     caxis([-4 4])
%     colorbar
%     axis equal tight
%     grid on

%     figure(1)
    subplot(4,nfreq,2*nfreq+1+(ii-1))
    h=imagesc(x,y,data_I);
    set(h,'alphadata',~isnan(data_I));
    title(['\bf' num2str(freq(ii)) ' Imag'])
    caxis([min(sub_data(:,6)) max(sub_data(:,6))])
    colorbar
    axis equal tight
    grid on
    
    subplot(4,nfreq,3*nfreq+1+(ii-1))
    h=imagesc(x,y,pred_I);
    set(h,'alphadata',~isnan(data_I));
    title('\bfPredicted Imag')
    caxis([min(sub_data(:,6)) max(sub_data(:,6))])
    colorbar
    axis equal tight
    grid on
    
%     figure(2)
%     subplot(2,nfreq,(ii-1)*2+2)
%     h=imagesc(x,y,data_I);
%     set(h,'alphadata',~isnan(data_I));
%     title(['\bf' num2str(freq(ii)) ' Imag'])
%     caxis([min(sub_data(:,6)) max(sub_data(:,6))])
%     colorbar
%     axis equal tight
%     grid on
%     
%     subplot(2,nfreq,(ii-1)*2+nfreq+2)
%     h=imagesc(x,y, ( data_I-pred_I ) ./ uncert_I);
%     set(h,'alphadata',~isnan(data_I));
%     caxis([-4 4])
%     title('\bfNormalized Residual')
%     colorbar
%     axis equal tight
%     grid on

%     figure(1)
%     subplot(2,nfreq,ii)
%     h=imagesc(x,y,data_R);
%     set(h,'alphadata',~isnan(data_R));
%     title(['\bf' num2str(freq(ii)) ' Real'])
%     caxis([min(sub_data(:,5)) max(sub_data(:,5))])
%     colorbar
%     colormap(jet)
%     axis equal tight
%     grid on
%     
%     subplot(2,nfreq,ii+nfreq)
%     h=imagesc(x,y,data_I);
%     set(h,'alphadata',~isnan(data_I));
%     title(['\bf' num2str(freq(ii)) ' Imag'])
%     caxis([min(sub_data(:,6)) max(sub_data(:,6))])
%     colorbar
%     axis equal tight
%     grid on
end

