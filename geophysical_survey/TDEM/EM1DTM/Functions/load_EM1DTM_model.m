function [m] = load_EM1DFM_model(work_dir,modelfile)
% function plot_EM1DFM_model
% Load inverted model from UBC-EM1DFM and ouput 1D mesh. 
%
% INPUTS:
% work_dir      : directory to get the files
% modelfile     : Resistivity model as putputed by em1dinv
%
% IN DEVELOPMENT
%
% Written by: D.Fournier
% Last update: February 25th, 2014


fprintf('Start loading inverted 1D models!\n\n')
fid=fopen([work_dir '\' modelfile],'rt');    

% First load parameters in header
line=fgets(fid); %gets next line
ndz = str2num(line); % number of layers


% Create and array of model points in cell center
% XYZ = zeros( nsnd*ndz , 3 );
m   = zeros( ndz , 1 );

for ii = 1 : ndz
   
   line=fgets(fid); %Skip head
   temp = str2num(line);
   
   m(ii)       =  temp(2);
      

end

fclose(fid);

fprintf('Model cells loaded succesfully!\n\n')

