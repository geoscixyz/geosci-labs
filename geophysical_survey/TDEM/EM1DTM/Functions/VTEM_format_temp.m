% reformat VTEM data

clear all
close all

addpath     'C:\Users\dominiquef.MIRAGEOSCIENCE\Dropbox\Master\FUNC_LIB';
work_dir    = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\Projects\UBC_Tli_Kwi_Cho\Modelling\Inversion\EM_Inversion\VTEM';

load([work_dir '\VTEMdbdtProject']);

nstn = max(dataMat(:,5));

data = zeros(nstn,27);
xyz = zeros(nstn,4);

for ii = 1 : nstn
    
    temp = dataMat(dataMat(:,5)==ii,:);
    data(ii,:) = temp(:,7)';
    
    tc = temp(:,6)';
    
    xyz(ii,1) = temp(1,2);
    xyz(ii,2) = temp(1,3);
    xyz(ii,3) = temp(1,9);
    
    xyz(ii,4)  = temp(1,10)-40;
    
end

save([work_dir '\VTEM_data_DF'],'data');
save([work_dir '\VTEM_xyz_DF'],'xyz');
save([work_dir '\VTEM_tc_DF'],'tc');


