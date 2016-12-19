import time
import numpy as np
import scipy as sp
import scipy.constants as constant
import matplotlib.pyplot as plt
import re
from matplotlib import animation
from JSAnimation import HTMLWriter

from SimPEG import Mesh, Utils, mkvc


"""
    FDEM Sphere
    ===========

    Setup for a forward and inverse for a sphere in a halfspace.
    Calculations are done with e3d_tiled code
    Created by @fourndo

"""
# Conductivity model [halfspace,sphere]
sig = [1e-2,1e-0]
air = 1e-8
mfile = 'Sphere_Tensor.con'

vmin = np.log10(sig[0])
vmax = np.log10(sig[1])
# Location and radisu of the sphere
loc = [0,0,-40.]
radi = 30.

# Survey parameters
rx_height = 20.
rx_offset = 8.
times = np.r_[np.linspace(1e-5, 1e-4, 10),
              np.linspace(2e-4, 1e-3, 9),
              np.linspace(2e-3, 1e-2, 9)]



wave = np.r_[np.c_[np.linspace(-5e-5, 0.0, 6), np.ones(6)],
             np.c_[times, np.zeros(len(times))]]

# Define time channels for the data
d_tc = times[1:-2:2] + 5e-6
ntimes = len(d_tc)

#Shift the waveform in time to avoid sampling on time gates
#wave[:,0] 

# Number of stations along x-y
nstn = 9
xlim = 150

# First we need to create a mesh and a model.
# This is our mesh
dx = 2.5
nC = 60
npad = 15

# Floor uncertainties for e3D inversion
floor = 100

hxind = [(dx,npad,-1.3),(dx, 2*nC),(dx,npad,1.3)]
hyind = [(dx,npad,-1.3),(dx, 2*nC),(dx,npad,1.3)]
hzind = [(dx,npad,-1.3),(dx, nC),(dx,npad,1.3)]

mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CC0')
mesh._x0[2] = -np.sum(mesh.hz[:(nC/2+npad)])

# Set background conductivity
model = np.ones(mesh.nC) * sig[0]

# Sphere anomaly
ind = Utils.ModelBuilder.getIndicesSphere(loc, radi, mesh.gridCC)
model[ind] = sig[1]

# Create quick topo
xtopo, ytopo = np.meshgrid(mesh.vectorNx,mesh.vectorNy)
ztopo = np.zeros_like(xtopo)

topo = np.c_[mkvc(xtopo),mkvc(ytopo),mkvc(ztopo)]

# Write out topo file
fid = open('Topo.topo','w')
fid.write(str(topo.shape[0])+'\n')
np.savetxt(fid, topo, fmt='%f',delimiter=' ',newline='\n')
fid.close()

ind = Utils.surface2ind_topo(mesh,topo,gridLoc='N')

# Change aircells
model[ind==0] = air

# Write out model
Mesh.TensorMesh.writeUBC(mesh,'Mesh.msh')
Mesh.TensorMesh.writeModelUBC(mesh,mfile,model)

# Create survey locs centered about origin and write to file
locx, locy = np.meshgrid(np.linspace(-xlim,xlim,nstn), np.linspace(-xlim,xlim,nstn))
locz = np.ones_like(locx) * rx_height

rxLoc = np.c_[mkvc(locx),mkvc(locy),mkvc(locz)]

nSrc = rxLoc.shape[0]
# Create a plane of observation through the grid for display
#pLocx, pLocz = np.meshgrid(np.linspace(-100,100,100), np.linspace(-50,50,50))
#pLocy = np.ones_like(pLocx) * mesh.vectorCCy[nC/2]
#
#pLocs = np.c_[mkvc(pLocx),mkvc(pLocy),mkvc(pLocz)]

# Write out loc file
fid = open('XYZ.loc', 'w')
np.savetxt(fid, rxLoc, fmt='%f',delimiter=' ',newline='\n')
fid.close()


# Write out time file
fid = open('times.dat', 'w')
np.savetxt(fid, d_tc, fmt='%f', delimiter=' ', newline='\n')
fid.close()


# Write out wave file
fid = open('wave.dat', 'w')
np.savetxt(fid, wave, fmt='%f', delimiter=' ', newline='\n')
fid.close()


fid = open('H3DTD_Sphere.loc', 'w')
fid.write('N_TRX ' + str(nSrc) + '\n\n')

for ii in range(nSrc):
    fid.write('TRX_LOOP\n')
    np.savetxt(fid, np.r_[rxLoc[ii], 20., 0, 0].reshape((1,6)), fmt='%e',delimiter=' ',newline='\n\n')
    fid.write('N_RECV 1\n')

    np.savetxt(fid, rxLoc[ii].reshape((1,3)), fmt='%e',delimiter=' ',newline='\n\n')

fid.close()

#%% Function to read in h3d pred file
def read_h3d_pred(predFile):

    sfile = open(predFile,'r')
    lines = sfile.readlines()

    obs = np.zeros((rxLoc.shape[0]*ntimes,13))
    ii = -1
    for line in lines[1:]:

        temp = np.array(line.split(), dtype=float)

        if np.any(temp):
            ii += 1
            obs[ii,:] = temp

    fid.close()
    return np.asarray(obs)


#%% 
dpred = read_h3d_pred('recv_h3dtd.txt')

dbdz = dpred[:,-1]

# Plot profile and sounding
fig = plt.figure(figsize=(8,8))
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)
pos =  ax2.get_position()

def animate(jj):

    removeFrame2()

    global ax1, ax2, fig

    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    pos =  ax2.get_position()
    ax2.set_position([pos.x0, pos.y0-0.025,  pos.width, pos.height])

    T_stn = np.zeros(ntimes)

    for ii in range(ntimes):

        xloc = dpred[ii::ntimes,0]
        yloc = dpred[ii::ntimes,1]
        sub_d = np.log10(dpred[ii::ntimes,-1])

        #Create a line of data for profile
        xx = np.linspace(xloc.min(),xloc.max(),nstn*10+1)
        yy = np.ones_like(xx) * np.mean(yloc)

        T_grid = sp.interpolate.griddata(np.c_[xloc,yloc],
                                         (sub_d), (xx, yy), method='cubic')


        ind = jj*4
        T_stn[ii] = T_grid[ind]


        ax1.plot(xx,T_grid, c='k', lw=np.sqrt(ii+1))
        ax1.text(xlim,(T_grid[-1]),str(d_tc[ii]) + ' s', fontsize=8,
                 bbox={'facecolor':'white',  'pad':1},
                 horizontalalignment='center', verticalalignment='center')

        ax1.plot([xx[ind]+rx_offset/2,xx[ind]+rx_offset/2],
                 [np.min(T_stn),np.max(T_stn)],
                 c='r', lw=2)

        ax1.text(xx[ind]+rx_offset/2,-8,'Sounding',
                 bbox={'facecolor':'white',  'pad':1},
                 horizontalalignment='center', verticalalignment='center', rotation = 90.)

        ax1.set_xlim([-xlim,xlim])
        ax1.set_ylim([-12,-4])

#    ax1.legend(['I{$H_z$}(-)','R{$H_z$}(+)'], loc=8)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('$\partial B_z/\partial t$')

    ax2.semilogx(d_tc, T_stn, c='r', lw=3)
    ax1.set_title('Profile')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('$\partial B_z/\partial t$')
#    ax2.legend(['I{$H_z$}(-)','R{$H_z$}(+)'], loc =2)
    ax2.set_title('Sounding')
    ax2.grid(True)



def removeFrame2():
    global ax1, ax2, fig
    fig.delaxes(ax1)
    fig.delaxes(ax2)
    plt.draw()

anim = animation.FuncAnimation(fig, animate,
                               frames=ntimes , interval=1000, repeat = False)
#/home/dominiquef/3796_AGIC_Research/DCIP3D/MtISa
anim.save('Time_slice.html', writer=HTMLWriter(embed_frames=True,fps=1))


# Plot
fig2 = plt.figure(figsize=(8,8))
ax1 = plt.subplot(2,1,1)
ax3 = plt.subplot(2,1,2)
pos =  ax3.get_position()
ax4 = fig2.add_axes([pos.x0, pos.y0+0.1,  pos.width, pos.height*0.5])
cb1 = fig.add_axes([pos.x0, pos.y0+0.1,  pos.width, pos.height])

X, Z = mesh.gridCC[:,0].reshape(mesh.vnC, order="F"), mesh.gridCC[:,2].reshape(mesh.vnC, order="F")
Y = mesh.gridCC[:,1].reshape(mesh.vnC, order="F")

temp = np.log10(model)
temp[temp==-8] = np.nan
temp = temp.reshape(mesh.vnC, order='F')

ptemp = temp[:,nC+npad,:].T

def animate(ii):

    global ax1, ax3, ax4, fig2
    removeFrame()

    xloc = dpred[ii::ntimes,0]
    yloc = dpred[ii::ntimes,1]
    sub_d = np.log10(dpred[ii::ntimes,-1])

    #Create a line of data for profile
    xx = np.linspace(locx[0,:].min(),locx[0,:].max(),nstn*10+1)
    yy = np.ones_like(xx) * np.mean(yloc)


    #Re-grid the data for visual
    xx = locx[0,:]
    yy = locy[:,0]

    x, y = np.meshgrid(np.linspace(xx.min(),xx.max(),50),np.linspace(yy.min(),yy.max(),50))

    T_grid = sp.interpolate.griddata(np.c_[xloc,yloc],
                                     (sub_d), (x, y), method='cubic')

    ax1 = plt.subplot(2,1,1)

    vminD, vmaxD = np.floor(T_grid.min()*10)/10-0.1, np.ceil(T_grid.max()*10)/10+0.1
    im1 = plt.contourf(x[0,:],y[:,0],T_grid,25, clim=(vminD,vmaxD), vmin=vminD, vmax=vmaxD)
    plt.scatter(mkvc(locx),mkvc(locy),c='k')
    ax1.set_xticks([-xlim,0,xlim])
    ax1.set_yticks([-xlim,0,xlim])
    ax1.set_aspect('equal')
    ax1.set_ylabel('Northing (m)')
    ax1.set_xlabel('Easting (m)')
    plt.title('$\partial B_z/\partial t$')
    pos =  ax1.get_position()
    ax1.set_position([pos.x0+0.1, pos.y0+0.15,  pos.width*0.75, pos.height*0.75])
    cb1 = fig2.add_axes([pos.x0+0.55, pos.y0+0.2,  pos.width*0.04, pos.height*0.4])
    cbar = plt.colorbar(im1,orientation="vertical",ticks=np.linspace(T_grid.min(),T_grid.max(), 3), cax = cb1, format="%.2f")
#    cbar.ax.set_yticklabels([vminD, np.mean([vminD,vmaxD]), vmaxD], format="$%.1f$")

    ax3 = plt.subplot(2,1,2)
#    ps = mesh.plotSlice(np.log10(model),normal = 'Y', ind=nC, ax=ax3, pcolorOpts={'cmap':'RdBu_r'})

    ax3.contourf(X[:,nC,:].T,Z[:,nC,:].T,ptemp,20,vmin=vmin, vmax=vmax, clim=(vmin,vmax), cmap='RdBu_r')
    plt.contour(X[:,nC,:].T,Z[:,nC,:].T,ptemp,1, colors='k', linestyles='solid')
    plt.scatter(mkvc(locx),mkvc(locz),c='r')
    ax3.set_title('')
    ax3.set_aspect('equal')
    ax3.set_xlim([-xlim-10,xlim+10])
    ax3.set_ylim([-100,30])
    plt.show()
#    pos =  ax3.get_position()
#    cb3 = fig.add_axes([pos.x0+0.5, pos.y0+0.05,  pos.width*0.25, pos.height*0.05])
#    plt.colorbar(ps[0],orientation="horizontal",ticks=np.linspace(np.log10(air),np.log10(sig[1]), 3), format="$10^{%.1f}$", cax = cb3, label=' S/m ')
#
    ax3.text(loc[0],loc[2],"$10^{0}$" + ' S/m ',
                 horizontalalignment='center', verticalalignment='center', color='w')

    ax3.text(loc[0]-50,loc[1]-50,"$10^{-2}$" + ' S/m ',
                 horizontalalignment='center', verticalalignment='top', color='w')

    #Create a line of data for profile
    xx = np.linspace(locx[0,:].min(),locx[0,:].max(),50)
    yy = np.ones_like(xx) * np.mean(locy[:,0])

    T_grid = sp.interpolate.griddata(np.c_[xloc,yloc],
                                     (sub_d), (xx, yy), method='cubic')


    pos =  ax3.get_position()
    ax3.set_position([pos.x0, pos.y0-0.05,  pos.width, pos.height])
    ax3.set_xlabel('Easting (m)')
    ax4 = fig2.add_axes([pos.x0, pos.y0+0.3,  pos.width, pos.height*0.4])

    ax4.plot(xx,T_grid,c='k', lw=2)
    ax4.set_xlim([-xlim-10,xlim+10])
    ax4.set_ylim([-12,-4])
    ax4.grid(True)
    ax4.set_ylabel('$\partial B_z/\partial t$')
#    ax4.set_ylim([np.min(np.c_[R_grid,np.abs(I_grid)])*0.5,np.max(np.c_[R_grid,np.abs(I_grid)])*1.5])
#    ax4.legend(['$\partial B_z/\partial t$',])
    ax4.set_title('Profile: N=0 m - Time: '+ str(d_tc[ii]*1e+3) + ' ms')
    ax4.set_xticklabels([])

def removeFrame():
    global ax1, ax3, ax4, fig2
    fig2.delaxes(ax1)
    fig2.delaxes(ax3)
    fig2.delaxes(ax4)
#    fig.delaxes(cb3)
#    fig.delaxes(cb1)
    plt.draw()

anim = animation.FuncAnimation(fig2, animate,
                               frames=ntimes , interval=1000, repeat = False)
#/home/dominiquef/3796_AGIC_Research/DCIP3D/MtISa
anim.save('Data_slice.html', writer=HTMLWriter(embed_frames=True,fps=1))

#%% Export data for inversion

fid = open('H3D_Sphere.obs', 'w')
fid.write('% Export from FDEM_Sphere.py\n')
fid.write('IGNORE nan\n')

#            fid.write('IGNORE -9.9999 \n\n')
fid.write('N_TRX ' + str(nSrc) + '\n\n')

uncert = np.zeros(ntimes)
for ii in range(ntimes):
    
    uncert[ii] = np.min(dpred[ii::ntimes,-1])*2.

count = -1
for ii in range(nSrc):
    fid.write('TRX_LOOP\n')
    np.savetxt(fid, np.r_[rxLoc[ii], 13., 0, 0].reshape((1,6)), fmt='%e',delimiter=' ',newline='\n\n')
    fid.write('N_RECV 1\n')
    fid.write('N_TIMES ' + str(ntimes) + '\n')
    
    for jj in range(ntimes):
        count += 1
        np.savetxt(fid, np.r_[rxLoc[ii],d_tc[jj],np.ones(16)*np.nan,dpred[count,-1],uncert[jj]].reshape((1,22)), fmt='%e',delimiter=' ',newline='\n')

fid.close()

#%% Load 1D data and plot obs vs pred

dpred1D = read_h3d_pred('Pre1D_2_E3Dpre.pre')

fig2 = plt.figure(figsize=(8,8))
ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,1,2)
pos =  ax3.get_position()
ax4 = fig2.add_axes([pos.x0, pos.y0+0.1,  pos.width, pos.height*0.5])
cb1 = fig.add_axes([pos.x0, pos.y0+0.1,  pos.width, pos.height])

mesh = Mesh.TensorMesh.readUBC('Mesh_4m.msh')
m1D = Mesh.TensorMesh.readModelUBC(mesh,'Inv_MOD_iter10.con')

X, Z = mesh.gridCC[:,0].reshape(mesh.vnC, order="F"), mesh.gridCC[:,2].reshape(mesh.vnC, order="F")
Y = mesh.gridCC[:,1].reshape(mesh.vnC, order="F")

temp = np.log10(m1D)
temp[temp==-8] = np.nan
temp = temp.reshape(mesh.vnC, order='F')

indy = int(mesh.nCy/2)
ptemp = temp[:,indy,:].T

vmax = -1
def animate(ii):

    global ax1, ax2, ax3, ax4, fig2
    removeFrame()

    xloc = dpred[ii::ntimes,0]
    yloc = dpred[ii::ntimes,1]
    sub_d = np.log10(dpred[ii::ntimes,-1])
    
    sub_pred = np.log10(dpred1D[ii::ntimes,-1])

    #Create a line of data for profile
    xx = np.linspace(locx[0,:].min(),locx[0,:].max(),nstn*10+1)
    yy = np.ones_like(xx) * np.mean(yloc)


    #Re-grid the data for visual
    xx = locx[0,:]
    yy = locy[:,0]

    x, y = np.meshgrid(np.linspace(xx.min(),xx.max(),50),np.linspace(yy.min(),yy.max(),50))

    T_grid = sp.interpolate.griddata(np.c_[xloc,yloc],
                                     (sub_d), (x, y), method='cubic')

    T_grid_pred = sp.interpolate.griddata(np.c_[xloc,yloc],
                                 (sub_pred), (x, y), method='cubic')
                                     
    ax1 = plt.subplot(2,2,1)

    vminD, vmaxD = np.floor(T_grid.min()*10)/10-0.1, np.ceil(T_grid.max()*10)/10+0.1
    im1 = plt.contourf(x[0,:],y[:,0],T_grid,25, clim=(vminD,vmaxD), vmin=vminD, vmax=vmaxD)
    plt.scatter(mkvc(locx),mkvc(locy),c='k')
    ax1.set_xticks([-xlim,0,xlim])
    ax1.set_yticks([-xlim,0,xlim])
    ax1.set_aspect('equal')
    ax1.set_ylabel('Northing (m)')
    ax1.set_xlabel('Easting (m)')
    plt.title('$\partial B_z^{obs}/\partial t$')
    pos =  ax1.get_position()
    ax1.set_position([pos.x0, pos.y0+0.1,  pos.width*0.75, pos.height*0.75])
    cb1 = fig2.add_axes([pos.x0+0.3, pos.y0+0.15,  pos.width*0.04, pos.height*0.4])
    plt.colorbar(im1,orientation="vertical",ticks=np.linspace(T_grid.min(),T_grid.max(), 3), cax = cb1, format="%.2f", label='Volts')
#    cbar.ax.set_yticklabels([vminD, np.mean([vminD,vmaxD]), vmaxD], format="$%.1f$")

    ax2 = plt.subplot(2,2,2)
    plt.contourf(x[0,:],y[:,0],T_grid_pred,25, clim=(vminD,vmaxD), vmin=vminD, vmax=vmaxD)
    plt.scatter(mkvc(locx),mkvc(locy),c='k')
    ax2.set_xticks([-xlim,0,xlim])
    ax2.set_yticks([])
    ax2.set_aspect('equal')
#    ax2.set_ylabel('Northing (m)')
    ax2.set_xlabel('Easting (m)')
    plt.title('$\partial B_z^{pred}/\partial t$')
    pos =  ax2.get_position()
    ax2.set_position([pos.x0+0.05, pos.y0+0.1,  pos.width*0.75, pos.height*0.75])
    
    
    ax3 = plt.subplot(2,1,2)
#    ps = mesh.plotSlice(np.log10(model),normal = 'Y', ind=nC, ax=ax3, pcolorOpts={'cmap':'RdBu_r'})

    im2 = ax3.contourf(X[:,indy,:].T,Z[:,indy,:].T,ptemp,20,vmin=vmin, vmax=vmax, clim=(vmin,vmax), cmap='RdBu_r')
    plt.scatter(mkvc(locx),mkvc(locz),c='r')
    
    pos =  ax3.get_position()
    cb1 = fig2.add_axes([pos.x0+0.27, pos.y0+0.05,  pos.width*0.3, pos.height*0.05])
    plt.colorbar(im2,orientation="horizontal",ticks=np.linspace(vmin,vmax, 3), cax = cb1, format="$10^{%.1f}$", label='S/m')
#    cbar.ax.set_yticklabels([vminD, np.mean([vminD,vmaxD]), vmaxD], format="$%.1f$")
    
    ax3.set_title('')
    ax3.set_aspect('equal')
    ax3.set_xlim([-xlim-10,xlim+10])
    ax3.set_ylim([-100,30])
    plt.show()

    #Create a line of data for profile
    xx = np.linspace(locx[0,:].min(),locx[0,:].max(),50)
    yy = np.ones_like(xx) * np.mean(locy[:,0])

    T_grid = sp.interpolate.griddata(np.c_[xloc,yloc],
                                     (sub_d), (xx, yy), method='cubic')
                                     
    T_grid_pred = sp.interpolate.griddata(np.c_[xloc,yloc],
                                     (sub_pred), (xx, yy), method='cubic')

    pos =  ax3.get_position()
    ax3.set_position([pos.x0, pos.y0-0.05,  pos.width, pos.height])
    ax3.set_xlabel('Easting (m)')
    ax4 = fig2.add_axes([pos.x0, pos.y0+0.3,  pos.width, pos.height*0.4])

    ax4.plot(xx,T_grid,c='k', lw=2)
    ax4.plot(xx,T_grid,c='r', lw=2)
    ax4.set_xlim([-xlim-10,xlim+10])
    ax4.set_ylim([-12,-4])
    ax4.grid(True)
    ax4.set_ylabel('$\partial B_z/\partial t$')
#    ax4.set_ylim([np.min(np.c_[R_grid,np.abs(I_grid)])*0.5,np.max(np.c_[R_grid,np.abs(I_grid)])*1.5])
    ax4.legend(['Observed','Predicted'])
    ax4.set_title('Profile: N=0 m - Time: '+ str(d_tc[ii]*1e+3) + ' ms')
    ax4.set_xticklabels([])

def removeFrame():
    global ax1,ax2, ax3, ax4, fig2
    fig2.delaxes(ax1)
    fig2.delaxes(ax2)
    fig2.delaxes(ax3)
    fig2.delaxes(ax4)
#    fig.delaxes(cb3)
#    fig.delaxes(cb1)
    plt.draw()

anim = animation.FuncAnimation(fig2, animate,
                               frames=ntimes , interval=1000, repeat = False)
#/home/dominiquef/3796_AGIC_Research/DCIP3D/MtISa
anim.save('Pred_slice.html', writer=HTMLWriter(embed_frames=True,fps=1))
