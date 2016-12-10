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

# Location and radisu of the sphere
loc = [0,0,-40.]
radi = 30.

# Survey parameters
rx_height = 20.
rx_offset = 8.
freqs = [400, 900, 2100, 5000, 12000, 26000, 60000, 140000] #freqs = [400]

# Number of stations along x-y
nstn = 9
xlim = 150

# First we need to create a mesh and a model.
# This is our mesh
dx = 2.
nC = 30
npad = 0

# Floor uncertainties for e3D inversion
floor = 100

hxind = [(dx,npad,-1.3),(dx, 2*nC),(dx,npad,1.3)]
hyind = [(dx, 2*nC)]
hzind = [(dx, 3*nC)]

mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CC0')
mesh._x0[2] = -np.sum(mesh.hz[:(2*nC)])

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

# Create a plane of observation through the grid for display
#pLocx, pLocz = np.meshgrid(np.linspace(-100,100,100), np.linspace(-50,50,50))
#pLocy = np.ones_like(pLocx) * mesh.vectorCCy[nC/2]
#
#pLocs = np.c_[mkvc(pLocx),mkvc(pLocy),mkvc(pLocz)]

# Write out topo file
fid = open('XYZ.loc','w')
np.savetxt(fid, rxLoc, fmt='%f',delimiter=' ',newline='\n')
fid.close()


fid = open('E3D_Obs.loc', 'w')
fid.write('! Export from FDEM_Sphere.py\n')
fid.write('IGNORE NaN\n')

fid.write('N_TRX ' + str(rxLoc.shape[0]*len(freqs)) + '\n\n')
for ii in range(rxLoc.shape[0]):

    txLoc = rxLoc[ii,:].copy()
    txLoc[0] -= rx_offset/2.

    for freq in freqs:

        fid.write('TRX_LOOP\n')
        np.savetxt(fid, np.r_[txLoc, 1., 0, 0].reshape((1,6)), fmt='%f',delimiter=' ',newline='\n')
        fid.write('FREQUENCY ' + str(freq) + '\n')
        fid.write('N_RECV 1\n')

        xloc =  rxLoc[ii,0] + rx_offset/2.
        
        np.savetxt(fid, np.r_[xloc,rxLoc[ii,1:], np.ones(24)*np.nan].reshape((1,27)), fmt='%e',delimiter=' ',newline='\n\n')

fid.close()

#%% WRITE AEM CODE FILES
fid = open('AEM_data.obs','w')
count_tx = 0



for ii in range(rxLoc.shape[0]):

    count_tx += 1
    count_fq = 0
    for freq in freqs:
        
        count_fq += 1
        
        fid.write('%i %i %i ' % (count_tx, count_fq, 1))
        np.savetxt(fid, np.ones((1,5))*-99, fmt='%f',delimiter=' ',newline='\n')
        
fid.close()

def loop(cnter, r, nseg):
    
    theta = np.linspace(0,2*np.pi,nseg)
    xx = cnter[0] + r*np.cos(theta)
    yy = cnter[1] + r*np.sin(theta)
    zz = cnter[2] * np.ones_like(xx)
    
    loc = np.c_[xx, yy, zz]
    return loc
    
# Write tx file
fid = open('AEM_tx.dat','w')
count_tx = 0
nseg = 9

for ii in range(rxLoc.shape[0]):
    
    count_tx += 1
    txLoc = rxLoc[ii,:].copy()
    txLoc[0] -= rx_offset/2.

    loc = loop(txLoc,1.,nseg)
    
    np.savetxt(fid, np.c_[count_tx, nseg, 1], fmt='%i',delimiter=' ',newline='\n')
    
    for jj in range(nseg)    :
    
         np.savetxt(fid, loc[jj,:].reshape((1,3)), fmt='%f',delimiter=' ',newline='\n')

fid.close()

# Write rx file
fid = open('AEM_rx.dat','w')
count_tx = 0


for ii in range(rxLoc.shape[0]):
    
    count_tx += 1
    txLoc = rxLoc[ii,:].copy()
    txLoc[0] += rx_offset/2.

    loc = loop(txLoc,1.,9)
    
    np.savetxt(fid, np.c_[count_tx, nseg, 1], fmt='%i',delimiter=' ',newline='\n')
    
    for jj in range(nseg):    
    
         np.savetxt(fid, loc[jj,:].reshape((1,3)), fmt='%f',delimiter=' ',newline='\n')

fid.close()

# Write rx file
fid = open('AEM_freq.dat','w')
count_fq = 0

for freq in freqs:
    
    count_fq +=1

    np.savetxt(fid, np.c_[count_fq, freq], fmt='%f',delimiter=' ',newline='\n')
    
fid.close()
#%% Read in e3d pred file
def read_e3d_pred(predFile):

    sfile = open(predFile,'r')
    lines = sfile.readlines()

    obs = np.zeros((rxLoc.shape[0]*len(freqs),19))
    count = -1
    ii = -1
    for line in lines:
        count += 1

        if re.match('FREQUENCY',line):
            freq = float(re.split('\s+',line)[1])


        if re.match('N_RECV',line):
            ii += 1
            obs[ii,:] = np.r_[freq,[float(x) for x in re.findall("-?\d+.?\d*(?:[Ee]-\d+)?",lines[count+1])]]

    fid.close()
    return np.asarray(obs)
#%%
dpred = read_e3d_pred('Broadband\\Sphere2_dpred0.txt')
dprim = read_e3d_pred('Broadband\\WholeSpace2_dpred0.txt')

# Adjust primary field and convert to ppm
H0true = 0.00017543534291434681*np.pi 
dH0 =  dprim[:,-2] - H0true

Hs_R =  (dpred[:,-2] - dprim[:,-2])/ (H0true) * 1e+6
Hs_I = (dpred[:,-1])/(H0true)* 1e+6

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
    
    R_stn = np.zeros(len(freqs))
    I_stn = np.zeros(len(freqs))
    for ii in range(len(freqs)):
        
        indx = dpred[:,0] == freqs[ii]
        sub_R = Hs_R[indx]
        sub_I = Hs_I[indx]
        
        #Create a line of data for profile
        xx = np.linspace(locx[0,:].min(),locx[0,:].max(),nstn*10+1) 
        yy = np.ones_like(xx) * np.mean(locy[:,0])  
        
        R_grid = sp.interpolate.griddata(np.c_[mkvc(locx),mkvc(locy)], 
                                         (sub_R), (xx, yy), method='cubic')
                                         
        I_grid = sp.interpolate.griddata(np.c_[mkvc(locx),mkvc(locy)], 
                                         (sub_I), (xx, yy), method='cubic')
                                         
        
        
        ind = jj*5             
        R_stn[ii] = R_grid[ind]  
        I_stn[ii] = I_grid[ind] 
        
    
        ax1.semilogy(xx,np.abs(I_grid),c='b',ls='--', lw=np.sqrt(ii+1))                           
        ax1.semilogy(xx,R_grid, c='r', lw=np.sqrt(ii+1))
        ax1.text(xlim,(R_grid[-1]),str(freqs[ii]) + ' Hz',
                 bbox={'facecolor':'white',  'pad':1},
                 horizontalalignment='center', verticalalignment='center')
                 
        ax1.semilogy([xx[ind]+rx_offset/2,xx[ind]+rx_offset/2],
                 [np.min([R_stn[ii],np.abs(I_stn[ii])]),np.max([R_stn[ii],np.abs(I_stn[ii])])],
                 c='k', lw=2)
                 
        ax1.text(xx[ind]+rx_offset/2,200,'Sounding',
                 bbox={'facecolor':'white',  'pad':1},
                 horizontalalignment='center', verticalalignment='center', rotation = 90.)
                 
        ax1.set_xlim([-xlim,xlim])
        ax1.set_ylim([5,1e+4])
    
#    ax1.legend(['I{$H_z$}(-)','R{$H_z$}(+)'], loc=8)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('|$H_z$|')
    
    ax2.semilogx(np.asarray(freqs),np.abs(I_stn),c='b',ls='--', lw=3)
    ax2.semilogx(np.asarray(freqs), np.abs(R_stn), c='r', lw=3)
    ax1.set_title('Profile')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('|$H_z$|')
    ax2.legend(['I{$H_z$}(-)','R{$H_z$}(+)'], loc =2)
    ax2.set_title('Sounding')
    ax2.grid(True)

    

def removeFrame2():
    global ax1, ax2, fig
    fig.delaxes(ax1)
    fig.delaxes(ax2)
    plt.draw()

anim = animation.FuncAnimation(fig, animate,
                               frames=10 , interval=1000, repeat = False)
#/home/dominiquef/3796_AGIC_Research/DCIP3D/MtISa
anim.save('Freq_slice.html', writer=HTMLWriter(embed_frames=True,fps=1))


# Plot
fig = plt.figure(figsize=(8,8))
ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,1,2)
pos =  ax3.get_position()
ax4 = fig.add_axes([pos.x0, pos.y0+0.1,  pos.width, pos.height*0.5])
cb3 = fig.add_axes([pos.x0, pos.y0+0.1,  pos.width, pos.height])

def animate(ii):
    
    global ax1, ax2, ax3, ax4, cb3, fig
    removeFrame()
    
    indx = dpred[:,0] == freqs[ii]
    sub_R = Hs_R[indx]
    sub_I = Hs_I[indx]

    #Re-grid the data for visual
    xx = locx[0,:]
    yy = locy[:,0]
    
    X, Y = np.meshgrid(np.linspace(xx.min(),xx.max(),50),np.linspace(yy.min(),yy.max(),50))
    
    R_grid = sp.interpolate.griddata(np.c_[mkvc(locx),mkvc(locy)], 
                                     (sub_R), (X, Y), method='cubic')
                     

    I_grid = sp.interpolate.griddata(np.c_[mkvc(locx),mkvc(locy)], 
                                     ((sub_I)), (X, Y), method='cubic')
                                             
    ax1 = plt.subplot(2,2,1)
    
    vmin, vmax = np.floor(R_grid.min()-10), np.ceil(R_grid.max()+10)
    im1 = plt.contourf(X[0,:],Y[:,0],R_grid,25, clim=(vmin,vmax), vmin=vmin, vmax=vmax)
    plt.scatter(mkvc(locx),mkvc(locy),c='k')
    ax1.set_xticks([-xlim,0,xlim])
    ax1.set_yticks([-xlim,0,xlim])
    ax1.set_aspect('equal')
    ax1.set_ylabel('Northing (m)')
    plt.title('Real{$H_z$} (ppm)')
    pos =  ax1.get_position()
    ax1.set_position([pos.x0+0.05, pos.y0+0.15,  pos.width*0.75, pos.height*0.75])
    cb1 = fig.add_axes([pos.x0+0.1, pos.y0+0.1,  pos.width*0.5, pos.height*0.05]) 
    plt.colorbar(im1,orientation="horizontal",ticks=np.round(np.linspace(R_grid.min(),R_grid.max(), 3)), cax = cb1) 

    
    ax2 = plt.subplot(2,2,2)
    vmin, vmax = np.floor(I_grid.min()-10), np.ceil(I_grid.max()+10)
    im2 = plt.contourf(X[0,:],Y[:,0],I_grid,25, clim=(vmin,vmax), vmin=vmin, vmax=vmax)
    plt.scatter(mkvc(locx),mkvc(locy),c='k')
    ax2.set_aspect('equal')
    ax2.set_xticks([-xlim,0,xlim])
    ax2.set_yticks([-xlim,0,xlim])
    ax2.set_yticklabels([])
    plt.title('Imag{$H_z$} (ppm)')
    pos =  ax2.get_position()
    ax2.set_position([pos.x0, pos.y0+0.15,  pos.width*0.75, pos.height*0.75])
    cb2 = fig.add_axes([pos.x0+0.05, pos.y0+0.1,  pos.width*0.5, pos.height*0.05]) 
    plt.colorbar(im2,orientation="horizontal",ticks=np.round(np.linspace(I_grid.min(),I_grid.max(), 3)), cax = cb2) 

    ax3 = plt.subplot(2,1,2)
    ps = mesh.plotSlice(np.log10(model),normal = 'Y', ind=nC, ax=ax3, pcolorOpts={'cmap':'RdBu_r'})
    plt.scatter(mkvc(locx),mkvc(locz),c='r')
    ax3.set_title('')
    ax3.set_aspect('equal')
    ax3.set_xlim([-xlim-10,xlim+10])
    ax3.set_ylim([-100,30])
    pos =  ax3.get_position()
    cb3 = fig.add_axes([pos.x0+0.5, pos.y0+0.05,  pos.width*0.25, pos.height*0.05]) 
    plt.colorbar(ps[0],orientation="horizontal",ticks=np.linspace(np.log10(air),np.log10(sig[1]), 3), format="$10^{%.1f}$", cax = cb3, label=' S/m ') 
    
    #Create a line of data for profile
    xx = np.linspace(locx[0,:].min(),locx[0,:].max(),50) 
    yy = np.ones_like(xx) * np.mean(locy[:,0])  
    
    R_grid = sp.interpolate.griddata(np.c_[mkvc(locx),mkvc(locy)], 
                                     (sub_R), (xx, yy), method='cubic')
                                     
    I_grid = sp.interpolate.griddata(np.c_[mkvc(locx),mkvc(locy)], 
                                     (sub_I), (xx, yy), method='cubic')
                                     
 
    
    pos =  ax3.get_position()
    ax3.set_position([pos.x0, pos.y0-0.05,  pos.width, pos.height])
    ax3.set_xlabel('Easting (m)')
    ax4 = fig.add_axes([pos.x0, pos.y0+0.3,  pos.width, pos.height*0.4])
    
    ax4.semilogy(xx,R_grid,c='r', lw=2)
    ax4.semilogy(xx,np.abs(I_grid),c='b',ls='--', lw=2)                           
    ax4.set_xlim([-xlim-10,xlim+10])
    ax4.set_ylim([10,5e+3])
    ax4.grid(True)
    ax4.set_ylabel('|$H_z$|')
#    ax4.set_ylim([np.min(np.c_[R_grid,np.abs(I_grid)])*0.5,np.max(np.c_[R_grid,np.abs(I_grid)])*1.5])
    ax4.legend(['R{$H_z$}','I{$H_z$}',])
    ax4.set_title('Profile:'+ str(freqs[ii]) + ' Hz')
    ax4.set_xticklabels([])
    
def removeFrame():
    global ax1, ax2, ax3, ax4, cb3, fig
    fig.delaxes(ax1)
    fig.delaxes(ax2)
    fig.delaxes(ax3)
    fig.delaxes(ax4)
    fig.delaxes(cb3)
#    fig.delaxes(cb2)
    plt.draw()

anim = animation.FuncAnimation(fig, animate,
                               frames=len(freqs) , interval=1000, repeat = False)
#/home/dominiquef/3796_AGIC_Research/DCIP3D/MtISa
anim.save('Data_slice.html', writer=HTMLWriter(embed_frames=True,fps=1))

#%% Export data for inversion
fid = open('E3D_Sphere.obs', 'w')
fid.write('! Export from FDEM_Sphere.py\n')
fid.write('IGNORE nan\n')

fid.write('N_TRX ' + str(rxLoc.shape[0]*len(freqs)) + '\n\n')
count = 0
for ii in range(rxLoc.shape[0]):

    txLoc = rxLoc[ii,:].copy()
    txLoc[0] -= rx_offset

    for freq in freqs:
        
        uncert = floor*H0true/1e+6
    
        fid.write('TRX_LOOP\n')
        np.savetxt(fid, np.r_[txLoc, 1., 0, 0].reshape((1,6)), fmt='%f',delimiter=' ',newline='\n')
        fid.write('FREQUENCY ' + str(freq) + '\n')
        fid.write('N_RECV 1\n')

        np.savetxt(fid, np.r_[rxLoc[ii,:], np.ones(20)*np.nan, dpred[count,-2],uncert,dpred[count,-1],uncert ].reshape((1,27)), fmt='%e',delimiter=' ',newline='\n\n')
        
        count+=1
        
fid.close()

