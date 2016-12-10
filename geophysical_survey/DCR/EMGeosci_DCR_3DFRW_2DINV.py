"""
        Experimental script for the forward modeling of DC resistivity data
        along a survey lines defined by the user. Two 3D models a available for
        the demo: two spheres or a block with near-surface conductors.

        Uses SimPEG to generate the forward problem in 3D and generate the
        input files.

        For stype = pole-dipole || dipole-dipole
        Calls DCIP2D for the inversion of a projected 2D section from the full
        3D model.

        For stype = 'gradient'
        Plot apparent resistivity map

        Assumes flat topo for now...

        Created: Mon December 7th, 2015
        Updated: December 9th, 2016

        @author: dominiquef

"""


from SimPEG import Mesh, Utils, mkvc
import SimPEG.EM.Static.DC as DC
import SimPEG.EM.Static.Utils as StaticUtils
import pylab as plt
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import griddata
import time
import re
import numpy.matlib as npm
import scipy.interpolate as interpolation
import os

#home_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\ownCloud\\Research\\MtIsa\\Modeling'
home_dir = '.'
dsep = '\\'

# Specify survey type
stype = 'gradient'
dtype = 'appConductivity'
DOI = False
INVERT = True
# Survey parameters
b = 20 # Tx-Rx seperation
a = 20 # Dipole spacing
n = 15  # Number of Rx per Tx

# Model parameters (background, sphere1, sphere2)
sig = np.r_[1e-2, 1e-1, 1e-3]

# Centroid of spheres
loc = np.c_[[-75., 0., -75.], [75., 0., -75.]]

# Radius of spheres
radi = np.r_[50.,50.]

# Forward solver
slvr = 'BiCGStab'#'LU'

# Preconditioner
pcdr = 'Jacobi'

# Inversion parameter
pct = 0.02 # Percent of abs(obs) value
flr = 2e-5 # Minimum floor value
chifact = 100
ref_mod = ([1e-2, 1e-1])

# DOI threshold
cutoff = 0.8

# number of padding cells
padc = 0

# Plotting param
xmin, xmax = -250, 250
ymin, ymax = -150, 150
zmin, zmax = -125, 25
vmin = -2.4771213
vmax = -1.4771213
depth = 200. # Maximum depth to plot
dx_in = 5

#`srvy_end = [(-200.  ,  0.), (200.  ,  0.)]
srvy_end = [(-225.,  0.), (225.,  0.)]
#%% SCRIPT STARTS HERE
nx = int(np.abs(srvy_end[0][0] - srvy_end[1][0]) /dx_in * 1.25)
ny = nx
ny += (ny+1)%2 # Make sure it is odd so the survey is centered

nz = int(np.abs( np.min(loc[2,:]) - np.max(radi) )  /dx_in )

# Create mesh
hxind = [(dx_in,15,-1.3), (dx_in, nx), (dx_in,15,1.3)]
hyind = [(dx_in,15,-1.3), (dx_in, int(ny/3)), (dx_in,15,1.3)]
hzind = [(dx_in,13,-1.3),(dx_in, nz)]

mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCN')

# Set background conductivity
model = np.ones(mesh.nC) * sig[0]

## SPHERE MODEL
## First anomaly
#ind = Utils.ModelBuilder.getIndicesSphere(loc[:,0],radi[0],mesh.gridCC)
#model[ind] = sig[1]
#
## Second anomaly
#ind = Utils.ModelBuilder.getIndicesSphere(loc[:,1],radi[1],mesh.gridCC)
#model[ind] = sig[2]
#
## Create blocs for pole-dipole
#ind = Utils.ModelBuilder.getIndicesBlock(([-150.,-50.,-20]),([-130.,50.,0]),mesh.gridCC)
#model[ind] = sig[1]/2.5
##
#ind = Utils.ModelBuilder.getIndicesBlock(([-20.,-50.,-20]),([0.,50.,0]),mesh.gridCC)
#model[ind] = sig[2]
##
#ind = Utils.ModelBuilder.getIndicesBlock(([125.,-50.,-20]),([145,50.,0]),mesh.gridCC)
#model[ind] = sig[1]/5

## BLOCK MODEL
# Create blocs for gradient plot
ind = Utils.ModelBuilder.getIndicesBlock(([-25.,-25.,-75]),([25.,25.,-20]),mesh.gridCC)
model[ind] = sig[1]

ind = Utils.ModelBuilder.getIndicesBlock(([-150.,-10.,-25]),([-130.,10.,0]),mesh.gridCC)
model[ind] = sig[1]/2.5
#
ind = Utils.ModelBuilder.getIndicesBlock(([-70.,-10.,-25]),([-50.,10.,0]),mesh.gridCC)
model[ind] = sig[2]
#
ind = Utils.ModelBuilder.getIndicesBlock(([110.,-10.,-25]),([130.,10.,0]),mesh.gridCC)
model[ind] = sig[1]/5.

ind = Utils.ModelBuilder.getIndicesBlock(([-140.,-100.,-25]),([-120.,-80.,0]),mesh.gridCC)
model[ind] = sig[1]/2.5
#
ind = Utils.ModelBuilder.getIndicesBlock(([-20.,-20.,-25]),([0.,0.,0]),mesh.gridCC)
model[ind] = sig[1]/2.5
#
ind = Utils.ModelBuilder.getIndicesBlock(([125.,100.,-25]),([145,120.,0]),mesh.gridCC)
model[ind] = sig[1]/5

ind = Utils.ModelBuilder.getIndicesBlock(([80.,-125.,-25]),([100,-105.,0]),mesh.gridCC)
model[ind] = sig[1]/2.5

ind = Utils.ModelBuilder.getIndicesBlock(([-110.,80.,-25]),([-90,100.,0]),mesh.gridCC)
model[ind] = sig[1]/2.5

#Set boundary conditions
mesh.setCellGradBC('neumann')

Div = mesh.faceDiv
Grad = mesh.cellGrad
Msig = Utils.sdiag(1./(mesh.aveF2CC.T*(1./model)))

A = Div*Msig*Grad

# Change one corner to deal with nullspace
A[0,0] = 1
A = sp.csc_matrix(A)

start_time = time.time()

if re.match(slvr,'BiCGStab'):
    # Create Jacobi Preconditioner
    if re.match(pcdr,'Jacobi'):
        dA = A.diagonal()
        P = sp.spdiags(1/dA,0,A.shape[0],A.shape[0])

        #LDinv = sp.linalg.splu(LD)

elif re.match(slvr,'LU'):
    # Factor A matrix
    Ainv = sp.linalg.splu(A)
    print("LU DECOMP--- %s seconds ---" % (time.time() - start_time))

#%% Create survey
# Display top section
top = int(mesh.nCz)-1

plt.figure()
axs = plt.subplot(1,1,1)
dat1 = mesh.plotSlice(model, ind=-15, normal='Z', grid=False, pcolorOpts={'alpha':0.5}, ax =axs)
axs.set_ylim(ymin,ymax)
axs.set_xlim(xmin,xmax)
plt.gca().set_aspect('equal', adjustable='box')

plt.show()

def plotPoles(survey,stype,axs):
    for ss in range(survey.nSrc):
        tx = np.c_[survey.srcList[ss].loc]

        if stype == 'dipole-dipole':
            axs.scatter(tx[0,0],tx[0,2],c='k',s=25)

        else:
            axs.scatter(tx[0],tx[2],c='k',s=25)

    tx = np.c_[survey.srcList[0].loc]

    if stype == 'dipole-dipole':
        axs.scatter(tx[0,0],tx[0,2],c='r',s=75, marker='v')
        axs.scatter(tx[0,1],tx[1,2],c='b',s=75, marker='v')
    else:
        axs.scatter(tx[0],tx[2],c='r',s=75, marker='v')

#cfm1=get_current_fig_manager().window
#%%
# Add z coordinate to all survey... assume flat
nz = mesh.vectorNz
var = np.c_[np.asarray(srvy_end),np.ones(2).T*nz[-1]]

# Snap the endpoints to the grid. Easier to create 2D section.
indx = Utils.closestPoints(mesh, var )
endl = np.c_[mesh.gridCC[indx,0],mesh.gridCC[indx,1],np.ones(2).T*nz[-1]]

survey = StaticUtils.gen_DCIPsurvey(endl, mesh, stype, a, b, n)
#survey = DC.SurveyDC.Survey(srcList)

Tx = StaticUtils.getSrc_locs(survey)

dl_len = np.sqrt( np.sum((endl[0,:] - endl[1,:])**2) )
dl_x = ( Tx[-1][0] - Tx[0][0] ) / dl_len
dl_y = ( Tx[-1][1] - Tx[0][1]  ) / dl_len
azm =  np.arctan(dl_y/dl_x)

# Plot stations along line
if stype == 'gradient':
    Rx = survey.srcList[0].rxList[0].locs
    plt.scatter(Tx[0][0::3],Tx[0][1::3],s=40,c='r')
    plt.scatter(np.c_[Rx[0][:,0],Rx[1][:,0]],np.c_[Rx[0][:,1],Rx[1][:,1]],s=20,c='y')

#%% Forward model data
data = []#np.zeros( nstn*nrx )
unct = []
problem = DC.Problem3D_CC(mesh)


for src in survey.srcList:
    start_time = time.time()

    # Select dipole locations for receiver
    rxloc_M = np.asarray(src.rxList[0].locs[0])
    rxloc_N = np.asarray(src.rxList[0].locs[1])

    # Number of receivers
    nrx = rxloc_M.shape[0]



    if not re.match(stype,'pole-dipole'):
        tx =  np.squeeze(src.loc)
        inds = Utils.closestPoints(mesh, tx )
        RHS = mesh.getInterpolationMat(tx, 'CC').T*( [-1,1] / mesh.vol[inds] )

    else:

        # Create an "inifinity" pole
        tx =  np.squeeze(src.loc)
        tinf = tx + np.array([dl_x,dl_y,0])*4.*dl_len
        inds = Utils.closestPoints(mesh, np.c_[tx,tinf].T)
        RHS = mesh.getInterpolationMat(np.c_[tx].T, 'CC').T*( [-1] / mesh.vol[inds[0]] )

    # Solve for phi on pole locations
    P1 = mesh.getInterpolationMat(rxloc_M, 'CC')
    P2 = mesh.getInterpolationMat(rxloc_N, 'CC')

    if re.match(slvr,'BiCGStab'):

        if re.match(pcdr,'Jacobi'):
            dA = A.diagonal()
            P = sp.spdiags(1/dA,0,A.shape[0],A.shape[0])

            # Iterative Solve
            Ainvb = sp.linalg.bicgstab(P*A,P*RHS, tol=1e-5)


        phi = mkvc(Ainvb[0])

    elif re.match(slvr,'LU'):
        #Direct Solve
        phi = Ainv.solve(RHS)



    # Compute potential at each electrode
    dtemp = (P1*phi - P2*phi)*np.pi

    data.append( dtemp )
    unct.append( np.abs(dtemp) * pct + flr)

    print("--- %s seconds ---" % (time.time() - start_time))


survey.dobs = np.hstack(data)
survey.std = np.hstack(unct)

if stype != 'gradient':
    #%% Run 2D inversion if pdp or dpdp survey
    # Otherwise just plot and apparent susceptibility map


    #%% Convert 3D obs to 2D and write to file
    survey2D = StaticUtils.convertObs_DC3D_to_2D(survey,np.ones(survey.nSrc),flag = 'Xloc')

#    survey2D = DC.SurveyDC.Survey(srcList2D)
    survey2D.pair(problem)

    survey2D.dobs = survey.dobs
    survey2D.std = survey.std

#    StaticUtils.writeUBC_DCobs(home_dir+'\FWR_3D_2_2D.dat',survey2D,'2D','SURFACE')

    #%% Create a 2D mesh along axis of Tx end points and keep z-discretization
    dx = np.min( [ np.min(mesh.hx), np.min(mesh.hy) ])
    nc = np.ceil(dl_len/dx)+3

    padx = dx*np.power(1.4,range(1,15))

    # Creating padding cells
    h1 = np.r_[padx[::-1], np.ones(nc)*dx , padx]

    # Create mesh with 0 coordinate centerer on the ginput points in cell center
    x0 = srvy_end[0][0] - np.sum(padx) * np.cos(azm)
    y0 = srvy_end[0][1] - np.sum(padx) * np.sin(azm)
    mesh2d = Mesh.TensorMesh([h1, mesh.hz], x0=(x0,mesh.x0[2]))

    # Create array of points for interpolating from 3D to 2D mesh
    xx = x0 + (np.cumsum(mesh2d.hx) - mesh2d.hx/2) * np.cos(azm)
    yy = y0 + (np.cumsum(mesh2d.hx) - mesh2d.hx/2) * np.sin(azm)
    zz = mesh2d.vectorCCy

    [XX,ZZ] = np.meshgrid(xx,zz)
    [YY,ZZ] = np.meshgrid(yy,zz)

    xyz2d = np.c_[mkvc(XX),mkvc(YY),mkvc(ZZ)]

    #plt.scatter(xx,yy,s=20,c='y')


    F = interpolation.NearestNDInterpolator(mesh.gridCC,model)
    m2D = np.reshape(F(xyz2d),[mesh2d.nCx,mesh2d.nCy]).T


    #%% Plot a section through the spheres
    fig = plt.figure(figsize=(8,9))

    axs = plt.subplot(3,1,1, aspect='equal')

    pos =  axs.get_position()
    axs.set_position([pos.x0 , pos.y0 + 0.05 ,  pos.width, pos.height])

    im1 = axs.pcolormesh(mesh2d.vectorCCx,mesh2d.vectorCCy,np.log10(m2D),clim=(vmin,vmax), vmin=vmin, vmax =vmax)

    # Add colorbar
#    cbar = fig.colorbar(im1, orientation="horizontal", ticks=np.linspace(vmin,vmax, 3), format="$10^{%.1f}$",fraction=0.04)
    #cbar.set_label("Conductivity S/m",size=10)

    plotPoles(survey2D,stype,axs)

    axs.set_ylim(zmin,zmax)
    axs.set_xlim(xmin,xmax)
    plt.gca().set_aspect('equal', adjustable='box')
    x = np.linspace(xmin,xmax, 5)
    axs.set_xticks(map(int, x))
    axs.set_xticklabels(map(str, map(int, x)),size=12)
    z = np.linspace(zmin,zmax, 3)
    axs.set_yticks(map(int, z))
    axs.set_ylabel('Depth (m)')
    axs.set_yticklabels(map(str, map(int, z)),size=12)
    axs.axes.get_xaxis().set_visible(False)

#    circle1=plt.Circle((loc[0,0]+dx_in/2.,loc[2,0]+dx_in/2.),radi[0],color='w',fill=False, lw=3)
#    circle2=plt.Circle((loc[0,1]+dx_in/2.,loc[2,1]+dx_in/2.),radi[1],color='k',fill=False, lw=3)
#    axs.add_artist(circle1)
#    axs.add_artist(circle2)

    #plt.plot((blocs[0, 0], blocs[0, 1]), (loc[2, 0], loc[2, 1]))



#    fig.savefig('TwoSphere_model.png')

    #%% Plot pseudo section
    #fig, axs = plt.subplots(1,1, figsize = (6,4))

    axs = plt.subplot(3,1,2, aspect='equal')

    pos =  axs.get_position()
    axs.set_position([pos.x0 , pos.y0 + 0.05 ,  pos.width, pos.height])

    # SEOGI: We need to be able to output the midz location for replacing
    # zlabel to n-spacing, or do it directly in the ploting function
    ph, ax, cbar, LEG = StaticUtils.plot_pseudoSection(survey2D,axs,surveyType=stype, dataType= dtype, clim = [vmin,vmax], scale='log')
    plt.gca().set_aspect('equal', adjustable='box')
    axs.set_ylim(zmin,zmax)
    axs.set_xlim(xmin,xmax)
    plt.gca().set_aspect('equal', adjustable='box')
    x = np.linspace(xmin,xmax, 5)
    axs.set_xticks(map(int, x))
    axs.set_xticklabels(map(str, map(int, x)),size=12)
#    z = np.linspace(-n*a,0, 5)
#    z_label = np.linspace(n,1, 5)
#    axs.set_yticks(map(int, z))
#    axs.set_yticklabels(map(str, map(int, z_label)),size=12)
#    axs.set_ylabel('n-spacing')


    plt.show()

    plt.plot((xmin,xmax),(0,0),color='k',lw=1)

    plotPoles(survey2D,stype,axs)

#    pos =  axs.get_position()
#    cbarax = fig.add_axes([pos.x0 + 0.2 , pos.y0 - 0.025,  pos.width*0.5, pos.height*0.05])  ## the parameters are the specified position you set
#    cbar = fig.colorbar(ph,cax=cbarax, orientation="horizontal", ax = axs, ticks=np.linspace(vmin,vmax, 3), format="$10^{%.1f}$")
#    cbar.set_label('log [App. Cond.]')
    #%% Run two inversions with different reference models and compute a DOI

    invmod = []
    refmod = []
    #plt.figure()
    #==============================================================================
    # fig = plt.figure(figsize=(7,7))
    #==============================================================================
    if INVERT:
        if DOI:

            ninv = range(2)

        else:

            ninv = range(1)

        for jj in ninv:

            # Create dcin2d inversion files and run
            inv_dir = home_dir + '\Inv2D'
            if not os.path.exists(inv_dir):
                os.makedirs(inv_dir)

            mshfile2d = 'Mesh_2D.msh'
            modfile2d = 'Model_2D.con'
            obsfile2d = 'FWR_3D_2_2D.dat'
            inp_file = 'dcinv2d.inp'


            # Export 2D mesh
            fid = open(inv_dir + dsep + mshfile2d,'w')
            fid.write('%i\n'% mesh2d.nCx)
            fid.write('%f %f 1\n'% (mesh2d.vectorNx[0],mesh2d.vectorNx[1]))
            np.savetxt(fid, np.c_[mesh2d.vectorNx[2:],np.ones(mesh2d.nCx-1)], fmt='\t %e %i',delimiter=' ',newline='\n')
            fid.write('\n')
            fid.write('%i\n'% mesh2d.nCy)
            fid.write('%f %f 1\n'%( 0,mesh2d.hy[-1]))
            np.savetxt(fid, np.c_[np.cumsum(mesh2d.hy[-2::-1])+mesh2d.hy[-1],np.ones(mesh2d.nCy-1)], fmt='\t %e %i',delimiter=' ',newline='\n')
            fid.close()

            # Export 2D model
            fid = open(inv_dir + dsep + modfile2d,'w')
            fid.write('%i %i\n'% (mesh2d.nCx,mesh2d.nCy))
            np.savetxt(fid, mkvc(m2D[::-1,:].T), fmt='%e',delimiter=' ',newline='\n')
            fid.close()

            # Export data file
            StaticUtils.writeUBC_DCobs(inv_dir+dsep+obsfile2d,survey2D,'2D','SIMPLE')

            # Write input file
            fid = open(inv_dir + dsep + inp_file,'w')
            fid.write('OBS LOC_X %s \n'% obsfile2d)
            fid.write('MESH FILE %s \n'% mshfile2d)
            fid.write('CHIFACT 2 %f\n'% chifact)
            fid.write('TOPO DEFAULT \n')
            fid.write('INIT_MOD VALUE %e\n'% (ref_mod[jj]))
            fid.write('REF_MOD VALUE %e\n'% (ref_mod[jj]))
            fid.write('ALPHA VALUE %f %f %f\n'% (1./a**2., 1, 1))
            fid.write('WEIGHT DEFAULT\n')
            fid.write('STORE_ALL_MODELS FALSE\n')
            fid.write('INVMODE SVD\n')
            #fid.write('CG_PARAM 200 1e-4\n')
            fid.write('USE_MREF FALSE\n')
            #fid.write('BOUNDS VALUE 1e-4 1e+2\n')
            fid.close()

            os.chdir(inv_dir)
            os.system('dcinv2d ' + inp_file)


            #Load model
            invmod.append( StaticUtils.readUBC_DC2DModel('dcinv2d.con') )
            refmod.append(ref_mod[jj])

            dpre = StaticUtils.readUBC_DC2Dpre('dcinv2d.pre')
            DCpre = dpre['DCsurvey']
            DCtemp = survey2D
            DCtemp.dobs = DCpre.dobs

        #==============================================================================
        #     axs = plt.subplot(2,2,jj+3)
        #     StaticUtils.plot_pseudoSection(DCtemp,axs,surveyType =stype,unitType = dtype, clim = (ph.get_clim()[0],ph.get_clim()[1]), colorbar=False)
        #     axs.set_title('Predicted', fontsize=10)
        #     plt.xlim([xmin,xmax])
        #     plt.ylim([zmin,zmax])
        #     plt.gca().set_aspect('equal', adjustable='box')
        #     axs.set_xticklabels([])
        #     axs.set_yticks(map(int, z))
        #     axs.set_yticklabels(map(str, map(int, z)),rotation='vertical')
        #     axs.set_ylabel('Depth (m)', fontsize=8)
        #==============================================================================



        #%% Replace alpha values from inversion
        if DOI:
            fig = plt.figure(figsize=(7,7))
            #fig, axs = plt.subplots(2,1, figsize = (6,4))



            axs = plt.subplot(3,1,1, aspect='equal')

            pos =  axs.get_position()
            axs.set_position([pos.x0 , pos.y0 + 0.05 ,  pos.width, pos.height])

            minv = np.reshape(invmod[1],(mesh2d.nCy,mesh2d.nCx))
            #plt.pcolormesh(mesh2d.vectorNx,mesh2d.vectorNy,np.log10(m2D),edgecolor="none",alpha=0.5,cmap = 'gray')
            ph = axs.pcolormesh(mesh2d.vectorNx,mesh2d.vectorNy,np.log10(minv),vmin = vmin, vmax = vmax,edgecolor="none")

            axs.set_ylim(zmin,zmax)
            axs.set_xlim(xmin,xmax)
            plt.gca().set_aspect('equal', adjustable='box')
            x = np.linspace(xmin,xmax, 5)
            axs.set_xticks(map(int, x))
            axs.set_xticklabels(map(str, map(int, x)),size=12)
            z = np.linspace(zmin,zmax, 4)
            axs.set_yticks(map(int, z))
            axs.set_yticklabels(map(str, map(int, z)),size=12)

#            circle1=plt.Circle((loc[0,0]+dx_in/2.,loc[2,0]+dx_in/2.),radi[0],color='w',fill=False, lw=3)
#            circle2=plt.Circle((loc[0,1]+dx_in/2.,loc[2,1]+dx_in/2.),radi[1],color='k',fill=False, lw=3)
#            axs.add_artist(circle1)
#            axs.add_artist(circle2)

            plotPoles(survey2D,stype,axs)

    #        if jj == 1:
    #            cbarax = fig.add_axes([pos.x0 + 0.2 , pos.y0 - 0.025,  pos.width*0.5, pos.height*0.05])  ## the parameters are the specified position you set
    #            cbar = fig.colorbar(ph,cax=cbarax, orientation="horizontal", ax = axs, ticks=np.linspace(vmin,vmax, 3), format="$10^{%.1f}$")

            #%% Compute DOI
            DOI = np.abs(invmod[1] - invmod[0]) / np.abs(refmod[1] - refmod[0])
            # Normalize between [0 1]
            DOI = DOI - np.min(DOI)
            DOI = (1.- DOI/np.max(DOI))

            #DOI[DOI > 0.80] = 1

            DOI = np.reshape(DOI,[mesh2d.nCy,mesh2d.nCx])

            axs = plt.subplot(3,1,2, aspect='equal')

            im1 = axs.pcolormesh(mesh2d.vectorCCx,mesh2d.vectorCCy,DOI)

            # Add colorbar
    #        cbar = fig.colorbar(im1, orientation="horizontal", ticks=np.linspace(0,1, 5), format="${%.1f}$",fraction=0.04)
    #        cbar.set_label("Conductivity S/m",size=10)
            pos =  axs.get_position()
            axs.set_position([pos.x0 , pos.y0 + 0.075 ,  pos.width, pos.height])

            axs.set_ylim(zmin,zmax)
            axs.set_xlim(xmin,xmax)
            plt.gca().set_aspect('equal', adjustable='box')
            x = np.linspace(xmin,xmax, 5)
            axs.set_xticks(map(int, x))
            axs.set_xticklabels(map(str, map(int, x)),size=12)
            z = np.linspace(zmin,zmax, 4)
            axs.set_yticks(map(int, z))
            axs.set_yticklabels(map(str, map(int, z)),size=12)
            axs.axes.get_xaxis().set_visible(False)

    #        circle1=plt.Circle((loc[0,0]+dx_in/2.,loc[2,0]+dx_in/2.),radi[0],color='w',fill=False, lw=3)
    #        circle2=plt.Circle((loc[0,1]+dx_in/2.,loc[2,1]+dx_in/2.),radi[1],color='k',fill=False, lw=3)
    #        axs.add_artist(circle1)
    #        axs.add_artist(circle2)

            #plt.plot((blocs[0, 0], blocs[0, 1]), (loc[2, 0], loc[2, 1]))

            for ss in range(survey2D.nSrc):
                tx = survey2D.srcList[ss].loc
                axs.scatter(tx[0],tx[2],c='k',s=25)

            tx = survey2D.srcList[0].loc
            axs.scatter(tx[0],tx[2],c='r',s=75, marker='v')
            axs.scatter(tx[3],tx[2],c='b',s=75, marker='v')

            pos =  axs.get_position()
            cbarax = fig.add_axes([pos.x0 + 0.2 , pos.y0 - 0.04,  pos.width*0.5, pos.height*0.05])  ## the parameters are the specified position you set
            cbar = fig.colorbar(im1,cax=cbarax, orientation="horizontal", ax = axs, ticks=np.linspace(0,1, 5), format="$%.1f$")
            cbar.set_label('DOI Index')

            pos =  axs.get_position()
            cbarax = fig.add_axes([pos.x0 + 0.2 , pos.y0 - 0.08,  pos.width*0.5, pos.height*0.05])  ## the parameters are the specified position you set
            cbar = fig.colorbar(im1,cax=cbarax, orientation="horizontal", ax = axs, ticks=np.linspace(0,1, 5), format="$%.1f$")
            cbar.set_label('DOI Index')


            #%% Plot a section through the spheres
            axs = plt.subplot(3,1,3, aspect='equal')

            #plt.tight_layout(pad=0.5)
            minv = np.reshape(invmod[0],(mesh2d.nCy,mesh2d.nCx))
            im4 = axs.pcolormesh(mesh2d.vectorNx,mesh2d.vectorNy,np.log10(minv),vmin = vmin, vmax = vmax,edgecolor="none")
            plt.show()

    #        # Add colorbar
    #        cbar = fig.colorbar(im4, orientation="horizontal", ticks=np.linspace(vmin,vmax, 3), format="$10^{%.1f}$",fraction=0.04)
    #        cbar.set_label("log(S/m)",size=10)

            rgba_plt = im4.get_facecolor()
            rgba_plt = im4.get_facecolor()
            rgba_plt[:,3] = mkvc(DOI.T)**2.
            im4.set_facecolor(rgba_plt)
            axs.set_ylim(zmin,zmax)
            axs.set_xlim(xmin,xmax)
            plt.gca().set_aspect('equal', adjustable='box')
            x = np.linspace(xmin,xmax, 5)
            axs.set_xticks(map(int, x))
            axs.set_xticklabels(map(str, map(int, x)),size=12)
            z = np.linspace(zmin,zmax, 4)
            axs.set_yticks(map(int, z))
            axs.set_yticklabels(map(str, map(int, z)),size=12)

            circle1=plt.Circle((loc[0,0]+dx_in/2.,loc[2,0]+dx_in/2.),radi[0],color='w',fill=False, lw=3)
            circle2=plt.Circle((loc[0,1]+dx_in/2.,loc[2,1]+dx_in/2.),radi[1],color='k',fill=False, lw=3)
            axs.add_artist(circle1)
            axs.add_artist(circle2)


            #plt.plot((blocs[0, 0], blocs[0, 1]), (loc[2, 0], loc[2, 1]))

            for ss in range(survey2D.nSrc):
                tx = survey2D.srcList[ss].loc
                axs.scatter(tx[0],tx[2],c='k',s=25)

            tx = survey2D.srcList[0].loc
            axs.scatter(tx[0],tx[2],c='r',s=75, marker='v')
            axs.scatter(tx[3],tx[2],c='b',s=75, marker='v')

            pos =  axs.get_position()
            cbarax = fig.add_axes([pos.x0 + 0.2 , pos.y0 - 0.04,  pos.width*0.5, pos.height*0.05])  ## the parameters are the specified position you set
            cbar = fig.colorbar(im4,cax=cbarax, orientation="horizontal", ax = axs, ticks=np.linspace(vmin,vmax, 3))
            cbar.ax.set_xticklabels([str(int(10**-vmin)), str(int(np.round(10**(-np.round(vmin+vmax)/2)/10)*10)), str(int(10**-vmax))])
            cbar.set_label('Resistivity ($\Omega \cdot m$)')

        else:
            axs = plt.subplot(3,1,3, aspect='equal')
            pos =  axs.get_position()
            axs.set_position([pos.x0 , pos.y0 + 0.05 ,  pos.width, pos.height])

            minv = np.reshape(invmod[jj],(mesh2d.nCy,mesh2d.nCx))
            #plt.pcolormesh(mesh2d.vectorNx,mesh2d.vectorNy,np.log10(m2D),edgecolor="none",alpha=0.5,cmap = 'gray')
            ph = axs.pcolormesh(mesh2d.vectorNx,mesh2d.vectorNy,np.log10(minv),vmin = vmin, vmax = vmax,edgecolor="none")

            axs.set_ylim(zmin,zmax)
            axs.set_xlim(xmin,xmax)
            plt.gca().set_aspect('equal', adjustable='box')
            x = np.linspace(xmin,xmax, 5)
            axs.set_xticks(map(int, x))
            axs.set_xticklabels(map(str, map(int, x)),size=12)
            z = np.linspace(zmin,zmax, 4)
            axs.set_yticks(map(int, z))
            axs.set_yticklabels(map(str, map(int, z)),size=12)
            axs.set_ylabel('Depth (m)')

    #        circle1=plt.Circle((loc[0,0]+dx_in/2.,loc[2,0]+dx_in/2.),radi[0],color='w',fill=False, lw=3)
    #        circle2=plt.Circle((loc[0,1]+dx_in/2.,loc[2,1]+dx_in/2.),radi[1],color='k',fill=False, lw=3)
    #        axs.add_artist(circle1)
    #        axs.add_artist(circle2)

            pos =  axs.get_position()
            cbarax = fig.add_axes([pos.x0 + 0.2 , pos.y0 - 0.04,  pos.width*0.5, pos.height*0.05])  ## the parameters are the specified position you set
            cbar = fig.colorbar(ph,cax=cbarax, orientation="horizontal", ax = axs, ticks=np.linspace(vmin,vmax, 3))
            cbar.ax.set_xticklabels([str(int(10**-vmin)), str(int(np.round(10**(-(vmin+vmax)/2)/10)*10)), str(int(10**-vmax))])
            cbar.set_label('Resistivity ($\Omega \cdot m$)')

            plotPoles(survey2D,stype,axs)

    else:
        pos =  axs.get_position()
        cbarax = fig.add_axes([pos.x0 + 0.2 , pos.y0 - 0.04,  pos.width*0.5, pos.height*0.05])  ## the parameters are the specified position you set
        cbar = fig.colorbar(ph,cax=cbarax, orientation="horizontal", ax = axs, ticks=np.linspace(vmin,vmax, 3))
        cbar.ax.set_xticklabels([str(int(10**-vmin)), str(int(np.round(10**(-np.round(vmin+vmax)/2)/10)*10)), str(int(10**-vmax))])
        cbar.set_label('App. Resistivity ($\Omega \cdot m$)')
    # Second plot for the predicted apparent resistivity data
#    ax2 = plt.subplot(2,1,2, aspect='equal')
#    plt.pcolor(mesh2d.vectorNx,mesh2d.vectorNy,np.log10(m2D),edgecolor="none",alpha=0.5,cmap = 'gray')
#    # Add the speudo section
#    dat = DC.plot_pseudoSection(survey2D,ax2,stype=stype, dtype = dtype, alpha = 0.5)
#    ax2.set_title('Pseudo-conductivity')
#
#    plt.xlim([survey2D.srcList[0].loc[0][0]-a,survey2D.srcList[-1].rxList[-1].locs[-1][0][0]+a])
#    plt.ylim([mesh2d.vectorNy[-1]-zmin,mesh2d.vectorNy[-1]+2*dx])


#%% Othrwise it is a gradient array, plot surface of apparent resisitivty
elif re.match(stype,'gradient'):
    Rx = survey.srcList[0].rxList[0].locs
    rC1P1 = np.sqrt( np.sum( (npm.repmat(Tx[0][0:2],Rx[0].shape[0], 1) - Rx[0][:,0:2])**2, axis=1 ))
    rC2P1 = np.sqrt( np.sum( (npm.repmat(Tx[0][3:5],Rx[0].shape[0], 1) - Rx[0][:,0:2])**2, axis=1 ))
    rC1P2 = np.sqrt( np.sum( (npm.repmat(Tx[0][0:2],Rx[0].shape[0], 1) - Rx[1][:,0:2])**2, axis=1 ))
    rC2P2 = np.sqrt( np.sum( (npm.repmat(Tx[0][3:5],Rx[0].shape[0], 1) - Rx[1][:,0:2])**2, axis=1 ))

    rC1C2 = np.sqrt( np.sum( (npm.repmat(Tx[0][0:2]-Tx[0][3:5],Rx[0].shape[0], 1) )**2, axis=1 ))
    rP1P2 = np.sqrt( np.sum( (Rx[0][:,0:2] - Rx[1][:,0:2])**2, axis=1 ))

    rho = np.abs(data[0]) * 2.*np.pi / ( 1/rC1P1 - 1/rC2P1 - 1/rC1P2 + 1/rC2P2 )

    Pmid = (Rx[0][:,0:2] + Rx[1][:,0:2])/2

    # Grid points
    grid_x, grid_z = np.mgrid[np.min(Rx[0][:,0]):np.max(Rx[1][:,0]):a/10, np.min(Rx[0][:,1]):np.max(Rx[1][:,1]):a/10]
    grid_rho = griddata(np.c_[Pmid[:,0],Pmid[:,1]], (abs(rho.T)), (grid_x, grid_z), method='cubic')


    #plt.subplot(2,1,2)


    fig = plt.figure(figsize = (8,8))
    axs = plt.subplot(2,1,1)
    dat1 = mesh.plotSlice(np.log10(model), ind=-5, normal='Z', grid=False, pcolorOpts={'alpha':0.5}, ax =axs, clim=(vmin,vmax))
    axs.set_ylim(ymin,ymax)
    axs.set_xlim(xmin,xmax)
    axs.axes.get_xaxis().set_visible(False)
    axs.axes.set_title('')
    plt.gca().set_aspect('equal', adjustable='box')


    # Plot stations along line
    plt.scatter(Tx[0][0::3],Tx[0][1::3],s=40,c='r')
    plt.scatter(np.c_[Rx[0][:,0],Rx[1][:,0]],np.c_[Rx[0][:,1],Rx[1][:,1]],s=10,c='y')
    plt.show()


    #plt.tight_layout(pad=0.5)
    axs = plt.subplot(2,1,2)
    im3 = plt.contourf(np.log10(1./grid_rho.T),10, extent = (np.min(grid_x),np.max(grid_x),np.min(grid_z),np.max(grid_z))  ,origin='lower',clim=(vmin,vmax),vmin=vmin,vmax=vmax)
    plt.contour(np.log10(1./grid_rho.T),10, extent = (np.min(grid_x),np.max(grid_x),np.min(grid_z),np.max(grid_z))  ,origin='lower',colors='k')

    #var = 'Gradient Array - a-spacing: ' + str(a) + ' m'
    #plt.title(var)
    plt.scatter(Tx[0][0::3],Tx[0][1::3],s=40,c='k')
    plt.scatter(np.c_[Rx[0][:,0],Rx[1][:,0]],np.c_[Rx[0][:,1],Rx[1][:,1]],s=10,c='k')
    plt.show()

    axs.set_ylim(ymin,ymax)
    axs.set_xlim(xmin,xmax)
    plt.gca().set_aspect('equal', adjustable='box')
    x = np.linspace(xmin,xmax, 5)
    axs.set_xticks(map(int, x))
    axs.set_xticklabels(map(str, map(int, x)),size=12)
    y = np.linspace(ymin,ymax, 5)
    axs.set_yticks(map(int, y))
    axs.set_yticklabels(map(str, map(int, y)),size=12)

    axs.set_xlabel('x')
    axs.set_ylabel('y')

    pos =  axs.get_position()
    axs.set_position([pos.x0 , pos.y0 +0.05,  pos.width, pos.height])  ## the parameters are the specified position you set
    cbarax = fig.add_axes([pos.x0 + 0.09 , pos.y0 - 0.025,  pos.width*0.75, pos.height*0.05])  ## the parameters are the specified position you set
    cbar = fig.colorbar(im3,cax=cbarax, orientation="horizontal", ax = axs, ticks=np.linspace(np.min(np.log10(1./rho)),np.max(np.log10(1./rho)), 3))
    cbar.ax.set_xticklabels([str(int(10**-np.min(np.log10(1./rho)))), str(int(np.round(10**(-np.round(np.min(np.log10(1./rho))+np.max(np.log10(1./rho)))/2)/10)*10)), str(int(round(10**-np.max(np.log10(1./rho)))))])
    cbar.set_label('App. Resistivity ($\Omega \cdot m$)')
