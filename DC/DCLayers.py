import sys
sys.path.append("./simpeg/")
sys.path.append("./simpegdc/")

import warnings
warnings.filterwarnings('ignore')

from SimPEG import *
import simpegDCIP as DC
from scipy.constants import epsilon_0

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    from ipywidgets import interact, IntSlider, FloatSlider, FloatText, ToggleButtons
    pass
except Exception, e:
    from IPython.html.widgets import  interact, IntSlider, FloatSlider, FloatText, ToggleButtons


npad = 20
cs = 0.5
hx = [(cs,npad, -1.3),(cs,200),(cs,npad, 1.3)]
hy = [(cs,npad, -1.3),(cs,100)]
mesh = Mesh.TensorMesh([hx, hy], "CN")

rhomin = 1e2
rhomax = 1e3

eps = 1e-9 #to stabilize division

def get_Layer_Potentials(rho1,rho2,h,A,B,xyz,infty=100):
    k = (rho2-rho1) / (rho2+rho1)
    
    r = lambda src_loc: np.sqrt((xyz[:,0] - src_loc[0])**2 + (xyz[:,1] - src_loc[1])**2 + (xyz[:,2] - src_loc[2])**2)+eps

    m = Utils.mkvc(np.arange(1,infty+1))
    sum_term = lambda r: np.sum(((k**m.T)*np.ones_like(Utils.mkvc(r,2))) / np.sqrt(1. + (2.*h*m.T/Utils.mkvc(r,2))**2),1)
    
    V = lambda I,src_loc: (I*rho1 / (2.*np.pi*r(src_loc))) * (1 + 2*sum_term(r(src_loc)))
    
    VA = V(1.,A)
    VB = V(-1.,B)
    
    return VA+VB

def get_Layer_E(rho1,rho2,h,A,B,xyz,infty=100):
    k = (rho2-rho1) / (rho2+rho1)
    
    r = lambda src_loc: np.sqrt((xyz[:,0] - src_loc[0])**2 + (xyz[:,1] - src_loc[1])**2 + (xyz[:,2] - src_loc[2])**2)+eps

    dr_dx = lambda src_loc: (xyz[:,0] - src_loc[0]) / r(src_loc)
    dr_dy = lambda src_loc: (xyz[:,1] - src_loc[1]) / r(src_loc)
    dr_dz = lambda src_loc: (xyz[:,2] - src_loc[2]) / r(src_loc)

    m = Utils.mkvc(np.arange(1,infty+1))

    sum_term = lambda r: np.sum(((k**m.T)*np.ones_like(Utils.mkvc(r,2))) / np.sqrt(1. + (2.*h*m.T/Utils.mkvc(r,2))**2),1)

    sum_term_deriv = lambda r: np.sum(((k**m.T)*np.ones_like(Utils.mkvc(r,2))) / (1. + (2.*h*m.T/Utils.mkvc(r,2))**2)**(3./2.) * ((2.*h*m.T)**2 / Utils.mkvc(r,2)**3) ,1)

    deriv_1 = lambda r: (-1./r) * (1. + 2.*sum_term(r))
    deriv_2 = lambda r: (2.*sum_term_deriv(r))

    Er = lambda I,r : - (I*rho1 / (2.*np.pi*r)) * (deriv_1(r) + deriv_2(r))

    Ex = lambda I,src_loc : Er(I,r(src_loc)) * dr_dx(src_loc)
    Ey = lambda I,src_loc : Er(I,r(src_loc)) * dr_dy(src_loc)
    Ez = lambda I,src_loc : Er(I,r(src_loc)) * dr_dz(src_loc)
    
    ex = Ex(1.,A) + Ex(-1.,B)
    ey = Ey(1.,A) + Ey(-1.,B)
    ez = Ez(1.,A) + Ez(-1.,B)

    return ex, ey, ez

def get_Layer_J(rho1,rho2,h,A,B,xyz,infty=100):
    ex, ey, ez = get_Layer_E(rho1, rho2, h, A, B, xyz)

    sig = 1./rho2*np.ones_like(xyz[:,0])
    # print sig
    sig[xyz[:,1] >= -h] = 1./rho1 # hack for 2D (assuming y is z)

    return sig * ex, sig * ey, sig * ez

G = lambda A, B, M, N: 1. / ( 1./(np.abs(A-M)+eps) - 1./(np.abs(M-B)+eps) - 1./(np.abs(N-A)+eps) + 1./(np.abs(N-B)+eps) )
rho_a = lambda VM,VN, A,B,M,N: (VM-VN)*2.*np.pi*G(A,B,M,N)

def solve_2D_potentials(rho1, rho2, h, A, B):
    sigma = 1./rho2*np.ones(mesh.nC)
    sigma[mesh.gridCC[:,1] >= -h] = 1./rho1 # hack for 2D (assuming y is z)

    q = np.zeros(mesh.nC)
    a = Utils.closestPoints(mesh, A[:2])
    b = Utils.closestPoints(mesh, B[:2])

    q[a] = 1./mesh.vol[a]
    q[b] = -1./mesh.vol[b]

    # q = q * 1./mesh.vol

    A = mesh.cellGrad.T * Utils.sdiag(1./(mesh.dim * mesh.aveF2CC.T * (1./sigma))) * mesh.cellGrad
    Ainv = SolverLU(A)

    V = Ainv * q
    return V

def solve_2D_E(rho1, rho2, h, A, B):
    V = solve_2D_potentials(rho1, rho2, h, A, B)
    E = -mesh.cellGrad * V
    E = mesh.aveF2CCV * E
    ex = E[:mesh.nC]
    ez = E[mesh.nC:]
    return ex, ez, V

def solve_2D_J(rho1, rho2, h, A, B):
    ex, ez, V = solve_2D_E(rho1, rho2, h, A, B)
    sigma = 1./rho2*np.ones(mesh.nC)
    sigma[mesh.gridCC[:,1] >= -h] = 1./rho1 # hack for 2D (assuming y is z)

    return Utils.sdiag(sigma) * ex, Utils.sdiag(sigma) * ez, V  

def plot_Layer_Potentials(rho1,rho2,h,A,B,M,N,imgplt='Model'):
    
    ylim = np.r_[-1., 1.]*rhomax/(5*2*np.pi)

    fig, ax = plt.subplots(2,1,figsize=(9,7))

    fig.subplots_adjust(right=0.8)
    x = np.linspace(-40.,40.,200)
    z = np.linspace(x.min(),0,100)
    
    pltgrid = Utils.ndgrid(x,z)
    xplt = pltgrid[:,0].reshape(x.size,z.size,order='F')
    zplt = pltgrid[:,1].reshape(x.size,z.size,order='F')

    V = get_Layer_Potentials(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.],Utils.ndgrid(x,np.r_[0.],np.r_[0.]))
    VM = get_Layer_Potentials(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.],Utils.mkvc(np.r_[M,0.,0],2).T)
    VN = get_Layer_Potentials(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.],Utils.mkvc(np.r_[N,0.,0],2).T)

    ax[0].plot(x,V,color=[0.1,0.5,0.1],linewidth=2)
    ax[0].grid(which='both',linestyle='-',linewidth=0.5,color=[0.2,0.2,0.2],alpha=0.5)
    ax[0].plot(A,0,'+',markersize = 12, markeredgewidth = 3, color=[1.,0.,0])
    ax[0].plot(B,0,'_',markersize = 12, markeredgewidth = 3, color=[0.,0.,1.])
    ax[0].set_ylabel('Potential, (V)',fontsize = 14)
    ax[0].set_xlabel('x (m)',fontsize = 14)
    ax[0].set_xlim([x.min(),x.max()])
    ax[0].set_ylim(ylim)

    ax[0].plot(M,VM,'o',color='k')
    ax[0].plot(N,VN,'o',color='k')

    props = dict(boxstyle='round', facecolor='grey', alpha=0.3)

    txtsp = 1

    xytextM = (M+0.5,np.max([np.min([VM,ylim.max()]),ylim.min()])+0.5)
    xytextN = (N+0.5,np.max([np.min([VN,ylim.max()]),ylim.min()])+0.5)


    props = dict(boxstyle='round', facecolor='grey', alpha=0.4)

    ax[0].annotate('%2.1e'%(VM), xy=xytextM, xytext=xytextM,fontsize = 14)
    ax[0].annotate('%2.1e'%(VN), xy=xytextN, xytext=xytextN,fontsize = 14)

#     ax[0].plot(np.r_[M,N],np.ones(2)*VN,color='k')
#     ax[0].plot(np.r_[M,M],np.r_[VM, VN],color='k')
#     ax[0].annotate('%2.1e'%(VM-VN) , xy=(M,(VM+VN)/2), xytext=(M-9,(VM+VN)/2.),fontsize = 14)

    props = dict(boxstyle='round', facecolor='grey', alpha=0.4)
    ax[0].text(x.max()+1,ylim.max()-0.1*ylim.max(),'$\\rho_a$ = %2.2f'%(rho_a(VM,VN,A,B,M,N)),
                verticalalignment='bottom', bbox=props, fontsize = 14)

    if imgplt is 'Model':
        model = rho2*np.ones(pltgrid.shape[0])
        model[pltgrid[:,1] >= -h] = rho1
        model = model.reshape(x.size,z.size, order='F')
        cb = ax[1].pcolor(xplt, zplt, model,norm=LogNorm())
        ax[1].plot([xplt.min(),xplt.max()], -h*np.r_[1.,1],color=[0.5,0.5,0.5],linewidth = 1.5 )

        clim = [rhomin,rhomax]
        clabel = 'Resistivity ($\Omega$m)'

    # elif imgplt is 'potential':
    #     Vplt = get_Layer_Potentials(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.],np.c_[pltgrid,np.zeros_like(pltgrid[:,0])])
    #     Vplt = Vplt.reshape(x.size,z.size, order='F')
    #     cb = ax[1].pcolor(xplt,zplt,Vplt)
    #     ax[1].contour(xplt,zplt,np.abs(Vplt),np.logspace(-2.,1.,10),colors='k',alpha=0.5)
    #     ax[1].set_ylabel('z (m)', fontsize=14)
    #     clim = ylim
    #     clabel = 'Potential (V)'

    elif imgplt is 'Potential':
        Pc = mesh.getInterpolationMat(pltgrid,'CC')

        V = solve_2D_potentials(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.])

        Vplt = Pc * V 
        Vplt = Vplt.reshape(x.size,z.size, order='F')

        fudgeFactor = get_Layer_Potentials(rho1,rho2,h, np.r_[A,0.,0.],np.r_[B,0.,0.],np.c_[x.min(),0.,0.] ) / Vplt[0,0] 

        cb = ax[1].pcolor(xplt,zplt,Vplt * fudgeFactor)
        ax[1].plot([xplt.min(),xplt.max()], -h*np.r_[1.,1],color=[0.5,0.5,0.5],linewidth = 1.5 )
        ax[1].contour(xplt,zplt,np.abs(Vplt),colors='k',alpha=0.5)
        ax[1].set_ylabel('z (m)', fontsize=14)
        clim = np.r_[-15., 15.]
        clabel = 'Potential (V)'

    elif imgplt is 'E':

        Pc = mesh.getInterpolationMat(pltgrid,'CC')

        ex, ez, V = solve_2D_E(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.])

        ex, ez = Pc * ex, Pc * ez 
        Vplt = (Pc*V).reshape(x.size,z.size, order='F')
        fudgeFactor = get_Layer_Potentials(rho1,rho2,h, np.r_[A,0.,0.],np.r_[B,0.,0.],np.c_[x.min(),0.,0.] ) / Vplt[0,0]
        

        # ex, ez, _ = get_Layer_E(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.],np.c_[pltgrid,np.zeros_like(pltgrid[:,0])])
        ex = fudgeFactor * ex.reshape(x.size,z.size,order='F')
        ez = fudgeFactor * ez.reshape(x.size,z.size,order='F')
        e = np.sqrt(ex**2.+ez**2.)

        cb = ax[1].pcolor(xplt,zplt,e,norm=LogNorm())
        ax[1].plot([xplt.min(),xplt.max()], -h*np.r_[1.,1],color=[0.5,0.5,0.5],linewidth = 1.5 )
        clim = np.r_[3e-3,1e1]

        ax[1].streamplot(x,z,ex.T,ez.T,color = 'k',linewidth= 2*(np.log(e.T) - np.log(e).min())/(np.log(e).max() - np.log(e).min())) 
        

        clabel = 'Electric Field (V/m)'

    elif imgplt is 'J':

        Pc = mesh.getInterpolationMat(pltgrid,'CC')

        Jx, Jz, V = solve_2D_J(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.])

        Jx, Jz = Pc * Jx, Pc * Jz 

        Vplt = (Pc*V).reshape(x.size,z.size, order='F')
        fudgeFactor = get_Layer_Potentials(rho1,rho2,h, np.r_[A,0.,0.],np.r_[B,0.,0.],np.c_[x.min(),0.,0.] ) / Vplt[0,0]

        Jx = fudgeFactor * Jx.reshape(x.size,z.size,order='F')
        Jz = fudgeFactor * Jz.reshape(x.size,z.size,order='F')

        J = np.sqrt(Jx**2.+Jz**2.)

        cb = ax[1].pcolor(xplt,zplt,J,norm=LogNorm())
        ax[1].plot([xplt.min(),xplt.max()], -h*np.r_[1.,1],color=[0.5,0.5,0.5],linewidth = 1.5 )
        ax[1].streamplot(x,z,Jx.T,Jz.T,color = 'k',linewidth = 2*(np.log(J.T)-np.log(J).min())/(np.log(J).max() - np.log(J).min()) )  
        ax[1].set_ylabel('z (m)', fontsize=14)

        clim = np.r_[3e-5,3e-2]
        clabel = 'Current Density (A/m$^2$)'


    # elif imgplt is 'e':
    #     ex, ez, _ = get_Layer_E(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.],np.c_[pltgrid,np.zeros_like(pltgrid[:,0])])
    #     ex = ex.reshape(x.size,z.size,order='F')
    #     ez = ez.reshape(x.size,z.size,order='F')
    #     e = np.sqrt(ex**2.+ez**2.)
    #     cb = ax[1].pcolor(xplt,zplt,e,norm=LogNorm())

    #     clim = np.r_[3e-2,1e2]

    #     ax[1].streamplot(x,z,ex.T,ez.T,color = 'k',linewidth= 2.5*(np.log(e.T) - np.log(e).min())/np.log(e).max())
        
    #     clabel = 'Electric Field (V/m)'

    # elif imgplt is 'ex':
    #     ex, ez, _ = get_Layer_E(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.],np.c_[pltgrid,np.zeros_like(pltgrid[:,0])])
    #     ex = ex.reshape(x.size,z.size,order='F')
    #     ez = ez.reshape(x.size,z.size,order='F')
    #     e = np.sqrt(ex**2.+ez**2.)
    #     cb = ax[1].pcolor(xplt,zplt,ex) #,norm=LogNorm())
        
    #     clim = np.r_[-20, 20]

    #     # clim = np.r_[3e-2,1e2]

    #     # ax[1].streamplot(x,z,ex.T,ez.T,color = 'k',linewidth= 2.5*(np.log(e.T) - np.log(e).min())/np.log(e).max())
        
    #     clabel = 'Electric Field (V/m)'
    
    # elif imgplt is 'ez':
    #     ex, ez, _ = get_Layer_E(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.],np.c_[pltgrid,np.zeros_like(pltgrid[:,0])])
    #     ex = ex.reshape(x.size,z.size,order='F')
    #     ez = ez.reshape(x.size,z.size,order='F')
    #     e = np.sqrt(ex**2.+ez**2.)
    #     cb = ax[1].pcolor(xplt,zplt,ez) #,norm=LogNorm())

    #     clim = np.r_[-20, 20]

    #     # ax[1].streamplot(x,z,ex.T,ez.T,color = 'k',linewidth= 2.5*(np.log(e.T) - np.log(e).min())/np.log(e).max())
        
    #     clabel = 'Electric Field (V/m)'

    # elif imgplt is 'j':
    #     Jx, Jz, _ = get_Layer_J(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.],np.c_[pltgrid,np.zeros_like(pltgrid[:,0])])

    #     Jx = Jx.reshape(x.size,z.size,order='F')
    #     Jz = Jz.reshape(x.size,z.size,order='F')

    #     J = np.sqrt(Jx**2.+Jz**2.)

    #     cb = ax[1].pcolor(xplt,zplt,J,norm=LogNorm())
    #     ax[1].streamplot(x,z,Jx.T,Jz.T,color = 'k',linewidth = 2.5*(np.log(J.T)-np.log(J).min())/np.max(np.log(J)))   
    #     ax[1].set_ylabel('z (m)', fontsize=14)

    #     clim = np.r_[3e-5,1e-1]
    #     clabel = 'Current Density (A/m$^2$)'

    # elif imgplt is 'jx':
    #     Jx, Jz, _ = get_Layer_J(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.],np.c_[pltgrid,np.zeros_like(pltgrid[:,0])])

    #     Jx = Jx.reshape(x.size,z.size,order='F')
    #     Jz = Jz.reshape(x.size,z.size,order='F')

    #     J = np.sqrt(Jx**2.+Jz**2.)

    #     cb = ax[1].pcolor(xplt,zplt,Jx) #,norm=LogNorm())
    #     # ax[1].streamplot(x,z,Jx.T,Jz.T,color = 'k',linewidth = 2.5*(np.log(J.T)-np.log(J).min())/np.max(np.log(J)))   
    #     ax[1].set_ylabel('z (m)', fontsize=14)

    #     clim = np.r_[-0.05,0.05]
    #     clabel = 'Current Density (A/m$^2$)'

    # elif imgplt is 'jz':
    #     Jx, Jz, _ = get_Layer_J(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.],np.c_[pltgrid,np.zeros_like(pltgrid[:,0])])

    #     Jx = Jx.reshape(x.size,z.size,order='F')
    #     Jz = Jz.reshape(x.size,z.size,order='F')

    #     J = np.sqrt(Jx**2.+Jz**2.)

    #     cb = ax[1].pcolor(xplt,zplt,Jz) #,norm=LogNorm())
    #     # ax[1].streamplot(x,z,Jx.T,Jz.T,color = 'k',linewidth = 2.5*(np.log(J.T)-np.log(J).min())/np.max(np.log(J)))   
    #     ax[1].set_ylabel('z (m)', fontsize=14)

    #     clim = np.r_[-0.05,0.05]
    #     clabel = 'Current Density (A/m$^2$)'

    # elif imgplt is 'e':
    #     Px = mesh.getInterpolationMat(pltgrid,'Fx')
    #     Pz = mesh.getInterpolationMat(pltgrid,'Fy')
    #     Vmesh = get_Layer_Potentials(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.],np.c_[mesh.gridCC,np.zeros(mesh.nC)])
    #     Emesh = -mesh.cellGrad*Vmesh
    #     Ex = (Px * Emesh).reshape(x.size,z.size,order='F')
    #     Ez = (Pz * Emesh).reshape(x.size,z.size,order='F')
    #     E = np.sqrt(Ex**2.+Ez**2.)
    #     cb = ax[1].pcolor(xplt,zplt,E,norm=LogNorm())
    #     ax[1].streamplot(x,z,Ex.T,Ez.T,color = 'k',linewidth= (np.log(E.T) - np.log(E).min())/np.max(np.log(E)))
    #     clabel = 'Electric Field (V/m)'
    #     clim = np.r_[3e-3,3e1]

    # elif imgplt is 'j':
    #     rho_model = rho2*np.ones(mesh.nC)
    #     rho_model[mesh.gridCC[:,1] >= -h] = rho1

    #     Px = mesh.getInterpolationMat(pltgrid,'CC')
    #     Pz = mesh.getInterpolationMat(pltgrid,'CC')

    #     Vmesh = get_Layer_Potentials(rho1,rho2,h,np.r_[A,0.,-0.5],np.r_[B,0.,-0.5],np.c_[mesh.gridCC,np.zeros(mesh.nC)])
    #     G = mesh.cellGrad
    #     # G[-1,-1] = 1*mesh.vol[-1]
    #     Emesh = -G*Vmesh
    #     # Jmesh = mesh.getFaceInnerProduct(1./Utils.mkvc(rho_model)) * Emesh
    #     Jmesh = Utils.sdiag(np.r_[1./rho_model,1./rho_model]) * mesh.aveF2CCV * Emesh

    #     Jx = (Px * Jmesh[:mesh.nC]).reshape(x.size,z.size,order='F')
    #     Jz = (Pz * Jmesh[mesh.nC:]).reshape(x.size,z.size,order='F')

    #     J = np.sqrt(Jx**2.+Jz**2.)

    #     cb = ax[1].pcolor(xplt,zplt,J,norm=LogNorm())
    #     ax[1].streamplot(x,z,Jx.T,Jz.T,color = 'k',linewidth = 2.5*(np.log(J.T)-np.log(J).min())/np.max(np.abs(np.log(J))))   
    #     ax[1].set_ylabel('z (m)', fontsize=14)

    #     clim = np.r_[3e-5,1e-1]
    #     clabel = 'Current Density (A/m$^2$)'

    # elif imgplt is 'jx':
    #     rho_model = rho2*np.ones(mesh.nC)
    #     rho_model[mesh.gridCC[:,1] >= -h] = rho1

    #     Px = mesh.getInterpolationMat(pltgrid,'Fx')
    #     Pz = mesh.getInterpolationMat(pltgrid,'Fy')
    #     Vmesh = get_Layer_Potentials(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.],np.c_[mesh.gridCC,np.zeros(mesh.nC)])


    #     G = mesh.cellGrad
    #     G[-1,-1] = 1*mesh.vol[-1]
    #     Emesh = -G*Vmesh
    #     Jmesh = mesh.getFaceInnerProduct(1./Utils.mkvc(rho_model)) * Emesh

    #     Jx = (Px * Jmesh).reshape(x.size,z.size,order='F')
    #     Jz = (Pz * Jmesh).reshape(x.size,z.size,order='F')

    #     J = np.sqrt(Jx**2.+Jz**2.)

    #     cb = ax[1].pcolor(xplt,zplt,Jx)
    #     ax[1].streamplot(x,z,Jx.T,Jz.T,color = 'k',linewidth = 2.5*(np.log(J.T)-np.log(J).min())/np.max(np.abs(np.log(J))))   
    #     ax[1].set_ylabel('z (m)', fontsize=14)

    #     clim = np.r_[3e-5,1e-1]
    #     clabel = 'Current Density (A/m$^2$)'

    # elif imgplt is 'jz':
    #     rho_model = rho2*np.ones(mesh.nC)
    #     rho_model[mesh.gridCC[:,1] >= -h] = rho1

    #     Px = mesh.getInterpolationMat(pltgrid,'Fx')
    #     Pz = mesh.getInterpolationMat(pltgrid,'Fy')
    #     Vmesh = get_Layer_Potentials(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.],np.c_[mesh.gridCC,np.zeros(mesh.nC)])
    #     G = mesh.faceDiv.T
    #     G[-1,-1] = 1
    #     Emesh = -G*Vmesh
    #     Jmesh = mesh.getFaceInnerProduct(1./Utils.mkvc(rho_model)) * Emesh

    #     Jx = (Px * Jmesh).reshape(x.size,z.size,order='F')
    #     Jz = (Pz * Jmesh).reshape(x.size,z.size,order='F')

    #     J = np.sqrt(Jx**2.+Jz**2.)

    #     cb = ax[1].pcolor(xplt,zplt,Jz)
    #     # ax[1].streamplot(x,z,Jx.T,Jz.T,color = 'k',linewidth = 2.5*(np.log(J.T)-np.log(J).min())/np.max(np.abs(np.log(J))))   
    #     ax[1].set_ylabel('z (m)', fontsize=14)

    #     clim = np.r_[3e-5,1e-1]
    #     clabel = 'Current Density (A/m$^2$)'

    # elif imgplt is 'charges':
    #     rho_model = rho2*np.ones(mesh.nC)
    #     rho_model[mesh.gridCC[:,1] >= -h] = rho1

    #     Vmesh = get_Layer_Potentials(rho1,rho2,h,np.r_[A,0.,0.],np.r_[B,0.,0.],np.c_[mesh.gridCC,np.zeros(mesh.nC)])
    #     Emesh = -mesh.cellGrad*Vmesh

    #     P = mesh.getInterpolationMat(pltgrid,'CC')

    #     charges = P * (mesh.faceDivy * Emesh[mesh.nFx:])* epsilon_0
    #     # charges = ( P * (  Emesh) ) * epsilon_0
    #     charges = charges.reshape(x.size,z.size,order='F')


    #     cb = ax[1].pcolor(xplt,zplt,charges)  
    #     ax[1].set_ylabel('z (m)', fontsize=14)
    #     clabel = 'Charge Density (C/m$^3$)'




    ax[1].set_xlim([x.min(),x.max()])
    ax[1].set_ylim([z.min(),0.])
    ax[1].set_ylabel('z (m)', fontsize=14)
    cbar_ax = fig.add_axes([1., 0.08, 0.04, 0.4])
    plt.colorbar(cb,cax=cbar_ax,label=clabel)
    if 'clim' in locals():
        cb.set_clim(clim)
    ax[1].set_xlabel('x(m)',fontsize=14)

    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_Layer_Potentials_app():
    plot_Layer_Potentials_interact = lambda rho1,rho2,h,A,B,M,N,Plot: plot_Layer_Potentials(rho1,rho2,h,A,B,M,N,Plot)
    app = interact(plot_Layer_Potentials_interact,
                rho1 = FloatSlider(min=rhomin,max=rhomax,step=10., value = 500.),
                rho2 = FloatSlider(min=rhomin,max=rhomax,step=10., value = 500.),
                h = FloatSlider(min=0.,max=40.,step=1.,value=5.),
                A = FloatSlider(min=-40.,max=40.,step=1.,value=-30.),
                B = FloatSlider(min=-40.,max=40.,step=1.,value=30.),
                M = FloatSlider(min=-40.,max=40.,step=1.,value=-10.),
                N = FloatSlider(min=-40.,max=40.,step=1.,value=10.),
                Plot = ToggleButtons(options =['Model','Potential','E','J',],value='Model'),
                )
    return app

if __name__ == '__main__':
    rho1, rho2 = rhomin, rhomax
    h = 5.
    A,B = -30., 30. 
    M,N = -10., 10.
    Plot =  'e'
    plot_Layer_Potentials(rho1,rho2,h,A,B,M,N,Plot)
