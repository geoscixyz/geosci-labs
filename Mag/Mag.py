from fromFatiando import *
from fromSimPEG import *
from scipy.constants import mu_0
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import IPython.html.widgets as widgets
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



filename = "data/HZrebarProfile.csv"
data = pd.DataFrame(pd.read_csv(filename, header = 0))
loc = data["Distance"].values


def definePrism(dx, dy, dz, depth, susc = 1., x0=0.,y0=0., pinc=0., pdec=0., Einc=90., Edec=0., Bigrf=1e3, Q = 0., rinc = 0., rdec = 0.):
    """
        wrapper on fatiando prism construction
        
        Prism geometry:
            - dx, dy, dz: width, length and height of prism
            - depth : depth to top of prism
            - susc : susceptibility of prism
            - x0, y0 : center of prism in horizontal plane
            - pinc, pdec : inclination and declination of prism
        
        Earth's field:
            - Einc, Edec : inclination and declination of Earth's magnetic field
            - Bigrf : amplitude of earth's field in units of nT
        
        Remnance:
            - Q : Koenigsberger ratio
            - Rinc, Rdec : inclination and declination of remnance in block 
        
    """
    
    Higrf = Bigrf * 1e-9 / mu_0

    x1, x2 = -dx/2. + x0, dx/2. + x0
    y1, y2 = -dy/2. + y0, dy/2. + y0
    z1, z2 = depth, depth + dz
    Mind = susc*Higrf
    rMag = Q*Mind
    
    # This is a bit of a hack: I am putting all of the parameters we will need later in the 'property' dictionary 
    return fatiandoGridMesh.Prism(x1, x2, y1, y2, z1, z2,{'magnetization': fatiandoUtils.ang2vec(rMag, rinc-pinc, rdec-pdec),'pinc':pinc,'pdec':pdec,'rinc':rinc,'rdec':rdec,'depth':depth,'Einc':Einc,'Edec':Edec,'Mind':Mind})

def getField(p, XYZ, comp='tf',irt='induced'):
    
    pinc,pdec = p.props['pinc'], p.props['pdec']
    Einc, Edec = p.props['Einc'], p.props['Edec']
    rinc,rdec = p.props['rinc'], p.props['rdec']
    Mind = p.props['Mind']
    
    
    x1, x2 = p.x1, p.x2
    y1, y2 = p.y1, p.y2
    z1, z2 = p.z1, p.z2
    
    XYZ = simpegCoordUtils.rotatePointsFromNormals(XYZ, fatiandoUtils.ang2vec(1., pinc, pdec), np.r_[1.,0.,0.], np.r_[(x1+x2)/2., (y1+y2)/2., (z1+z2)/2.] )

    xp_eval, yp_eval, zp_eval = XYZ[:,0], XYZ[:,1], XYZ[:,2]

    if (irt is 'induced') or (irt is 'total'):
        if comp is 'bx': fieldi = fatiandoMagPrism.bx(xp_eval,yp_eval,zp_eval,[p],fatiandoUtils.ang2vec(Mind, Einc-pinc, Edec-pdec))
        if comp is 'by': fieldi = fatiandoMagPrism.by(xp_eval,yp_eval,zp_eval,[p],fatiandoUtils.ang2vec(Mind, Einc-pinc, Edec-pdec))
        if comp is 'bz': fieldi = fatiandoMagPrism.bz(xp_eval,yp_eval,zp_eval,[p],fatiandoUtils.ang2vec(Mind, Einc-pinc, Edec-pdec))
        if comp is 'tf': fieldi = fatiandoMagPrism.tf(xp_eval,yp_eval,zp_eval,[p],Einc-pinc,Edec-pdec,fatiandoUtils.ang2vec(Mind, Einc-pinc, Edec-pdec))
            
    if (irt is 'remanent') or (irt is 'total'):
        if comp is 'bx': fieldr = fatiandoMagPrism.bx(xp_eval,yp_eval,zp_eval,[p])
        elif comp is 'by': fieldr = fatiandoMagPrism.by(xp_eval,yp_eval,zp_eval,[p])
        elif comp is 'bz': fieldr = fatiandoMagPrism.bz(xp_eval,yp_eval,zp_eval,[p])
        elif comp is 'tf': fieldr = fatiandoMagPrism.tf(xp_eval,yp_eval,zp_eval,[p],Einc-pinc,Edec-pdec)
        
    if irt is 'induced':
        return fieldi
    elif irt is 'remanent':
        return fieldr
    elif irt is 'total':
        return fieldi, fieldr

def profiledataInd(x0, depth, susc):
    magnT = data["Anomaly"].values
    p = definePrism(3., 0.02, 0.03, depth, pinc=0., pdec=90., susc = susc, Einc=70.2, Edec=16.5, Bigrf=52000, x0=x0)

    nx, ny = 100, 1
    shape = (nx, ny)
    surveyArea = (loc.min()-8, loc.max()-8, 0., 0.)
    z = -1.9
    xpl, ypl, zpl = fatiandoGridMesh.regular(surveyArea,shape, z=z)
    xyz = np.vstack([xpl,ypl,zpl]).T

    fig, ax = plt.subplots(1,2, figsize = (12, 4))

    ax[0].plot(x0, depth, 'ko')
    ax[0].text(x0+0.5, depth, 'Rebar', color='k')
    ax[0].text(-1.,-2.0, 'Magnetometer height (1.9 m)', color='b')
    ax[0].plot(np.r_[-5, 5], np.r_[-1.9, -1.9], 'b--')

    ax[0].plot(np.r_[-5, 5], np.r_[0., 0.], 'k--')
    ax[0].set_xlim(-4, 4)
    ax[0].set_ylim(-3, 4)
    ax[1].plot(xpl, getField(p, xyz), 'k')
    ax[1].plot(loc-8, magnT[::-1], 'ko')
    ax[0].set_xlabel("Northing (m)")
    ax[0].set_ylabel("Depth (m)")
    ax[1].set_ylabel("Total field anomaly (nT)")

    for i in range(2):
        ax[i].grid(True)
        ax[i].set_xlabel("Northing (m)")
    ax[0].invert_yaxis()
    plt.show()
    return True

def fitlineInd():
    Q = widgets.interactive(profiledataInd, x0=widgets.FloatSliderWidget(min=-3, max=3, step=0.1, value=-3.), \
             depth=widgets.FloatSliderWidget(min=0,max=3,step=0.1,value=2.5), \
             susc=widgets.FloatSliderWidget(min=0., max=800.,step=5., value=1.))
    return Q

def profiledataRem(x0, depth, susc, Q, rinc, rdec):
    magnT = data["Anomaly"].values
    ## Seogi
    # Not sure why do I need to put -x0 ... 
    p = definePrism(3., 0.02, 0.03, depth, pinc=0., pdec=90., susc = susc, Einc=70.2, Edec=16.5, Bigrf=52000, x0=-x0, Q=Q, rinc = rinc, rdec = rdec)
    nx, ny = 100, 1
    shape = (nx, ny)
    surveyArea = (loc.min()-8, loc.max()-8, 0., 0.)
    z = -1.9
    xpl, ypl, zpl = fatiandoGridMesh.regular(surveyArea,shape, z=z)
    xyz = np.vstack([xpl,ypl,zpl]).T


    fig, ax = plt.subplots(1,2, figsize = (12, 4))
    ax[0].plot(x0, depth, 'ko')
    ax[0].text(x0+0.5, depth, 'Rebar', color='k')
    ax[0].text(-1.,-2.0, 'Magnetometer height (1.9 m)', color='b')
    ax[0].plot(np.r_[-5, 5], np.r_[-1.9, -1.9], 'b--')

    magi,magr = getField(p, xyz, 'tf', 'total')

    ax[0].plot(np.r_[-5, 5], np.r_[0., 0.], 'k--')
    ax[0].set_xlim(-4, 4)
    ax[0].set_ylim(-3, 4)
    ax[1].plot(xpl, magi+magr, 'k')
    ax[1].plot(xpl, magi, 'b')
    ax[1].plot(xpl, magr, 'r')
    ax[1].plot(loc-8, magnT[::-1], 'ko')
    ax[0].set_xlabel("Northing (m)")
    ax[0].set_ylabel("Depth (m)")
    ax[1].set_ylabel("Total field anomaly (nT)")

    for i in range(2):
        ax[i].grid(True)
        ax[i].set_xlabel("Northing (m)")
    ax[0].invert_yaxis()    
    plt.show()
    return True



def fitlineRem(x0, depth0, susc0):
    Q = widgets.interactive(profiledataRem, x0=widgets.FloatSliderWidget(min=-3, max=3, step=0.1, value=x0), \
             depth=widgets.FloatSliderWidget(min=0,max=3,step=0.1,value=depth0), \
             susc=widgets.FloatSliderWidget(min=0., max=800.,step=5., value=susc0),\
             Q=widgets.FloatSliderWidget(min=0., max=1.,step=0.01, value=0.),\
             rinc=widgets.FloatSliderWidget(min=-180., max=180.,step=1., value=0.),\
             rdec=widgets.FloatSliderWidget(min=-180., max=180.,step=1., value=0.))
    return Q

def plotObj3D(p, elev, azim, xmax = 10., ymax = 10., z=-1.9, nx=100, ny=100,
              profile=None, x0=0., y0=0.):

    # define the survey area
    surveyArea = (-xmax, xmax, -ymax, ymax)
    shape = (nx,ny)
    xp, yp, zp = fatiandoGridMesh.regular(surveyArea,shape, z=z)    

    depth = p.props['depth']
    x1, x2 = p.x1, p.x2
    y1, y2 = p.y1, p.y2
    z1, z2 = p.z1, p.z2
    pinc, pdec = p.props['pinc'], p.props['pdec']
    
    fig = plt.figure(figsize = (7, 7))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams.update({'font.size': 13})
    
    ax.set_xlim3d(surveyArea[:2])
    ax.set_ylim3d(surveyArea[2:])
#     ax.set_zlim3d(depth+np.array(surveyArea[:2]))
    ax.set_zlim3d(-3, surveyArea[-1]*2)

    xpatch = [x1,x1,x2,x2]
    ypatch = [y1,y2,y2,y1]
    zpatch = [z1,z1,z1,z1]
    xyz = simpegCoordUtils.rotatePointsFromNormals(np.vstack([xpatch,ypatch,zpatch]).T, np.r_[1., 0., 0.],fatiandoUtils.ang2vec(1.,pinc,pdec),np.r_[(x1+x2)/2., (y1+y2)/2., (z1+z2)/2.])
    ax.add_collection3d(Poly3DCollection([zip(xyz[:,0], xyz[:,1], xyz[:,2])]))
    zpatch = [z2,z2,z2,z2]
    xyz = simpegCoordUtils.rotatePointsFromNormals(np.vstack([xpatch,ypatch,zpatch]).T, np.r_[1., 0., 0.],fatiandoUtils.ang2vec(1.,pinc,pdec),np.r_[(x1+x2)/2., (y1+y2)/2., (z1+z2)/2.])
    ax.add_collection3d(Poly3DCollection([zip(xyz[:,0], xyz[:,1], xyz[:,2])]))
    
    xpatch = [x1,x1,x1,x1]
    ypatch = [y1,y2,y2,y1]
    zpatch = [z1,z1,z2,z2]                                  
    xyz = simpegCoordUtils.rotatePointsFromNormals(np.vstack([xpatch,ypatch,zpatch]).T, np.r_[1., 0., 0.],fatiandoUtils.ang2vec(1.,pinc,pdec),np.r_[(x1+x2)/2., (y1+y2)/2., (z1+z2)/2.])
    ax.add_collection3d(Poly3DCollection([zip(xyz[:,0], xyz[:,1], xyz[:,2])]))    
    xpatch = [x2,x2,x2,x2]                                 
    xyz = simpegCoordUtils.rotatePointsFromNormals(np.vstack([xpatch,ypatch,zpatch]).T, np.r_[1., 0., 0.],fatiandoUtils.ang2vec(1.,pinc,pdec),np.r_[(x1+x2)/2., (y1+y2)/2., (z1+z2)/2.])
    ax.add_collection3d(Poly3DCollection([zip(xyz[:,0], xyz[:,1], xyz[:,2])])) 
    
    xpatch = [x1,x2,x2,x1]
    ypatch = [y1,y1,y1,y1]
    zpatch = [z1,z1,z2,z2]                                  
    xyz = simpegCoordUtils.rotatePointsFromNormals(np.vstack([xpatch,ypatch,zpatch]).T, np.r_[1., 0., 0.],fatiandoUtils.ang2vec(1.,pinc,pdec),np.r_[(x1+x2)/2., (y1+y2)/2., (z1+z2)/2.])
    ax.add_collection3d(Poly3DCollection([zip(xyz[:,0], xyz[:,1], xyz[:,2])]))   
    ypatch = [y2,y2,y2,y2]                                 
    xyz = simpegCoordUtils.rotatePointsFromNormals(np.vstack([xpatch,ypatch,zpatch]).T, np.r_[1., 0., 0.],fatiandoUtils.ang2vec(1.,pinc,pdec),np.r_[(x1+x2)/2., (y1+y2)/2., (z1+z2)/2.])
    ax.add_collection3d(Poly3DCollection([zip(xyz[:,0], xyz[:,1], xyz[:,2])])) 
    
    ax.set_xlabel('Northing (X; m)')
    ax.set_ylabel('Easting (Y; m)')
    ax.set_zlabel('Depth (Z; m)')
    ax.invert_zaxis()
    ax.invert_yaxis()

    ax.plot(xp,yp,z,'.g', alpha=0.1)
    if profile == "X":    
        ax.plot(np.r_[surveyArea[:2]],np.r_[0., 0.],np.r_[z, z],'r-')
    elif profile == "Y":    
        ax.plot(np.r_[0., 0.], np.r_[surveyArea[2:]],np.r_[z, z],'r-')
    elif profile == "XY":
        ax.plot(np.r_[surveyArea[:2]],np.r_[0., 0.],np.r_[z, z],'r-')
        ax.plot(np.r_[0., 0.], np.r_[surveyArea[2:]],np.r_[z, z],'r-')

    ax.view_init(elev,azim) 
    plt.show()
    return True


def Prism(dx, dy, dz, depth, pinc, pdec, elev, azim):
    p = definePrism(dx, dy, dz, depth,pinc=pinc, pdec=pdec, susc = 1., Einc=90., Edec=0., Bigrf=1e-6)
    return plotObj3D(p, elev, azim, profile="X")

def ViewPrism(dx, dy, dz, depth):
    Q = widgets.interactive(Prism,dx=widgets.FloatText(value=dx),dy=widgets.FloatText(value=dy), dz=widgets.FloatText(value=dz)\
                    ,depth=widgets.IntSlider(min=0,max=10,step=1,value=depth)
                    ,pinc=(-90, 90, 10), pdec=(-90, 90., 10) \
                    ,elev=widgets.FloatText(value=30), azim=widgets.FloatText(value=200))
    return Q

def PrismSurvey(dx, dy, dz, depth, pinc, pdec):
    elev, azim = 30, 200
    p = definePrism(dx, dy, dz, depth,pinc=pinc, pdec=pdec, susc = 1., Einc=90., Edec=0., Bigrf=1e-6)
    return p, plotObj3D(p, elev, azim, profile=None, z=0., xmax=20, ymax=20)

def ViewPrismSurvey(dx, dy, dz, depth):    
    Q = widgets.interactive(PrismSurvey,dx=widgets.FloatText(value=dx),dy=widgets.FloatText(value=dy), dz=widgets.FloatText(value=dz)\
                    ,depth=widgets.FloatSlider(min=0,max=10,step=0.3,value=depth)
                    ,pinc=(-90, 90, 10), pdec=(-90, 90., 10))
    return Q    


def linefun(x1, x2, y1, y2, nx):
    x = np.linspace(x1, x2, nx)
    if x1==x2:
        y = np.ones_like(x)*y1
    else:
        slope = (y2-y1)/(x2-x1)
        y=slope*(x-x1)+y1
    return x, y

def plogMagSurvey2D(h, depth, susc, Einc, Edec, Bigrf, x1, y1, x2, y2, npts2D, npts, z, comp, irt,  Q, rinc, rdec):
    nx, ny = npts2D, npts2D
    surveyArea = (-20., 20., -20., 20.)
    shape = (nx,ny)
    xp, yp, zp = fatiandoGridMesh.regular(surveyArea,shape, z=z)
    xyz = np.vstack([xp,yp,zp]).T
    X = xp.reshape((nx, ny))
    Y = yp.reshape((nx, ny))
    p = definePrism(h.kwargs['dx'], h.kwargs['dy'], h.kwargs['dz'], depth, pinc=h.kwargs['pinc'], pdec=h.kwargs['pdec'], susc = susc, Einc=Einc, Edec=Edec, Bigrf=Bigrf,
                    Q = Q, rinc = rinc, rdec = rdec)
    import matplotlib.gridspec as gridspec
    x, y = linefun(x1, x2, y1, y2, npts)
    xyz_line = np.c_[x, y, np.ones_like(x)*z]
    fig = plt.figure(figsize=(18*1.5,3.4*1.5))
    plt.rcParams.update({'font.size': 14})
    gs1 = gridspec.GridSpec(2, 7)
    gs1.update(left=0.05, right=0.48, wspace=0.05)
    ax1 = plt.subplot(gs1[:2, :3])
    ax2 = plt.subplot(gs1[0, 4:])
    ax1.axis("equal")
    if irt == 'total':
        out = getField(p, xyz, comp, 'induced')+getField(p, xyz, comp, 'remanent')        
    else:
        out = getField(p, xyz, comp, irt)
    dat = ax1.contourf(Y,X, out.reshape((shape)), 100)
    cb = plt.colorbar(dat, ax=ax1, ticks=np.linspace(out.min(), out.max(), 5))
    cb.set_label("nT")
    ax1.plot(x, y, 'w.', ms=3)
    ax1.text(x[0], y[0], 'A', fontsize = 16, color='w')
    ax1.text(x[-1], y[-1], 'B', fontsize = 16, color='w')
    ax1.set_xlabel('Easting (Y; m)')
    ax1.set_ylabel('Northing (X; m)')
    ax1.set_xlim(X.min(), X.max())
    ax1.set_ylim(Y.min(), Y.max())
    ax1.set_title(irt+' '+comp)
    out_linei = getField(p, xyz_line, comp,'induced')
    out_liner = getField(p, xyz_line, comp,'remanent')
    out_linet = out_linei+out_liner
    distance = np.sqrt((x-x1)**2+(y-y1)**2)
    ax2.plot(distance,out_linei, 'b.-')
    ax2.plot(distance,out_liner, 'r.-')
    ax2.plot(distance,out_linet, 'k.-')
    ax2.set_xlim(distance.min(), distance.max())

    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Magnetic field (nT)")

    ax2.text(distance.min(), out_linei.max()*0.8, 'A', fontsize = 16)
    ax2.text(distance.max()*0.97, out_linei.max()*0.8, 'B', fontsize = 16)
    ax2.legend(("induced", "remanent", "total"), bbox_to_anchor=(0.5, -0.3))
    ax2.grid(True)
    plt.show()
    return True
    

def ViewMagSurvey2DInd(h):
    
    def MagSurvey2DInd(depth, susc, Einc, Edec, Bigrf, x1, y1, x2, y2, npts2D, npts, z, comp, irt, Q, rinc, rdec):
        return plogMagSurvey2D(h, depth, susc, Einc, Edec, Bigrf, x1, y1, x2, y2, npts2D, npts, z, comp, irt, Q, rinc, rdec)    
    
    out = widgets.interactive (MagSurvey2DInd 
                    ,depth=widgets.FloatSlider(min=0,max=10,step=1,value=h.kwargs['depth']) \
                    # ,susc=widgets.FloatSlider(min=0,max=200,step=5,value=0) \
                    ,susc=widgets.FloatText(value=1.) \
                    ,Einc=widgets.FloatText(value=70.), Edec=widgets.FloatText(value=16.) \
                    ,Bigrf=widgets.FloatText(value=52000.) \
                    ,x1=widgets.FloatText(value=-10) \
                    ,y1=widgets.FloatText(value=-10) \
                    ,x2=widgets.FloatText(value=10) \
                    ,y2=widgets.FloatText(value=10) \
                    ,npts2D=widgets.IntSlider(min=5,max=200,step=1,value=20) \
                    ,npts=widgets.IntSlider(min=5,max=200,step=1,value=20) \
                    ,z=widgets.FloatText(value=-1.9) \
                    ,comp=widgets.ToggleButtons(options=['tf','bx','by','bz'])
                    ,irt=widgets.ToggleButtons(options=['induced','remanent', 'total']) 
                    ,Q=widgets.FloatText(value=0.)
                    ,rinc=widgets.FloatText(value=0.), rdec=widgets.FloatText(value=0.) \
                    )
    return out
# fig.tight_layout()
