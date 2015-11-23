import sys
sys.path.append("./simpeg")
sys.path.append("./simpegdc/")

import warnings
warnings.filterwarnings('ignore')

from SimPEG import Mesh, Maps
import numpy as np
import simpegDCIP as DC
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import warnings
warnings.filterwarnings('ignore') # ignore warnings: only use this once you are sure things are working

try:
    from IPython.html.widgets import  interact, IntSlider, FloatSlider, FloatText, ToggleButtons
    pass
except Exception, e:    
    from ipywidgets import interact, IntSlider, FloatSlider, FloatText, ToggleButtons


# Mesh, mapping can be globals global
npad = 8
cs = 1.
hx = [(cs,npad, -1.3),(cs,100),(cs,npad, 1.3)]
hy = [(cs,npad, -1.3),(cs,50)]
mesh = Mesh.TensorMesh([hx, hy], "CN")
circmap = Maps.CircleMap(mesh)
circmap.slope = 1e5
mapping = circmap
dx = 5 
xr = np.arange(-40,41,dx)
dxr = np.diff(xr)


def DC2Dsurvey(flag="PoleDipole"):

    if flag =="PoleDipole":
        ntx, nmax = xr.size-2, 8
    elif flag =="DipolePole":
        ntx, nmax = xr.size-2, 8
    elif flag =="DipoleDipole":
        ntx, nmax = xr.size-3, 8
    else: 
        raise Exception('Not Implemented')
    xzlocs = getPseudoLocs(xr, ntx, nmax, flag)

    txList = []
    zloc = -2.5
    for i in range(ntx):
        if flag == "PoleDipole":
            A = np.r_[xr[i], zloc]
            B = np.r_[mesh.vectorCCx.min(), zloc]   
            if i < ntx-nmax+1:
                M = np.c_[xr[i+1:i+1+nmax], np.ones(nmax)*zloc]        
                N = np.c_[xr[i+2:i+2+nmax], np.ones(nmax)*zloc]                
            else:
                M = np.c_[xr[i+1:ntx+1], np.ones(ntx-i)*zloc]        
                N = np.c_[xr[i+2:i+2+nmax], np.ones(ntx-i)*zloc]
        elif flag =="DipolePole":
            A = np.r_[xr[i], zloc]
            B = np.r_[xr[i+1], zloc]
            if i < ntx-nmax+1:
                M = np.c_[xr[i+2:i+2+nmax], np.ones(nmax)*zloc]        
                N = np.c_[np.ones(nmax)*mesh.vectorCCx.max(), np.ones(nmax)*zloc]                
            else:
                M = np.c_[xr[i+2:ntx+2], np.ones(ntx-i)*zloc]        
                N = np.c_[np.ones(ntx-i)*mesh.vectorCCx.max(), np.ones(ntx-i)*zloc] 
        elif flag =="DipoleDipole":
            A = np.r_[xr[i], zloc]
            B = np.r_[xr[i+1], zloc]    
            if i < ntx-nmax:
                M = np.c_[xr[i+2:i+2+nmax], np.ones(len(xr[i+2:i+2+nmax]))*zloc]       
                N = np.c_[xr[i+3:i+3+nmax], np.ones(len(xr[i+3:i+3+nmax]))*zloc]                
            else:
                M = np.c_[xr[i+2:len(xr)-1], np.ones(len(xr[i+2:len(xr)-1]))*zloc]      
                N = np.c_[xr[i+3:len(xr)], np.ones(len(xr[i+3:len(xr)]))*zloc]
                                          
        rx = DC.RxDipole(M, N)
        src = DC.SrcDipole([rx], A, B)
        txList.append(src)

    survey = DC.SurveyDC(txList)
    problem = DC.ProblemDC_CC(mesh, mapping = mapping)
    problem.pair(survey)    

    sigblk, sighalf = 2e-2, 2e-3
    xc, yc, r = -15, -8, 4
    mtrue = np.r_[np.log(sigblk), np.log(sighalf), xc, yc, r]
    dtrue = survey.dpred(mtrue) 
    perc = 0.1
    floor = np.linalg.norm(dtrue)*1e-3
    np.random.seed([1])
    uncert = np.random.randn(survey.nD)*perc + floor
    dobs = dtrue + uncert 

    return dobs, uncert, survey, xzlocs

def getPseudoLocs(xr, ntx, nmax, flag = "PoleDipole"):
    xloc = []
    yloc = []    
    for i in range(ntx):
        if i < ntx-nmax+1:

            if flag is 'DipoleDipole':
                txmid = xr[i]+dxr[i]*0.5
                rxmid = xr[i+1:i+1+nmax]+dxr[i+1:i+1+nmax]*0.5

            elif flag is 'PoleDipole':
                txmid = xr[i]
                rxmid = xr[i+1:i+1+nmax]+dxr[i+1:i+1+nmax]*0.5

            elif flag is 'DipolePole':
                txmid = xr[i]+dxr[i]*0.5
                rxmid = xr[i+1:i+1+nmax]

            mid = (txmid+rxmid)*0.5
            xloc.append(mid)
            yloc.append(np.arange(nmax)+1.)
        else:
            if flag is 'DipoleDipole':
                txmid = xr[i]+dxr[i]*0.5
                rxmid = xr[i+1:ntx+1]+dxr[i+1:ntx+1]*0.5

            elif flag is 'PoleDipole':
                txmid = xr[i]
                rxmid = xr[i+1:ntx+1]+dxr[i+1:ntx+1]*0.5

            elif flag is 'DipolePole':
                txmid = xr[i]+dxr[i]*0.5
                rxmid = xr[i+1:ntx+1]

            mid = (txmid+rxmid)*0.5    
            xloc.append(mid)
            yloc.append(np.arange(mid.size)+1.)
    xlocvec = np.hstack(xloc)
    ylocvec = np.hstack(yloc)
    return np.c_[xlocvec, ylocvec]

def PseudoSectionPlotfnc(i,j,survey,flag="PoleDipole"):
    matplotlib.rcParams['font.size'] = 14
    nmax = 8
    dx = 5
    xr = np.arange(-40,41,dx)
    ntx = xr.size-2
    dxr = np.diff(xr)
    TxObj = survey.srcList
    TxLoc = TxObj[i].loc
    RxLoc = TxObj[i].rxList[0].locs
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(xr.min()-5, xr.max()+5), ylim=(nmax+1, -2))
    plt.plot(xr, np.zeros_like(xr), 'ko', markersize=4)
    if flag == "PoleDipole":
        plt.plot(TxLoc[0][0], np.zeros(1), 'rv', markersize=10)
        # print([TxLoc[0][0],0])
        ax.annotate('A', xy=(TxLoc[0][0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
    else:
        plt.plot([TxLoc[0][0],TxLoc[1][0]], np.zeros(2), 'rv', markersize=10)
        # print([[TxLoc[0][0],0],[TxLoc[1][0],0]])
        ax.annotate('A', xy=(TxLoc[0][0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
        ax.annotate('B', xy=(TxLoc[1][0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')        
    # for i in range(ntx):
    if i < ntx-nmax+1:
        if flag == "PoleDipole":
            txmid = TxLoc[0][0]
        else:
            txmid = (TxLoc[0][0] + TxLoc[1][0])*0.5

        MLoc = RxLoc[0][j]
        NLoc = RxLoc[1][j]
        # plt.plot([MLoc[0],NLoc[0]], np.zeros(2), 'b^', markersize=10)
        # ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
        # ax.annotate('N', xy=(NLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')        
        if flag == "DipolePole":
            plt.plot(MLoc[0], np.zeros(1), 'bv', markersize=10)
            ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
            rxmid = MLoc[0]
        else:
            rxmid = (MLoc[0]+NLoc[0])*0.5
            plt.plot(MLoc[0], np.zeros(1), 'bv', markersize=10)
            plt.plot(NLoc[0], np.zeros(1), 'b^', markersize=10)
            ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
            ax.annotate('N', xy=(NLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
        mid = (txmid+rxmid)*0.5
        midSep = np.sqrt(np.square(txmid-rxmid))
        plt.plot(txmid, np.zeros(1), 'ro')
        plt.plot(rxmid, np.zeros(1), 'bo')
        plt.plot(mid, midSep/2., 'go')
        plt.plot(np.r_[txmid, mid], np.r_[0, midSep/2.], 'k:')    
        # for j in range(nmax):
            # plt.plot(np.r_[rxmid[j], mid[j]], np.r_[0, j+1], 'k:')
        plt.plot(np.r_[rxmid, mid], np.r_[0, midSep/2.], 'k:')       

    else:
        if flag == "PoleDipole":
            txmid = TxLoc[0][0]
        else:
            txmid = (TxLoc[0][0] + TxLoc[1][0])*0.5


        MLoc = RxLoc[0][j]
        NLoc = RxLoc[1][j]
        if flag == "DipolePole":
            plt.plot(MLoc[0], np.zeros(1), 'bv', markersize=10)
            ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
            rxmid = MLoc[0]
        else:
            rxmid = (MLoc[0]+NLoc[0])*0.5
            plt.plot(MLoc[0], np.zeros(1), 'bv', markersize=10)
            plt.plot(NLoc[0], np.zeros(1), 'b^', markersize=10)
            ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
            ax.annotate('N', xy=(NLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
        # plt.plot([MLoc[0],NLoc[0]], np.zeros(2), 'b^', markersize=10)
        # ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
        # ax.annotate('N', xy=(NLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')        
    
        # rxmid = xr[i+1:ntx+1]+dxr[i+1:ntx+1]*0.5
        mid = (txmid+rxmid)*0.5    
        plt.plot((txmid+rxmid)*0.5, np.arange(mid.size)+1., 'bo')
        plt.plot(rxmid, np.zeros(rxmid.size), 'go')
        plt.plot(np.r_[txmid, mid[-1]], np.r_[0, mid.size], 'k:')    
        for j in range(ntx-i):
            plt.plot(np.r_[rxmid[j], mid[j]], np.r_[0, j+1], 'k:')    
    plt.xlabel("X (m)")
    plt.ylabel("N-spacing")    
    plt.xlim(xr.min()-5, xr.max()+5)
    plt.ylim(nmax*dx/2+dx, -2*dx)
    plt.show()
    return 

def DipoleDipolefun(i):
    matplotlib.rcParams['font.size'] = 14
    plt.figure(figsize=(10, 3))
    nmax = 8
    xr = np.linspace(-40, 40, 20)
    ntx = xr.size-2
    dxr = np.diff(xr)
    plt.plot(xr[:-1]+dxr*0.5, np.zeros_like(xr[:-1]), 'ko')
    plt.plot(xr[i]+dxr[i]*0.5, np.zeros(1), 'ro')
    # for i in range(ntx):
    if i < ntx-nmax+1:
        txmid = xr[i]+dxr[i]*0.5
        rxmid = xr[i+1:i+1+nmax]+dxr[i+1:i+1+nmax]*0.5
        mid = (txmid+rxmid)*0.5
        plt.plot(rxmid, np.zeros(rxmid.size), 'go')
        plt.plot(mid, np.arange(nmax)+1., 'bo')
        plt.plot(np.r_[txmid, mid[-1]], np.r_[0, nmax], 'k:')    
        for j in range(nmax):
            plt.plot(np.r_[rxmid[j], mid[j]], np.r_[0, j+1], 'k:')    

    else:
        txmid = xr[i]+dxr[i]*0.5
        rxmid = xr[i+1:ntx+1]+dxr[i+1:ntx+1]*0.5
        mid = (txmid+rxmid)*0.5    
        plt.plot((txmid+rxmid)*0.5, np.arange(mid.size)+1., 'bo')
        plt.plot(rxmid, np.zeros(rxmid.size), 'go')
        plt.plot(np.r_[txmid, mid[-1]], np.r_[0, mid.size], 'k:')    
        for j in range(ntx-i):
            plt.plot(np.r_[rxmid[j], mid[j]], np.r_[0, j+1], 'k:')    
    plt.xlabel("X (m)")
    plt.ylabel("N-spacing")    
    plt.xlim(xr.min(), xr.max())
    plt.ylim(nmax+1, -1)
    plt.show()
    return 

def PseudoSectionWidget(survey,flag):
    dx = 5 
    xr = np.arange(-40,41,dx)
    if flag =="PoleDipole":
        ntx, nmax = xr.size-2, 8
        dxr = np.diff(xr)
    elif flag =="DipolePole":
        ntx, nmax = xr.size-1, 7
        dxr = xr
    elif flag =="DipoleDipole":
        ntx, nmax = xr.size-3, 8
        dxr = np.diff(xr)
    xzlocs = getPseudoLocs(dxr, ntx, nmax,flag)
    PseudoSectionPlot = lambda i,j: PseudoSectionPlotfnc(i,j,survey,flag)
    return interact(PseudoSectionPlot, i=IntSlider(min=0, max=ntx-1, step = 1, value=0),j=IntSlider(min=0, max=nmax-1, step = 1, value=0))

def MidpointPseudoSectionWidget():
    ntx = 18
    return interact(DipoleDipolefun, i=IntSlider(min=0, max=ntx-1, step = 1, value=0))

def DC2Dfwdfun(mesh, survey, mapping, xr, xzlocs, rhohalf, rhoblk, xc, yc, r, dobs, uncert, predmis, nmax=8, plotFlag=None):
    matplotlib.rcParams['font.size'] = 14
    sighalf, sigblk = 1./rhohalf, 1./rhoblk
    m0 = np.r_[np.log(sighalf), np.log(sighalf), xc, yc, r]
    dini = survey.dpred(m0)
    mtrue = np.r_[np.log(sigblk), np.log(sighalf), xc, yc, r]
    dpred  = survey.dpred(mtrue)
    xi, yi = np.meshgrid(np.linspace(xr.min(), xr.max(), 120), np.linspace(1., nmax, 100))
    appres = dpred/dini/sighalf
    appresobs = dobs/dini/sighalf
    pred = pylab.griddata(xzlocs[:,0], xzlocs[:,1], appres, xi, yi, interp='linear')
    
    if plotFlag is not None:
        fig = plt.figure(figsize = (12, 6))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)

        dat1 = mesh.plotImage(np.log10(1./(mapping*mtrue)), ax=ax1, clim=(0, 3), grid=True, gridOpts={'color':'k', 'alpha':0.5})
        cb1 = plt.colorbar(dat1[0], ticks=np.linspace(0, 3, 5), ax=ax1, format="$10^{%4.1f}$")
        cb1.set_label("Resistivity (ohm-m)")
        ax1.set_ylim(-30, 0.)
        ax1.set_xlim(-40, 40)
        ax1.set_xlabel("")
        ax1.set_ylabel("Depth (m)")    

        dat2 = ax2.contourf(xi, yi, pred, 10)
        ax2.contour(xi, yi, pred, 10, colors='k', alpha=0.5)
        ax2.plot(xzlocs[:,0], xzlocs[:,1],'k.', ms = 3)
        cb2 = plt.colorbar(dat2, ax=ax2, ticks=np.linspace(appres.min(), appres.max(), 5),format="%4.0f")
        cb2.set_label("Apparent Resistivity \n (ohm-m)")
        ax2.text(-38, 7, "Predicted")

        ax2.set_ylim(nmax+1, 0.)
        ax2.set_ylabel("N-spacing")    
        ax2.set_xlabel("Distance (m)")    

    else:
        obs = pylab.griddata(xzlocs[:,0], xzlocs[:,1], appresobs, xi, yi, interp='linear')
        fig = plt.figure(figsize = (12, 8))
        ax1 = plt.subplot(311)
        dat1 = mesh.plotImage(np.log10(1./(mapping*mtrue)), ax=ax1, clim=(0, 3), grid=True, gridOpts={'color':'k', 'alpha':0.5})
        cb1 = plt.colorbar(dat1[0], ticks=np.linspace(0, 3, 5), ax=ax1, format="$10^{%4.1f}$")
        cb1.set_label("Resistivity (ohm-m)")
        ax1.set_ylim(-30, 0.)
        ax1.set_xlim(-40, 40)
        ax1.set_xlabel("")
        ax1.set_ylabel("Depth (m)")    
        ax2 = plt.subplot(312)
        dat2 = ax2.contourf(xi, yi, obs, 10)
        ax2.contour(xi, yi, obs, 10, colors='k', alpha=0.5)
        ax2.plot(xzlocs[:,0], xzlocs[:,1],'k.', ms = 3)
        cb2 = plt.colorbar(dat2, ax=ax2, ticks=np.linspace(appresobs.min(), appresobs.max(), 5), format="%4.1f")
       
        cb2.set_label("Apparent Resistivity \n (ohm-m)")
        ax2.set_ylim(nmax+1, 0.)
        ax2.set_ylabel("N-spacing")    
        ax2.text(-38, 7, "Observed")
        
        ax3 = plt.subplot(313)
        if predmis=="pred":
            dat3 = ax3.contourf(xi, yi, pred, 10)
            ax3.contour(xi, yi, pred, 10, colors='k', alpha=0.5)
            ax3.plot(xzlocs[:,0], xzlocs[:,1],'k.', ms = 3)
            cb3 = plt.colorbar(dat3, ax=ax3, ticks=np.linspace(appres.min(), appres.max(), 5),format="%4.0f")
            cb3.set_label("Apparent Resistivity \n (ohm-m)")
            ax3.text(-38, 7, "Predicted")
        elif predmis=="mis":
            mis = (appresobs-appres)/(0.1*appresobs)
            Mis = pylab.griddata(xzlocs[:,0], xzlocs[:,1], mis, xi, yi, interp='linear')        
            dat3 = ax3.contourf(xi, yi, Mis, 10)
            ax3.contour(xi, yi, Mis, 10, colors='k', alpha=0.5)
            ax3.plot(xzlocs[:,0], xzlocs[:,1],'k.', ms = 3)
            cb3 = plt.colorbar(dat3, ax=ax3, ticks=np.linspace(mis.min(), mis.max(), 5), format="%4.2f")
            cb3.set_label("Normalized misfit")
            ax3.text(-38, 7, "Misifit")    
        ax3.set_ylim(nmax+1, 0.)
        ax3.set_ylabel("N-spacing")    
        ax3.set_xlabel("Distance (m)")    

    plt.show()
    return 

def DC2DPseudoWidgetWrapper(rhohalf,rhosph,xc,zc,r,surveyType):
    dobs, uncert, survey, xzlocs = DC2Dsurvey(surveyType)
    DC2Dfwdfun(mesh, survey, mapping, xr, xzlocs, rhohalf, rhosph, xc, zc, r, dobs, uncert, 'pred',plotFlag='PredOnly')
    return None

def DC2DPseudoWidget():
    # print xzlocs
    Q = interact(DC2DPseudoWidgetWrapper,
         rhohalf = FloatSlider(min=10, max=1000, step=1, value = 1000),
         rhosph = FloatSlider(min=10, max=1000, step=1, value = 50),
         xc = FloatSlider(min=-40, max=40, step=1, value =  0),
         zc = FloatSlider(min= -20, max=0, step=1, value =  -10),
         r = FloatSlider(min= 0, max=15, step=0.5, value = 5),
         surveyType = ToggleButtons(options=['DipoleDipole','PoleDipole','DipolePole'])         
        )
    return Q

def DC2DfwdWrapper(rhohalf,rhosph,xc,zc,r,predmis,surveyType):
    dobs, uncert, survey, xzlocs = DC2Dsurvey(surveyType)
    DC2Dfwdfun(mesh, survey, mapping, xr, xzlocs, rhohalf, rhosph, xc, zc, r, dobs, uncert, predmis)
    return None

def DC2DfwdWidget():
    # print xzlocs
    Q = interact(DC2DfwdWrapper,
         rhohalf = FloatSlider(min=10, max=1000, step=1, value = 1000),
         rhosph = FloatSlider(min=10, max=1000, step=1, value = 50),
         xc = FloatSlider(min=-40, max=40, step=1, value =  0),
         zc = FloatSlider(min= -20, max=0, step=1, value =  -10),
         r = FloatSlider(min= 0, max=15, step=0.5, value = 5),
         predmis = ToggleButtons(options=['pred','mis']),
         surveyType = ToggleButtons(options=['DipoleDipole','PoleDipole','DipolePole'])         
        )
    return Q
