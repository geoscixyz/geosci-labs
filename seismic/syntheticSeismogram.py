import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def getPlotLog(d,log,dmax=300):
    d = np.array(d, dtype=float)
    log = np.array(log, dtype=float)

    dplot   = np.kron(d,np.ones(2))
    logplot = np.kron(log,np.ones(2))

    # dplot   = dplot[1:]
    dplot   = np.append(dplot[1:],dmax)

    return dplot, logplot


def getImpedance(rholog,vlog):
    """
    docstring for getImpedance
    """
    rholog, vlog = np.array(rholog, dtype=float), np.array(vlog, dtype=float),
    return rholog*vlog


def getReflectivity(d,rho,v,usingT=True):
    Z   = getImpedance(rho,v)         # acoustic impedance
    R   = np.diff(Z)/(Z[:-1] + Z[1:]) # reflection coefficients

    nlayer = len(v) # number of layers

    rseries = R

    if usingT:
        for i in range(nlayer-1):
            rseries[i+1:] = rseries[i+1:]*(1.-R[i]**2)

    return rseries, R


def getTimeDepth(d,v,dmax=300):
    """
    docstring for getTimeDepth
    """

    d = np.sort(d)
    d = np.append(d,dmax)

    twttop  = 2.*np.diff(d)/v    # 2-way travel time within each layer
    twttop  = np.append(0.,twttop)
    twttop  = np.cumsum(twttop)       # 2-way travel time from surface to top of each layer

    return d, twttop


def getLogs(d, rho, v, usingT=True):
    """
    docstring for getLogs
    """
    dpth, rholog  = getPlotLog(d,rho)
    _   , vlog    = getPlotLog(d,v)
    zlog          = getImpedance(rholog,vlog)
    rseries, _    = getReflectivity(d,rho,v,usingT)
    return dpth, rholog, vlog, zlog, rseries


def syntheticSeismogram(d, rho, v, wavtyp, wavf, usingT=True, dt=0.0004, dmax=300):
    """
    function syntheticSeismogram(d, rho, v, wavtyp, wavf, usingT)

    syntheicSeismogram generates and displays a synthetic seismogram for
    a simple 1-D layered model.

    Inputs:
        v      : velocity of each layer (m/s)
        rho    : density of each layer (kg/m^3)
        d      : depth to the top of each layer (m)
                    The last layer is assumed to be a half-space
        wavtyp : type of Wavelet
                    The wavelet options are:
                        Ricker: takes one frequency
                        Gaussian: still in progress
                        Ormsby: takes 4 frequencies
                        Klauder: takes 2 frequencies
        usingT : use transmission coefficients?

    Lindsey Heagy
    lheagy@eos.ubc.ca
    Created:  November 30, 2013
    Modified: October 3, 2014
    """

    v, rho, d = np.array(v, dtype=float),   np.array(rho, dtype=float), np.array(d, dtype=float)
    usingT    = np.array(usingT, dtype=bool)

    _, t = getTimeDepth(d,v,dmax)
    rseries,R = getReflectivity(d,rho,v)

    # time for reflectivity series
    tref   = t[1:-1]

    # create time vector
    t = np.arange(t.min(),t.max(),dt)

    # make wavelet
    twav   = np.arange(-2.0/np.min(wavf), 2.0/np.min(wavf), dt)

    # Get source wavelet
    wav = {'RICKER':getRicker, 'ORMSBY':getOrmsby, 'KLAUDER':getKlauder}[wavtyp](wavf,twav)

    rseriesconv = np.zeros(len(t))
    for i in range(len(tref)):
         index = np.abs(t - tref[i]).argmin()
         rseriesconv[index] = rseries[i]

    # Do the convolution
    seis  = np.convolve(wav,rseriesconv)
    tseis = np.min(twav)+dt*np.arange(len(seis))
    index = np.logical_and(tseis >= 0, tseis <= np.max(t))
    tseis = tseis[index]
    seis  = seis[index]

    return tseis, seis, twav, wav, tref, rseries



## WAVELET DEFINITIONS
pi = np.pi
def getRicker(f,t):
    assert len(f) == 1, 'Ricker wavelet needs 1 frequency as input'
    f = f[0]
    pift = pi*f*t
    wav = (1 - 2*pift**2)*np.exp(-pift**2)
    return wav

# def getGauss(f,t):
#     assert len(f) == 1, 'Gauss wavelet needs 1 frequency as input'
#     f = f[0]

def getOrmsby(f,t):
    assert len(f) == 4, 'Ormsby wavelet needs 4 frequencies as input'
    f = np.sort(f) #Ormsby wavelet frequencies must be in increasing order
    pif   = pi*f
    den1  = pif[3] - pif[2]
    den2  = pif[1] - pif[0]
    term1 = (pif[3]*np.sinc(pif[3]*t))**2 - (pif[2]*np.sinc(pif[2]))**2
    term2 = (pif[1]*np.sinc(pif[1]*t))**2 - (pif[0]*np.sinc(pif[0]))**2

    wav   = term1/den1 - term2/den2;
    return wav

def getKlauder(f,t,T=5.0):
    assert len(f) == 2, 'Klauder wavelet needs 2 frequencies as input'

    k  = np.diff(f)/T
    f0 = np.sum(f)/2.0
    wav = np.real(np.sin(pi*k*t*(T-t))/(pi*k*t)*np.exp(2*pi*1j*f0*t))
    return wav



## Plotting Functions

def plotLogFormat(log, dpth,xlim, col='blue'):
    """
    docstring for plotLogFormat
    """
    ax = plt.plot(log,dpth,linewidth=2,color=col)
    plt.xlim(xlim)
    plt.ylim((dpth.min(),dpth.max()))
    plt.grid()
    plt.gca().invert_yaxis()
    plt.setp(plt.xticks()[1],rotation='90',fontsize=9)
    plt.setp(plt.yticks()[1],fontsize=9)

    return ax


def plotLogs(d, rho, v, usingT=True):
    """
    function plotLogs(d,rho,v,usingT)
    """
    d = np.sort(d)

    dpth, rholog, vlog, zlog, rseries  = getLogs(d, rho, v, usingT)
    nd   = len(dpth)

    plt.figure()
    xlimrho = (1.99,5.01)
    xlimv   = (0.29,4.01)
    xlimz   = (xlimrho[0]*xlimv[0], xlimrho[1]*xlimv[1])

    # Plot Density
    plt.subplot(141)
    plotLogFormat(rholog*10**-3,dpth,xlimrho,'blue')
    plt.title('$\\rho$')
    plt.xlabel('Density \n $\\times 10^3$ (kg /m$^3$)',fontsize=9)
    plt.ylabel('Depth (m)',fontsize=9)

    plt.subplot(142)
    plotLogFormat(vlog*10**-3,dpth,xlimv,'red')
    plt.title('$v$')
    plt.xlabel('Velocity \n $\\times 10^3$ (m/s)',fontsize=9)
    plt.setp(plt.yticks()[1],visible=False)

    plt.subplot(143)
    plotLogFormat(zlog*10.**-6.,dpth,xlimz,'green')
    plt.gca().set_title('$Z = \\rho v$')
    plt.gca().set_xlabel('Impedance \n $\\times 10^{6}$ (kg m$^{-2}$ s$^{-1}$)',fontsize=9)
    plt.setp(plt.yticks()[1],visible=False)

    plt.subplot(144)
    plt.hlines(d[1:],np.zeros(nd-1),rseries,linewidth=2)
    plt.plot(np.zeros(nd),dpth,linewidth=2,color='black')
    plt.title('Reflectivity');
    plt.xlim((-1.,1.))
    plt.gca().set_xlabel('Reflectivity')
    plt.grid()
    plt.gca().invert_yaxis()
    plt.setp(plt.xticks()[1],rotation='90',fontsize=9)
    plt.setp(plt.yticks()[1],visible=False)

    plt.show()


def plotTimeDepth(d,v):
    """
    docstring for plotTimeDepth
    """

    dpth,t = getTimeDepth(d,v)
    plt.figure()
    plt.plot(dpth,t,linewidth=2);
    plt.title('Depth-Time');
    plt.grid()
    plt.gca().set_xlabel('Depth (m)',fontsize=9)
    plt.gca().set_ylabel('Two Way Time (s)',fontsize=9)


    plt.show()


def plotSeismogram(d, rho, v, wavtyp, wavf, usingT=True):
    """
    docstring for plotSeismogram
    """

    tseis, seis, twav, wav, tref, rseriesconv = syntheticSeismogram(d, rho, v, wavtyp, wavf, usingT)

    plt.figure()

    plt.subplot(131)
    plt.plot(wav,twav,linewidth=1,color='black')
    plt.title('Wavelet')
    plt.xlim((-1.,1.))
    plt.grid()
    plt.gca().invert_yaxis()
    plt.setp(plt.xticks()[1],rotation='90',fontsize=9)
    plt.setp(plt.yticks()[1],fontsize=9)

    plt.subplot(132)
    plt.plot(np.zeros(tref.size),(tseis.max(),tseis.min()),linewidth=2,color='black')
    plt.hlines(tref,np.zeros(len(rseriesconv)),rseriesconv,linewidth=2) #,'marker','none'
    plt.title('Reflectivity')
    plt.grid()
    plt.ylim((0,tseis.max()))
    plt.gca().invert_yaxis()
    plt.xlim((-1.,1.))
    plt.setp(plt.xticks()[1],rotation='90',fontsize=9)
    plt.setp(plt.yticks()[1],fontsize=9)

    plt.subplot(133)
    plt.plot(seis,tseis,color='black',linewidth=1)
    plt.title('Seismogram')
    plt.grid()
    plt.ylim((tseis.min(),tseis.max()))
    plt.gca().invert_yaxis()
    plt.xlim((-1.,1.))
    plt.setp(plt.xticks()[1],rotation='90',fontsize=9)
    plt.setp(plt.yticks()[1],fontsize=9)
    plt.show()


## INTERACTIVE PLOT WRAPPERS
def plotLogsInteract(d2,d3,rho1,rho2,rho3,v1,v2,v3,usingT=False):
    """
    docstring plotLogsInteract
    """
    d   = np.array((0.,d2,d3), dtype=float)
    rho = np.array((rho1,rho2,rho3), dtype=float)
    v   = np.array((v1,v2,v3), dtype=float)
    plotLogs(d, rho, v, usingT)


def plotTimeDepthInteract(d2,d3,v1,v2,v3):
    """
    docstring plotTimeDepthInteract
    """
    d   = np.array((0.,d2,d3), dtype=float)
    v   = np.array((v1,v2,v3), dtype=float)
    plotTimeDepth(d,v)

def plotSeismogramInteract(f)

if __name__ == '__main__':

    d      = [0., 50., 100.]      # Position of top of each layer (m)
    v      = [1000.,  1000., 1500.]  # Velocity of each layer (m/s)
    rho    = [2000., 2300., 2500.] # Density of each layer (kg/m^3)
    wavtyp = 'RICKER'           # Wavelet type
    wavf   = [50.]              # Wavelet Frequency
    usingT = False               # Use Transmission Coefficients?

#    syntheticSeismogram(d, rho, v, wavtyp, wavf, usingT)

    plotLogsInteract(d[1],d[2],rho[0],rho[1],rho[2],v[0],v[1],v[2])
    #plotTimeDepth(d,v)
    #plotSeismogram(d, rho, v, wavtyp, wavf, usingT)
