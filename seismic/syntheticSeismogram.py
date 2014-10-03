import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def getPhysPropLogs(d, rho, v, usingT=True, resolution=400):
    """
    function getLogs(d,rho,v,usingT)


    """

    # Ensure that these are float numpy arrays
    v, rho, d = np.array(v, dtype=float),   np.array(rho, dtype=float), np.array(d, dtype=float)
    usingT    = np.array(usingT, dtype=bool)

    nlayer = len(v) # number of layers

    # Check that the number of layers match
    assert len(rho) == nlayer, 'Number of layer densities must match number of layer velocities'
    #assert len(d)   == nlayer, 'Number of layer tops must match the number of layer velocities'

    dpth = np.linspace(0,300.,resolution) # create depth vector
    nd   = len(dpth)

    rholog  = np.zeros(nd)  # density
    vlog    = np.zeros(nd)  # velocity

    # Loop over layers to put information in logs
    for i in range(nlayer):
        di         = (dpth >= d[i]) # current depth indicies
        rholog[di] = rho[i]         # density
        vlog[di]   = v[i]           # velocity

        if i < nlayer-1:
            di = np.logical_and(di, dpth < d[i+1])
            ir = np.arange(resolution)[di][-1:][0]

    return rholog, vlog


def getImpedance(rholog,vlog):
    """
    docstring for getImpedance
    """
    rholog, vlog = np.array(rholog, dtype=float), np.array(vlog, dtype=float),
    return rholog*vlog


def getReflectivity(d,rho,v,usingT=True,resolution=400):
    Z   = getImpedance(rho,v)         # acoustic impedance
    R   = np.diff(Z)/(Z[:-1] + Z[1:]) # reflection coefficients

    nlayer = len(v) # number of layers
    dpth = np.linspace(0,300.,resolution) # create depth vector
    nd   = len(dpth)

    rseries    = np.zeros(nd)  # velocity

    for i in range(nlayer):
        di  = (dpth >= d[i]) # current depth indicies
        if i < nlayer-1:
            di  = np.logical_and(di, dpth < d[i+1])
            ir = np.arange(resolution)[di][-1:][0]
            if usingT:
                if i == 0:
                    rseries[ir] = R[i]
                else:
                    rseries[ir] = R[i]*np.prod(1-R[i-1]**2)
            else:
                rseries[ir] = R[i]

    return rseries, R


def getTimeDepth(d,v,resolution=400):
    """
    docstring for getTimeDepth
    """

    d = np.sort(d) 

    twttop  = 2.*np.diff(d)/v[:-1]     # 2-way travel time within each layer
    twttop  = np.cumsum(twttop)       # 2-way travel time from surface to top of each layer

    nlayer = len(d)

    dpth = np.linspace(0,300.,resolution) # create depth vector
    nd   = len(dpth)

    t       = np.zeros(nd)  # time

    for i in range(nlayer):
        di  = (dpth >= d[i]) # current depth indicies
        if i < nlayer-1:
            di  = np.logical_and(di, dpth < d[i+1])
        if i > 0:
            t[di] = 2.*(dpth[di] - d[i])/v[i] + twttop[i-1]
        else:
            t[di] = 2.*dpth[di]/v[i]

    return t


def getLogs(d, rho, v, usingT=True, resolution=400):
    """
    docstring for getLogs
    """
    rholog, vlog  = getPhysPropLogs(d, rho, v, usingT, resolution)
    zlog          = getImpedance(rholog,vlog)
    rseries,_     = getReflectivity(d, rho, v, usingT, resolution)
    return rholog, vlog, zlog, rseries


def plotLogFormat(log, dpth,xlim, col='blue'):
    """
    docstring for plotLogFormat
    """

    #xmin = log.min() -1.*0.1*(log.max()-log.min())
    #xmax = log.max() +1.*0.1*(log.max()-log.min())
    #xlim = (xmin,xmax)

    ax = plt.plot(log,dpth,linewidth=2,color=col)
    plt.xlim(xlim)
    plt.ylim((dpth.min(),dpth.max()))
    plt.grid()
    plt.gca().invert_yaxis()
    plt.setp(plt.xticks()[1],rotation='90',fontsize=9)
    plt.setp(plt.yticks()[1],fontsize=9)

    return ax



def plotLogs(d, rho, v, usingT=True, resolution=400):
    """
    function plotLogs(d,rho,v,usingT)
    """
    d = np.sort(d)

    rholog, vlog, zlog, rseries  = getLogs(d, rho, v, usingT, resolution)
    t = getTimeDepth(d,v,resolution)

    dpth = np.array(np.linspace(0.,300.,resolution)) # create depth vector
    nd   = len(dpth)


    plt.figure(1)

    xlimrho = (1.99,5.01)
    xlimv   = (0.29,4.01)
    xlimz   = (xlimrho[0]*xlimv[0], xlimrho[1]*xlimv[1])

    # Plot Density
    plt.subplot(141)
    plotLogFormat(rholog*10**-3,dpth,xlimrho,'blue')
    plt.title('$\\rho$')
    plt.xlabel('Density \n ($\\times 10^3$ kg /m$^3$)',fontsize=9)
    plt.ylabel('Depth (m)',fontsize=9)

    plt.subplot(142)
    plotLogFormat(vlog*10**-3,dpth,xlimv,'red')
    plt.title('$v$')
    plt.xlabel('Velocity \n ($\\times 10^3$ m/s)',fontsize=9)
    plt.setp(plt.yticks()[1],visible=False)

    plt.subplot(143)
    plotLogFormat(zlog*10.**-6.,dpth,xlimz,'green')
    plt.gca().set_title('$Z = \\rho v$')
    plt.gca().set_xlabel('Impedance \n $\\times 10^{6}$ (kg m$^{-2}$ s$^{-1}$)',fontsize=9)
    plt.setp(plt.yticks()[1],visible=False)

    plt.subplot(144)
    plt.hlines(dpth,np.zeros(nd),rseries,linewidth=2)
    plt.plot(np.zeros(nd),dpth,linewidth=2,color='black')
    plt.title('Reflectivity');
    plt.xlim((-1.,1.))
    plt.gca().set_xlabel('Reflectivity')
    plt.grid()
    plt.gca().invert_yaxis()
    plt.setp(plt.xticks()[1],rotation='90',fontsize=9)
    plt.setp(plt.yticks()[1],visible=False)

    plt.show()


def plotTimeDepth(d,v,resolution=400):
    """
    docstring for plotTimeDepth
    """
    dpth = np.linspace(0,300.,resolution) # create depth vector
    nd   = len(dpth)

    t = getTimeDepth(d,v,resolution)

    plt.figure()
    plt.plot(t,dpth,linewidth=2);
    plt.title('Depth-Time');
    plt.grid()
    plt.gca().invert_yaxis()
    plt.gca().set_xlabel('Time (s)',fontsize=9)
    plt.gca().set_ylabel('Depth (m)',fontsize=9)

    plt.show()




def syntheticSeismogram(d, rho, v, wavtyp, wavf, usingT=True, resolution=400):
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
        usingT :



    Lindsey Heagy
    lheagy@eos.ubc.ca
    Created:  November 30, 2013
    Modified: October 2, 2014

    Defaults:
    v   = np.array([350, 1000, 2000])  # Velocity of each layer (m/s)
    rho = np.array([1700, 2000, 2500]) # Density of each layer (kg/m^3)
    d   = np.array([0, 100, 200])      # Position of top of each layer (m)
    """

    v, rho, d = np.array(v, dtype=float),   np.array(rho, dtype=float), np.array(d, dtype=float)
    usingT    = np.array(usingT, dtype=bool)

    rholog, vlog = getPhysPropLogs(d, rho, v, usingT, resolution)

    t = getTimeDepth(d,v)
    rseries,R = getReflectivity(d,rho,v)

    # make wavelet
    dtwav  = np.abs(np.min(np.diff(t)))/10.0
    twav   = np.arange(-2.0/np.min(wavf), 2.0/np.min(wavf), dtwav)

    # Get source wavelet
    wav = {'RICKER':getRicker, 'ORMSBY':getOrmsby, 'KLAUDER':getKlauder}[wavtyp](wavf,twav)

    # create synthetic seismogram
    tref  = np.arange(0,np.max(t),dtwav) #+ np.min(twav)  # time discretization for reflectivity series
    tr    = t[np.abs(rseries) > 0]
    rseriesconv = np.zeros(len(tref))

    for i in range(len(tr)):
        index = np.abs(tref - tr[i]).argmin()
        rseriesconv[index] = R[i]

    seis  = np.convolve(wav,rseriesconv)
    tseis = np.min(twav)+dtwav*np.arange(len(seis))
    index = np.logical_and(tseis >= 0, tseis <= np.max(t))
    tseis = tseis[index]
    seis  = seis[index]

    ##
    plt.figure(3)

    plt.subplot(131)
    plt.plot(wav,twav,linewidth=1,color='black')
    plt.title('Wavelet')
    plt.xlim((-1.,1.))
    plt.grid()
    plt.gca().invert_yaxis()
    plt.setp(plt.xticks()[1],rotation='90',fontsize=9)
    plt.setp(plt.yticks()[1],fontsize=9)

    plt.subplot(132)
    plt.plot(np.zeros(tref.size),tref,linewidth=2,color='black')
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
    plt.ylim((0,tseis.max()))
    plt.gca().invert_yaxis()
    plt.xlim((-1.,1.))
    plt.setp(plt.xticks()[1],rotation='90',fontsize=9)
    plt.setp(plt.yticks()[1],fontsize=9)
    plt.show()



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


# INTERACTIVE PLOT WRAPPERS
def plotLogsInteract(d2,d3,rho1,rho2,rho3,v1,v2,v3):
    """
    docstring plotLogsInteract
    """
    d   = np.array((0.,d2,d3), dtype=float)
    rho = np.array((rho1,rho2,rho3), dtype=float)
    v   = np.array((v1,v2,v3), dtype=float)
    plotLogs(d, rho, v)

def plotTimeDepthInteract(d2,d3,v1,v2,v3):
    """
    docstring plotTimeDepthInteract
    """
    d   = np.array((0.,d2,d3), dtype=float)
    v   = np.array((v1,v2,v3), dtype=float)
    plotTimeDepth(d,v)

if __name__ == '__main__':

    d      = [0., 50., 100.]      # Position of top of each layer (m)
    v      = [600.,  1000., 1500.]  # Velocity of each layer (m/s)
    rho    = [2000., 2300., 2500.] # Density of each layer (kg/m^3)
    wavtyp = 'RICKER'           # Wavelet type
    wavf   = [50.]              # Wavelet Frequency
    usingT = False               # Use Transmission Coefficients?

#    syntheticSeismogram(d, rho, v, wavtyp, wavf, usingT)

    #plotLogsInteract(d[1],d[2],rho[0],rho[1],rho[2],v[0],v[1],v[2])
    #plotTimeDepth(d,v)
    syntheticSeismogram(d, rho, v, wavtyp, wavf, usingT)