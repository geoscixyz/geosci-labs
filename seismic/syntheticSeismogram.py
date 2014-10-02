import numpy as np
import matplotlib.pyplot as plt



def syntheticSeismogram(v, rho, d, wavtyp, wavf, usingT):
    """
    function syntheticSeismogram(v, rho, d, wavtyp, wavf, usingT)

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
    Modified: October 1, 2014

    v   = np.array([350, 1000, 2000])  # Velocity of each layer (m/s)
    rho = np.array([1700, 2000, 2500]) # Density of each layer (kg/m^3)
    d   = np.array([0, 100, 200])      # Position of top of each layer (m)
    """

    # Ensure that these are float numpy arrays
    v, rho, d , wavf = np.array(v, dtype=float),   np.array(rho, dtype=float), np.array(d, dtype=float), np.array(wavf,dtype=float)
    usingT           = np.array(usingT, dtype=bool)

    nlayer = len(v) # number of layers

    # Check that the number of layers match
    assert len(rho) == nlayer, 'Number of layer densities must match number of layer velocities'
    assert len(d)   == nlayer, 'Number of layer tops must match the number of layer velocities'

    # compute necessary parameters
    Z   = rho*v                       # acoustic impedance
    R   = np.diff(Z)/(Z[:-1] + Z[1:]) # reflection coefficients
    twttop  = 2*np.diff(d)/v[:-1]     # 2-way travel time within each layer
    twttop  = np.cumsum(twttop)       # 2-way travel time from surface to top of each layer

    # create model logs
    resolution = 400                                                      # How finely we discretize in depth
    dpth       = np.linspace(0,np.max(d)+3*np.max(np.diff(d)),resolution) # create depth vector
    nd         = len(dpth)

    # Initialize logs
    rholog  = np.zeros(nd)  # density
    vlog    = np.zeros(nd)  # velocity
    zlog    = np.zeros(nd)  # acoustic impedance
    rseries = np.zeros(nd)  # reflectivity series
    t       = np.zeros(nd)  # time

    # Loop over layers to put information in logs
    for i in range(nlayer):
        di         = (dpth >= d[i]) # current depth indicies
        rholog[di] = rho[i]         # density
        vlog[di]   = v[i]           # velocity
        zlog[di]   = Z[i]           # acoustic impedance
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
        if i > 0:
            t[di] = 2*(dpth[di] - d[i])/v[i] + twttop[i-1]
        else:
            t[di] = 2*dpth[di]/v[i]


    # make wavelet
    dtwav  = np.abs(np.min(np.diff(t)))/10.0
    twav   = np.arange(-2.0/np.min(wavf), 2.0/np.min(wavf), dtwav)

    # Get source wavelet
    wav = {'RICKER':getRicker, 'ORMSBY':getOrmsby, 'KLAUDER':getKlauder}[wavtyp](wavf,twav)

    # create synthetic seismogram
    tref  = np.arange(0,np.max(t),dtwav) + np.min(twav)  # time discretization for reflectivity series
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
    plt.figure(1)

    # Plot Density
    plt.subplot(141)
    plt.plot(rholog,dpth,linewidth=2)
    plt.title('Density')
    # xlim([min(rholog) max(rholog)] + [-1 1]*0.1*[max(rholog)-min(rholog)])
    # ylim([min(dpth),max(dpth)])
    # set(gca,'Ydir','reverse')
    plt.grid()
    plt.gca().invert_yaxis()

    plt.subplot(142)
    plt.plot(vlog,dpth,linewidth=2)
    plt.title('Velocity')
    # xlim([min(vlog) max(vlog)] + [-1 1]*0.1*[max(vlog)-min(vlog)])
    # ylim([min(dpth),max(dpth)])
    # set(gca,'Ydir','reverse')
    plt.grid()
    plt.gca().invert_yaxis()

    plt.subplot(143)
    plt.plot(zlog,dpth,linewidth=2)
    plt.title('Acoustic Impedance')
    # xlim([min(zlog) max(zlog)] + [-1 1]*0.1*[max(zlog)-min(zlog)])
    # ylim([min(dpth),max(dpth)])
    # set(gca,'Ydir','reverse')
    plt.grid()
    plt.gca().invert_yaxis()

    plt.subplot(144)
    plt.hlines(dpth,np.zeros(nd),rseries,linewidth=2) #,'marker','none'
    plt.title('Reflectivity Series');
    # set(gca,'cameraupvector',[-1, 0, 0]);
    plt.grid()
    plt.gca().invert_yaxis()
    # set(gca,'ydir','reverse');

    plt.figure(2)
    plt.plot(t,dpth,linewidth=2);
    plt.title('Depth-Time');
    # plt.xlim([np.min(t), np.max(t)] + [-1, 1]*0.1*[np.max(t)-np.min(t)]);
    # plt.ylim([np.min(dpth),np.max(dpth)]);
    # set(gca,'Ydir','reverse');
    plt.grid()
    plt.gca().invert_yaxis()
    ##
    plt.figure(3)
    # plt.subplot(141)
    # plt.plot(dpth,t,linewidth=2);
    # title('Time-Depth');
    # ylim([min(t), max(t)] + [-1 1]*0.1*[max(t)-min(t)]);
    # xlim([min(dpth),max(dpth)]);
    # set(gca,'Ydir','reverse');
    # plt.grid()
    plt.gca().invert_yaxis()

    plt.subplot(132)
    plt.hlines(tref,np.zeros(len(rseriesconv)),rseriesconv,linewidth=2) #,'marker','none'
    plt.title('Reflectivity Series')
    plt.grid()
    plt.gca().invert_yaxis()

    plt.subplot(131)
    plt.plot(wav,twav,linewidth=2)
    plt.title('Wavelet')
    plt.grid()
    plt.gca().invert_yaxis()
    # set(gca,'ydir','reverse')

    plt.subplot(133)
    plt.plot(seis,tseis,linewidth=2)
    plt.grid()
    plt.gca().invert_yaxis()
    # set(gca,'ydir','reverse')

    plt.show()


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

if __name__ == '__main__':

    d      = [0, 50, 100]      # Position of top of each layer (m)
    v      = [350, 1000, 2000]  # Velocity of each layer (m/s)
    rho    = [1700, 2000, 2500] # Density of each layer (kg/m^3)
    wavtyp = 'RICKER'           # Wavelet type
    wavf   = [100]              # Wavelet Frequency
    usingT = False               # Use Transmission Coefficients?

    syntheticSeismogram(v, rho, d, wavtyp, wavf, usingT)
