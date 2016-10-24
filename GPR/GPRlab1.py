import numpy as np
from Wiggle import wiggle, PrimaryWave, ReflectedWave
from scipy.constants import mu_0, epsilon_0
import matplotlib.pyplot as plt
from PIL import Image

def PrimaryWidget(tinterpL, epsrL, tinterpH, epsrH):
    data = np.load('shot_raypaths.npy')
    time = np.load('time.npy')
    dt = time[1]-time[0]
    v1 = 1./np.sqrt(epsilon_0*epsrL*mu_0)
    v2 = 1./np.sqrt(epsilon_0*epsrH*mu_0)
    dx = 0.3
    nano = 1e9
    xorig = np.arange(data.shape[0])*dx
    out1 = PrimaryWave(xorig, v1, tinterpL/nano)
    out2 = ReflectedWave(xorig, v2, tinterpH/nano)

    kwargs = {
    'skipt':1,
    'scale': 0.5,
    'lwidth': 0.1,
    'dx': dx,
    'sampr': dt*nano,
    }

    extent = [0., 30, 300, 0]
    fig, ax1 = plt.subplots(1,1, figsize = (8,5))
    ax1.invert_yaxis()
    ax1.axis(extent)
    ax1.set_xlabel('Offset (m)')
    ax1.set_ylabel('Time (ns)')
    ax1.set_title('Shot Gather')
    wiggle(data, ax = ax1, **kwargs)
    ax1.plot(xorig, out1*nano, 'b', lw = 2)
    ax1.plot(xorig, out2*nano, 'r', lw = 2)

    plt.show()

def PrimaryFieldWidget(tinterp, epsr):
    imgcmp = Image.open('./Figures/ubc_GPRcmp.png')
    fig = plt.figure(figsize = (6,7))
    ax = plt.subplot(111)
    plt.imshow(imgcmp, extent = [0, 150, 150, 0])
    x = np.arange(81)*0.1
    xconvert = x*150./8.
    v = 1./np.sqrt(mu_0*epsilon_0*epsr)
    nano = 1e9
    # tinterp = 30
    y = (1./v*x)*nano + tinterp
    plt.plot(xconvert, y, lw = 2)
    plt.xticks(np.arange(11)*15,  np.arange(11)*0.8+2.4) #+2.4 for offset correction
    plt.xlim(0., 150.)
    plt.ylabel('Time (ns)')
    plt.xlabel('Offset (m)')

    plt.show()

def PipeWidget(epsr, h, xc, r):
    imgcmp = Image.open('./Figures/ubc_GPRdata.png')
    imgcmp = imgcmp.resize((400, 500))
    fig = plt.figure(figsize = (7,8))
    ax = plt.subplot(111)

    plt.imshow(imgcmp, extent = [0, 400, 250, 0])
    x = np.arange(41)*1.
    xconvert = x*10.
    v = 1./np.sqrt(mu_0*epsilon_0*epsr)
    nano = 1e9
    time = (np.sqrt(((x-xc)**2+4*h**2)) - r)/v
    plt.plot(xconvert, time*nano, 'r--',lw = 2)
    plt.xticks(np.arange(11)*40,  np.arange(11)*4.0 )
    plt.xlim(0., 400)
    plt.ylim(250., 0.)
    plt.ylabel('Time (ns)')
    plt.xlabel('Survey line location (m)')

    plt.show()

def WallWidget(epsr, h, x1, x2):
    imgcmp = Image.open('./Figures/ubc_GPRdata.png')
    imgcmp = imgcmp.resize((400, 500))
    fig = plt.figure(figsize = (7,8))
    ax = plt.subplot(111)

    plt.imshow(imgcmp, extent = [0, 400, 250, 0])
    x = np.arange(41)*1.
    ind1 = x <= x1
    ind2 = x >= x2
    ind3 = np.logical_not(np.logical_or(ind1, ind2))
    scale = 10.
    xconvert = x*scale
    v = 1./np.sqrt(mu_0*epsilon_0*epsr)
    nano = 1e9
    def arrival(x, xc, h, v):
        return (np.sqrt(((x-xc)**2+4*h**2)))/v

    plt.plot(xconvert[ind1], arrival(x[ind1], x1, h, v)*nano, 'b--',lw = 2)
    plt.plot(xconvert[ind2], arrival(x[ind2], x2, h, v)*nano, 'b--',lw = 2)
    plt.plot(np.r_[x1*scale, x2*scale], np.r_[2.*h/v, 2.*h/v]*nano, 'b--',lw = 2)

#     plt.plot(xconvert[ind3], arrival(x[ind3], xc?, h, v)*nano, 'r--',lw = 2)
    plt.xticks(np.arange(11)*40,  np.arange(11)*4.0 )
    plt.xlim(0., 400)
    plt.ylim(250., 0.)
    plt.ylabel('Time (ns)')
    plt.xlabel('Survey line location (m)')

    plt.show()
