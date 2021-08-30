from ipywidgets import interact, FloatSlider, ToggleButtons, IntSlider, FloatText, IntText, Checkbox, RadioButtons
from scipy.interpolate import interp1d
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

###############################################
# BASIC FUNCTIONS
###############################################

def compute_vint_analytic(v0, a, T, dt, tmax):
    """
    Compute the interval velocity at all times
    v0: global amplitude constant
    a: relative amplitude of sinusoid
    T: oscillation period in seconds
    dt: sampling interval
    tmax: maximum time
    """
    
    t = np.arange(0., tmax+0.001*dt, dt)
    
    return v0*(1 + a*np.sin(2.*np.pi*t/T))

def compute_vrms_analytic(v0, a, T, dt, tmax):
    """
    Compute the RMS velocity
    v0: global amplitude constant
    a: relative amplitude of sinusoid
    T: oscillation period in seconds
    dt: sampling interval
    tmax: maximum time
    """
    
    t = np.arange(dt, tmax+0.001*dt, dt)
    
    w = 2.*np.pi/T
    vrms = 2.*(a**2+2.)*w*t - a**2*np.sin(2.*w*t) - 8.*a*np.cos(w*t) + 8.*a
    return np.r_[v0, v0*np.sqrt(vrms/(4.*w*t))]

def compute_vint_inverted(t, vrms):
    """
    Invert to recover interval velocity.
    t: times (s)
    vrms: RMS velocities (m/s)
    """
    
    # Compute derivative at times using forward difference
    dvdt_approx = np.diff(vrms)/np.diff(t)
    
    # Return the approximated intervel velocity at the sampled times
    vint = vrms[:-1] * np.sqrt(np.abs(1 + 2*t[:-1]*dvdt_approx/vrms[:-1]))
    
    return vint


################################################
# PLOTTING FUNCTIONS CALLED BY APPS
################################################

def widget_fun_vint_vrms_analytic(v0, a, T, tmax):
    """
    Widget function for computing and plotting the true interval
    velocity and resulting RMS velocity
    
    v0: global amplitude constant
    a: relative amplitude of sinusoid
    T: oscillation period in seconds
    tmax: maximum time    
    """
    
    mpl.rcParams.update({'font.size': 12})
    
    # Compute true vint and vrms
    dt = T/200
    t = np.arange(0., tmax+0.001*dt, dt)  # ensure machine precision issues are not problematic
    vint = compute_vint_analytic(v0, a, T, dt, tmax)
    vrms = compute_vrms_analytic(v0, a, T, dt, tmax)
    
    # Plot
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.8])
    
    ax.plot(t, vint, 'k', lw=2)
    ax.plot(t, vrms, 'b', lw=2)
    ax.set_xlim([0., np.max(t)])
    ax.grid()
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Interval Velocity vs. RMS Velocity')
    ax.legend(['Interval Velocity', 'RMS Velocity'], loc='upper right')
    
    plt.show(fig)


def widget_fun_infinite_data(v0, a, T, tmax):
    """
    Widget function for plotting the inversion results when an infinite number
    of noiseless Vrms data are provided
    
    v0: global amplitude constant
    a: relative amplitude of sinusoid
    T: oscillation period in seconds
    tmax: maximum time    
    """
    
    mpl.rcParams.update({'font.size': 12})
    
    dt = T/200.
    
    # Compute true vint and vrms
    t = np.arange(0., tmax+0.001*dt, dt)  # ensure machine precision issues are not problematic
    vint = compute_vint_analytic(v0, a, T, dt, tmax)
    vrms = compute_vrms_analytic(v0, a, T, dt, tmax)
    
    # Compute recovered vint
    vint_inv = compute_vint_inverted(t, vrms)
    
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.8])
    
    ax.plot(t, vint, 'k', lw=2)
    ax.plot(t, vrms, 'b', lw=2)
    ax.plot(t[:-1], vint_inv, 'r--', lw=2)
    ax.set_xlim([0., np.max(t)])
    ax.grid()
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Recovered Interval Velocity Using Infinite Noiseless Data')
    ax.legend(['True Interval Velocity', 'RMS Velocity', 'Recovered Interval Velocity'], loc='upper right')
    
    plt.show(fig)

    
def widget_fun_sampled_data(v0, a, T, tmax, n, interp_type='linear'):
    """
    Widget function for plotting the inversion results for inversion
    with a finite number of uniformly sampled points
    
    v0: global amplitude constant
    a: relative amplitude of sinusoid
    T: oscillation period in seconds
    tmax: maximum time    
    n: number of samples per period
    interp_type: 'linear', 'quadratic', 'cubic'
    """
    
    mpl.rcParams.update({'font.size': 12})
    
    # Compute true vint and vrms
    dt_1 = T/200
    t_1 = np.arange(0., tmax+0.001*dt_1, dt_1)  # ensure machine precision issues are not problematic
    vint_true = compute_vint_analytic(v0, a, T, dt_1, tmax)
    vrms_true = compute_vrms_analytic(v0, a, T, dt_1, tmax)
    
    # Compute sampled and interpolated RMS velocity
    dt_2 = T/n
    t_2 = np.arange(0., tmax+0.001*dt_2, dt_2)
    vrms_obs = compute_vrms_analytic(v0, a, T, dt_2, tmax)
    interp_fun = interp1d(t_2, vrms_obs, kind=interp_type, fill_value='extrapolate', assume_sorted=True)
    vrms_interp = interp_fun(t_1)
    
    # Compute recovered interval velocity
    vint_inv = compute_vint_inverted(t_1, vrms_interp)
    
    # PLOT
    fig = plt.figure(figsize=(8, 9))
    
    # True interval and rms velocities
    ax1 = fig.add_axes([0.1, 0.65, 0.85, 0.25])
    ax1.plot(t_1, vint_true, 'k', lw=2)
    ax1.plot(t_1, vrms_true, 'b', lw=2)
    ax1.set_xlim([0., np.max(t_1)])
    ax1.set_ylim([v0-1.2*v0*a, v0+1.2*v0*a])
    ax1.set_xticklabels([])
    ax1.grid()
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('True Interval Velocity and RMS Velocity')
    ax1.legend(['True Interval Velocity', 'True RMS Velocity'], loc='upper right')
    
    ax2 = fig.add_axes([0.1, 0.35, 0.85, 0.25])
    ax2.plot(t_2, vrms_obs, 'bo', markersize=5)
    ax2.plot(t_1, vrms_interp, 'r--', lw=2)
    ax2.set_xlim([0., np.max(t_1)])
    ax2.set_ylim([v0-1.2*v0*a, v0+1.2*v0*a])
    ax2.set_xticklabels([])
    ax2.grid()
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Sampling and Interpolating RMS Velocity')
    ax2.legend(['Sampled RMS Velocity', 'Interpolated RMS Velocity'], loc='upper right')
    
    ax3 = fig.add_axes([0.1, 0.05, 0.85, 0.25])
    ax3.plot(t_1, vint_true, 'k', lw=2)
    ax3.plot(t_1[:-1], vint_inv, 'r', lw=2)
    ax3.set_xlim([0., np.max(t_1)])
    ax3.grid()
    ax3.set_ylim([v0-1.2*v0*a, v0+1.2*v0*a])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('True and Recovered Interval Velocity')
    ax3.legend(['True Interval Velocity', 'Recovered Interval Velocity'], loc='upper right')
    
    plt.show(fig)

def widget_fun_noisy_data(v0, a, T, tmax, n, std, interp_type='linear'):
    """
    Widget function for plotting the inversion results for inversion
    with a finite number of uniformly sampled points
    
    v0: global amplitude constant
    a: relative amplitude of sinusoid
    T: oscillation period in seconds
    tmax: maximum time
    n: number of samples per period
    std: standard deviation for added Gaussian noise
    interp_type: 'linear', 'quadratic', 'cubic'
    show_errors: show error bars
    """
    
    mpl.rcParams.update({'font.size': 12})
    np.random.seed(233)
    
    # Compute true vint and vrms
    dt_1 = T/200
    t_1 = np.arange(0., tmax+0.001*dt_1, dt_1)  # ensure machine precision issues are not problematic
    vint_true = compute_vint_analytic(v0, a, T, dt_1, tmax)
    vrms_true = compute_vrms_analytic(v0, a, T, dt_1, tmax)
    
    # Compute sampled and interpolated RMS velocity
    dt_2 = T/n
    t_2 = np.arange(0., tmax+0.001*dt_2, dt_2)
    vrms_obs = compute_vrms_analytic(v0, a, T, dt_2, tmax)
    vrms_obs += std * np.random.rand(len(vrms_obs))
    interp_fun = interp1d(t_2, vrms_obs, kind=interp_type, fill_value='extrapolate', assume_sorted=True)
    vrms_interp = interp_fun(t_1)
    
    # Compute recovered interval velocity
    vint_inv = compute_vint_inverted(t_1, vrms_interp)
    
    # PLOT
    fig = plt.figure(figsize=(8, 9))
    
    # True interval and rms velocities
    ax1 = fig.add_axes([0.1, 0.65, 0.85, 0.25])
    ax1.plot(t_1, vint_true, 'k', lw=2)
    ax1.plot(t_1, vrms_true, 'b', lw=2)
    ax1.set_xlim([0., np.max(t_1)])
    ax1.set_ylim([v0-1.2*v0*a, v0+1.2*v0*a])
    ax1.set_xticklabels([])
    ax1.grid()
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('True Interval Velocity and RMS Velocity')
    ax1.legend(['True Interval Velocity', 'True RMS Velocity'], loc='upper right')
    
    ax2 = fig.add_axes([0.1, 0.35, 0.85, 0.25])
    ax2.plot(t_2, vrms_obs, 'bo', markersize=5)
    ax2.plot(t_1, vrms_interp, 'r--', lw=2)
    ax2.set_xlim([0., np.max(t_1)])
    ax2.set_ylim([v0-1.2*v0*a, v0+1.2*v0*a])
    ax2.set_xticklabels([])
    ax2.grid()
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Sampling and Interpolating RMS Velocity')
    ax2.legend(['Sampled RMS Velocity', 'Interpolated RMS Velocity'], loc='upper right')
    
    ax3 = fig.add_axes([0.1, 0.05, 0.85, 0.25])
    ax3.plot(t_1, vint_true, 'k', lw=2)
    ax3.plot(t_1[:-1], vint_inv, 'r', lw=2)
    ax3.set_xlim([0., np.max(t_1)])
    ax3.grid()
    ax3.set_ylim([v0-1.2*v0*a, v0+1.2*v0*a])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('True and Recovered Interval Velocity')
    ax3.legend(['True Interval Velocity', 'Recovered Interval Velocity'], loc='upper right')
    
    plt.show(fig)




###############################################
# NOTEBOOK APPS
###############################################

def ForwardWidget():

    i = interact(
        widget_fun_vint_vrms_analytic,
        v0=FloatSlider(
            min=1.,
            max=5.,
            value=2.,
            step=0.25,
            continuous_update=False,
            readout_format='.2f',
            description="$v_0$ [m/s]",
        ),
        a=FloatSlider(
            min=0.1,
            max=1.,
            value=0.5,
            step=0.1,
            continuous_update=False,
            readout_format='.1f',
            description="$a$",
        ),
        T=FloatSlider(
            min=0.0002,
            max=0.005,
            value=0.001,
            step=0.0002,
            continuous_update=False,
            readout_format='.4f',
            description="T [s]",
        ),
        tmax=FloatSlider(
            min=0.001,
            max=0.01,
            value=0.005,
            step=0.001,
            continuous_update=False,
            readout_format='.3f',
            description="tmax [s]",
        ),
    )

    return i



def InversionBasicWidget():

    i = interact(
        widget_fun_infinite_data,
        v0=FloatSlider(
            min=1.,
            max=5.,
            value=2.,
            step=0.25,
            continuous_update=False,
            readout_format='.2f',
            description="$v_0$ [m/s]",
        ),
        a=FloatSlider(
            min=0.1,
            max=1.,
            value=0.5,
            step=0.1,
            continuous_update=False,
            readout_format='.1f',
            description="$a$",
        ),
        T=FloatSlider(
            min=0.0002,
            max=0.005,
            value=0.001,
            step=0.0002,
            continuous_update=False,
            readout_format='.4f',
            description="T [s]",
        ),
        tmax=FloatSlider(
            min=0.001,
            max=0.01,
            value=0.005,
            step=0.001,
            continuous_update=False,
            readout_format='.3f',
            description="tmax [s]",
        ),
    )

    return i



def InversionSampledDataWidget():

    i = interact(
        widget_fun_sampled_data,
        v0=FloatSlider(
            min=1.,
            max=5.,
            value=2.,
            step=0.25,
            continuous_update=False,
            readout_format='.2f',
            description="$v_0$ [m/s]",
        ),
        a=FloatSlider(
            min=0.1,
            max=1.,
            value=0.5,
            step=0.1,
            continuous_update=False,
            readout_format='.1f',
            description="$a$",
        ),
        T=FloatSlider(
            min=0.0002,
            max=0.005,
            value=0.001,
            step=0.0002,
            continuous_update=False,
            readout_format='.4f',
            description="T [s]",
        ),
        tmax=FloatSlider(
            min=0.001,
            max=0.01,
            value=0.005,
            step=0.001,
            continuous_update=False,
            readout_format='.3f',
            description="tmax [s]",
        ),
        n=IntSlider(
            min=1,
            max=50,
            value=6,
            step=1,
            continuous_update=False,
            description="n",
        ),
        interp_type=RadioButtons(
            options=['linear', 'quadratic', 'cubic'],
            value='quadratic',
            description='Interpolation:'
        ),
    )

    return i


def InversionNoisyDataWidget():

    i = interact(
        widget_fun_noisy_data,
        v0=FloatSlider(
            min=1.,
            max=5.,
            value=2.,
            step=0.25,
            continuous_update=False,
            readout_format='.2f',
            description="$v_0$ [m/s]",
        ),
        a=FloatSlider(
            min=0.1,
            max=1.,
            value=0.5,
            step=0.1,
            continuous_update=False,
            readout_format='.1f',
            description="$a$",
        ),
        T=FloatSlider(
            min=0.0002,
            max=0.005,
            value=0.001,
            step=0.0002,
            continuous_update=False,
            readout_format='.4f',
            description="T [s]",
        ),
        tmax=FloatSlider(
            min=0.001,
            max=0.01,
            value=0.005,
            step=0.001,
            continuous_update=False,
            readout_format='.3f',
            description="tmax [s]",
        ),
        n=IntSlider(
            min=1,
            max=50,
            value=6,
            step=1,
            continuous_update=False,
            description="n",
        ),
        std=FloatSlider(
            min=0,
            max=0.05,
            value=0.02,
            step=0.005,
            continuous_update=False,
            readout_format='.3f',
            description="std",
        ),
        interp_type=RadioButtons(
            options=['linear', 'quadratic', 'cubic'],
            value='quadratic',
            description='Interpolation:'
        ),
    )

    return i

