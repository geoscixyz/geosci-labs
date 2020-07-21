import numpy as np
import matplotlib.pyplot as plt  # Matplotlib
from matplotlib import rcParams  # To adjust some plot settings
from empymod import bipole, utils  # Load required empymod functions
from ipywidgets import interactive, FloatText, ToggleButtons, widgets
from scipy.constants import mu_0
from ipywidgets.widgets import HBox, VBox


def show_canonical_model():
    # Plot-style adjustments
    rcParams["font.size"] = 14

    d1, d2, d3, d4 = 0.0, 1000.0, 2000.0, 2200.0
    rho0, rho1, rho2, rho3, rho4 = 1e8, 0.3, 1.0, 100.0, 1.0
    lamda0, lamda1, lamda2, lamda3, lamda4 = 1.0, 1.0, 1.0, 1.0, 1.0

    depth = [d1, d2, d3, d4]  # Layer boundaries
    res = [rho0, rho1, rho2, rho3, rho4]  # Anomaly resistivities
    aniso = [
        lamda0,
        lamda1,
        lamda2,
        lamda3,
        lamda4,
    ]  # Layer anisotropies (same for anomaly and background)

    # Spatial parameters
    # zsrc = 999.0  # Src-depth
    # dx = 500.0
    # dy = 500.0
    # x = np.r_[-100.0, np.arange(10) * dx + dx]  # Offsets
    # y = np.r_[-100.0, np.arange(10) * dy + dy]  # Offsets

    # mid_d = (np.array(depth)[:-1] + np.array(depth)[1:]) * 0.5
    # z = np.r_[
    #     -500,
    #     depth[0],
    #     mid_d[0],
    #     depth[1],
    #     mid_d[1],
    #     mid_d[2],
    #     depth[2] + 500.0,
    #     depth[2] + 1000.0,
    # ]
    # nlayers = 5
    # srcloc = np.r_[0.0, 0.0, zsrc]

    pdepth = np.repeat(np.r_[-500, depth], 2)
    pdepth[:-1] = pdepth[1:]
    pdepth[-1] = 2 * depth[-1]
    pres = np.repeat(res, 2)
    pani = np.repeat(aniso, 2)

    # Create figure
    fig = plt.figure(figsize=(7, 5), facecolor="w")
    fig.subplots_adjust(wspace=0.25, hspace=0.4)
    # plt.suptitle(name, fontsize=20)

    # Plot Resistivities
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(pres, -pdepth, "k")
    plt.xscale("log")
    plt.xlim([0.2 * np.array(res).min(), 100 * np.array(res)[1:].max()])
    plt.ylim([-1.5 * depth[-1], 500])
    plt.ylabel("z (m)")
    plt.xlabel(r"Resistivity $\rho_h\ (\Omega\,\rm{m})$")

    # Plot anisotropies
    ax2 = plt.subplot(1, 2, 2)
    plt.plot(pani, -pdepth, "k")
    plt.xlim([0, 2])
    plt.ylim([-1.5 * depth[-1], 500])
    plt.xlabel(r"$\rho_v$ / $\rho_h$")
    ax2.yaxis.tick_right()
    ax1.grid(True)
    ax2.grid(True)
    ax1.text(1.5, -0.5 * (-500 + d1), "Air ($\\rho_0$)")
    ax1.text(1.5, -0.5 * (d1 + d2), "Seawater ($\\rho_1$)")
    ax1.text(1.5, -0.5 * (d2 + d3), "Sea sediment ($\\rho_2$)")
    ax1.text(1.5, -0.5 * (d3 + d4) - 50.0, "Reservoir ($\\rho_3$)")
    ax1.text(1.5, -0.5 * (d4 + d4 + 1000.0), "Sea sediment ($\\rho_4$)")
    plt.show()
    return


def csem_layered_earth(
    srcloc,
    rxlocs,
    depth,
    res,
    aniso,
    frequency,
    nlayers=5,
    src_type="electric",
    rx_type="electric",
    src_direction="x",
    rx_direction="x",
    verb=0,
):
    """
    Simulating CSEM response in a layered earth
    """
    # Safety checks
    if len(depth) != nlayers - 1:
        raise Exception("Length of depth should be nlayers-1")
    if len(res) != nlayers:
        raise Exception("Length of res should be nlayers")
    if len(aniso) != nlayers:
        raise Exception("Length of aniso should be nlayers")

    if srcloc.size != 3:
        raise Exception("size of srcloc should be 3! (x, y, z) location")

    rxlocs = np.atleast_2d(rxlocs)

    if src_direction == "x":
        src = np.r_[srcloc.flatten(), 0.0, 0.0].tolist()
    elif src_direction == "y":
        src = np.r_[srcloc.flatten(), 90.0, 0.0].tolist()
    elif src_direction == "z":
        src = np.r_[srcloc.flatten(), 0.0, 90.0].tolist()
    else:
        raise Exception("src_direction should be x, y, or z")

    if rx_direction == "x":
        rx = [rxlocs[:, 0], rxlocs[:, 1], rxlocs[:, 2], 0.0, 0.0]
    elif rx_direction == "y":
        rx = [rxlocs[:, 0], rxlocs[:, 1], rxlocs[:, 2], 90.0, 0.0]
    elif rx_direction == "z":
        rx = [rxlocs[:, 0], rxlocs[:, 1], rxlocs[:, 2], 0.0, 90.0]
    else:
        raise Exception("rx_direction should be x, y, or z")
    if len(np.unique(np.array(res))) == 1:
        xdirect = True
    else:
        xdirect = False
    inpdat = {
        "res": res,
        "src": src,
        "rec": rx,
        "depth": depth,
        "freqtime": frequency,
        "aniso": aniso,
        "verb": verb,
        "xdirect": xdirect,
        "mrec": "electric" != rx_type,
        "msrc": "electric" != src_type,
    }
    out = bipole(**inpdat)
    return utils.EMArray(out)


def viz_plane(
    x,
    z,
    vx,
    vz,
    depth,
    zsrc,
    label="Electric field (V/A-m)",
    xlabel="X (m)",
    clim=None,
    title=None,
    xlim=None,
    ylim=None,
    plane="XZ",
):
    plt.figure(figsize=(12, 5))
    ax = plt.subplot(111)
    # nskip = 5
    eps = 1e-31
    vabs = np.sqrt(vx ** 2 + vz ** 2) + eps
    fxrtemp = (-vx / vabs).reshape((x.size, z.size))[:, :].T
    fzrtemp = (-vz / vabs).reshape((x.size, z.size))[:, :].T
    fabs_temp = vabs.reshape((x.size, z.size))[:, :].T
    X, Z = np.meshgrid(x, z)

    if clim is not None:
        vabs[vabs < 10 ** clim[0]] = 10 ** clim[0]
        vabs[vabs > 10 ** clim[1]] = 10 ** clim[1]
        vmin, vmax = clim[0], clim[1]
        out = ax.contourf(
            x,
            -z,
            np.log10(vabs.reshape((x.size, z.size)).T),
            100,
            clim=(vmin, vmax),
            vmin=vmin,
            vmax=vmax,
        )
        out = ax.scatter(np.zeros(3) - 1000, np.zeros(3), c=np.linspace(vmin, vmax, 3))
    else:
        out = ax.contourf(x, -z, np.log10(vabs.reshape((x.size, z.size)).T), 100)
        vmin, vmax = out.get_clim()
    cb = plt.colorbar(out, ticks=np.linspace(vmin, vmax, 5), format="10$^{%.0f}$")
    cb.set_label(label)
    for d_temp in depth:
        ax.plot(np.r_[-100, 10000], np.ones(2) * -d_temp, "w-", lw=1, alpha=0.5)
    if plane == "XZ":
        scale = 0.0025
    else:
        scale = 0.0025
    Q = ax.quiver(
        x,
        -z,
        fxrtemp,
        fzrtemp,
        pivot="mid",
        units="xy",
        scale=scale,
        color="k",
        alpha=0.5,
    )
    Q.headlength = 2
    Q.headaxislength = 2
    temp = np.log10(fabs_temp / fabs_temp.max())
    ax.scatter(
        X.flatten(),
        -Z.flatten(),
        color="b",
        s=(temp + abs(temp).max()) / abs(temp).max() * 100.0 * 1,
        alpha=0.8,
    )
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Z (m)")
    ax.set_title(title)
    if plane == "XZ":
        ax.quiver(
            -100,
            zsrc,
            1,
            0,
            scale=3,
            units="inches",
            headlength=2,
            headaxislength=2,
            color="r",
        )
        ax.text(-100, zsrc - 200.0, "HED")
    elif plane == "YZ":
        ax.plot(0, zsrc, "ro")
        ax.text(-100, zsrc - 200.0, "HED")
    elif plane == "XY":
        ax.plot(0, 0, "ro")
        ax.text(-100, -200.0, "HED")
    #     plt.gca().set_aspect("equal")
    return out


def csem_fields_app(
    frequency,
    zsrc,
    rho0,
    rho1,
    rho2,
    rho3,
    rho4,
    dz1,
    dz2,
    dz3,
    rv_rh,
    Field,
    Plane,
    Fixed,
    vmin,
    vmax,
):

    # lamda0, lamda1, lamda2, lamda3, lamda4 = 1.0, 1.0, 1.0, 1.0, 1.0
    d = np.r_[0.0, np.cumsum(np.r_[dz1, dz2, dz3])]
    depth = [d[0], d[1], d[2], d[3]]  # Layer boundaries
    res = [rho0, rho1, rho2, rho3, rho4]  # Anomaly resistivities
    aniso = [
        1.0,
        1.0,
        np.sqrt(rv_rh),
        1.0,
        1.0,
    ]  # Layer anisotropies (same for anomaly and background)

    # Modelling parameters
    # verb = 1

    # Spatial parameters
    #     zsrc=999.               # Src-depth
    nrx = 20
    dx = 500.0
    dy = 500.0
    x = np.r_[-100.0, np.arange(nrx) * dx + dx]  # Offsets
    y = np.r_[-100.0, np.arange(nrx) * dy + dy]  # Offsets

    mid_d = (np.array(depth)[:-1] + np.array(depth)[1:]) * 0.5
    z = np.r_[
        -500,
        depth[0],
        mid_d[0],
        depth[1],
        mid_d[1],
        mid_d[2],
        depth[2] + 500.0,
        depth[2] + 1000.0,
    ]
    # nlayers = 5
    srcloc = np.r_[0.0, 0.0, -zsrc]

    if Plane == "XZ":
        X, Y, Z = np.meshgrid(x, np.r_[0.0], z)
        rxlocs = np.c_[X.flatten(), Y.flatten(), Z.flatten()]
        ex = csem_layered_earth(
            srcloc, rxlocs, depth, res, aniso, frequency, rx_direction="x"
        )
        ez = csem_layered_earth(
            srcloc, rxlocs, depth, res, aniso, frequency, rx_direction="z"
        )
        hy = csem_layered_earth(
            srcloc,
            rxlocs,
            depth,
            res,
            aniso,
            frequency,
            rx_direction="y",
            rx_type="magnetic",
        )
        x0, x1 = x.copy(), z.copy()
        xlabel = "X (m)"
        if Field == "E":
            label = "Electric field (V/m)"
            val0, val1 = -ex.real, ez.real
        elif Field == "H":
            return print("Only Hy exists, so that you cannot make vector plot")

        elif Field == "P":
            label = "Poynting vector (W/A-m$^2$)"
            sx = 0.5 * (ez * hy.conj()).real
            sz = 0.5 * (ex * hy.conj()).real
            val0, val1 = sx.copy(), sz.copy()

    elif Plane == "YZ":
        X, Y, Z = np.meshgrid(np.r_[0.0], y, z)
        rxlocs = np.c_[X.flatten(), Y.flatten(), Z.flatten()]
        ex = csem_layered_earth(
            srcloc, rxlocs, depth, res, aniso, frequency, rx_direction="x"
        )
        hy = csem_layered_earth(
            srcloc,
            rxlocs,
            depth,
            res,
            aniso,
            frequency,
            rx_direction="y",
            rx_type="magnetic",
        )
        hz = csem_layered_earth(
            srcloc,
            rxlocs,
            depth,
            res,
            aniso,
            frequency,
            rx_direction="z",
            rx_type="magnetic",
        )
        x0, x1 = y.copy(), z.copy()
        xlabel = "Y (m)"
        if Field == "E":
            return print("Only ex exists, so that you cannot make vector plot")
        elif Field == "H":
            label = "Magnetic field (A/m)"
            val0, val1 = -hy.imag, hz.imag
        elif Field == "P":
            label = "Poynting vector (W/A-m$^2$)"
            sy = 0.5 * (ex * hz.conj()).real
            sz = 0.5 * (ex * hy.conj()).real
            val0, val1 = sy.copy(), sz.copy()

    ex = csem_layered_earth(
        srcloc, rxlocs, depth, res, aniso, frequency, rx_direction="x"
    )
    ez = csem_layered_earth(
        srcloc, rxlocs, depth, res, aniso, frequency, rx_direction="z"
    )
    hy = csem_layered_earth(
        srcloc,
        rxlocs,
        depth,
        res,
        aniso,
        frequency,
        rx_direction="y",
        rx_type="magnetic",
    )
    if Fixed:
        clim = (np.log10(vmin), np.log10(vmax))
    else:
        clim = None
    ylim = (-z).min(), 100
    viz_plane(
        x0,
        x1,
        val0,
        val1,
        depth,
        zsrc,
        title=" ",
        label=label,
        xlabel=xlabel,
        plane=Plane,
        clim=clim,
        ylim=ylim,
    )


def FieldsApp():
    Q1 = interactive(
        csem_fields_app,
        rho0=FloatText(value=1e8, description="$\\rho_{0} \ (\Omega m)$"),
        rho1=FloatText(value=0.3, description="$\\rho_{1} \ (\Omega m)$"),
        rho2=FloatText(value=1.0, description="$\\rho_{2} \ (\Omega m)$"),
        rho3=FloatText(value=100.0, description="$\\rho_{3} \ (\Omega m)$"),
        rho4=FloatText(value=1.0, description="$\\rho_{4} \ (\Omega m)$"),
        zsrc=FloatText(value=-950.0, description="src height (m)"),
        rv_rh=FloatText(value=1.0, description="$\\rho_{2 \ v} / \\rho_{2 \ h}$"),
        dz1=FloatText(value=1000.0, description="dz1 (m)"),
        dz2=FloatText(value=1000.0, description="dz2 (m)"),
        dz3=FloatText(value=200.0, description="dz3 (m)"),
        frequency=FloatText(value=0.5, description="f (Hz)"),
        Field=ToggleButtons(options=["E", "H", "P"], value="E"),
        Plane=ToggleButtons(options=["XZ", "YZ"], value="XZ"),
        Fixed=widgets.widget_bool.Checkbox(value=False),
        vmin=FloatText(value=None),
        vmax=FloatText(value=None),
        __manual=True,
    )
    return Q1


def csem_data_app(
    frequency,
    zsrc,
    zrx,
    rho0,
    rho1,
    rho2,
    rho3,
    rho4,
    rv_rh,
    dz1,
    dz2,
    dz3,
    frequency_bg,
    zsrc_bg,
    zrx_bg,
    rho0_bg,
    rho1_bg,
    rho2_bg,
    rho3_bg,
    rho4_bg,
    rv_rh_bg,
    dz1_bg,
    dz2_bg,
    dz3_bg,
    Field,
    Plane,
    Component,
    Complex,
    scale,
    Fixed,
    vmin,
    vmax,
):
    rcParams["font.size"] = 16
    # lamda0, lamda1, lamda2, lamda3, lamda4 = 1.0, 1.0, 1.0, 1.0, 1.0
    d = np.r_[0.0, np.cumsum(np.r_[dz1, dz2, dz3])]
    depth = [d[0], d[1], d[2], d[3]]  # Layer boundaries
    d_bg = np.r_[0.0, np.cumsum(np.r_[dz1_bg, dz2_bg, dz3_bg])]
    depth_bg = [d_bg[0], d_bg[1], d_bg[2], d_bg[3]]  # Layer boundaries

    res = [rho0, rho1, rho2, rho3, rho4]  # Anomaly resistivities
    res_bg = [rho0_bg, rho1_bg, rho2_bg, rho3_bg, rho4_bg]  # Anomaly resistivities
    aniso = [
        1.0,
        1.0,
        np.sqrt(rv_rh),
        1.0,
        1.0,
    ]  # Layer anisotropies (same for anomaly and background)
    aniso_bg = [
        1.0,
        1.0,
        np.sqrt(rv_rh_bg),
        1.0,
        1.0,
    ]  # Layer anisotropies (same for anomaly and background)

    # Modelling parameters
    # verb = 1

    # Spatial parameters
    dx = 100.0
    dy = 100.0
    nrx = 200
    x = np.r_[np.arange(nrx) * dx + dx]  # Offsets
    y = np.r_[np.arange(nrx) * dy + dy]  # Offsets
    z = np.ones_like(x) * -zrx
    srcloc = np.r_[0.0, 0.0, -zsrc]
    srcloc = np.r_[0.0, 0.0, -zsrc]

    if Plane == "XZ":
        rxlocs = np.c_[x, np.zeros_like(x), z]
        xlabel = "X offset (m)"
        # x0 = y.copy()

    elif Plane == "YZ":
        rxlocs = np.c_[np.zeros_like(x), y, z]
        xlabel = "Y offset (m)"
        # x0 = x.copy()

    if Field == "E":
        label = "Electric field (V/m)"
        data = csem_layered_earth(
            srcloc, rxlocs, depth, res, aniso, frequency, rx_direction=Component
        )
        data_bg = csem_layered_earth(
            srcloc,
            rxlocs,
            depth_bg,
            res_bg,
            aniso_bg,
            frequency_bg,
            rx_direction=Component,
        )

    elif Field == "H":
        label = "Magnetic field (A/m)"
        data = csem_layered_earth(
            srcloc,
            rxlocs,
            depth,
            res,
            aniso,
            frequency,
            rx_direction=Component,
            rx_type="magnetic",
        )
        data_bg = csem_layered_earth(
            srcloc,
            rxlocs,
            depth_bg,
            res_bg,
            aniso_bg,
            frequency_bg,
            rx_direction=Component,
            rx_type="magnetic",
        )

    elif Field == "Zxy":
        label = "Impedance (V/A)"
        data_ex = csem_layered_earth(
            srcloc, rxlocs, depth, res, aniso, frequency, rx_direction="x"
        )
        data_ex_bg = csem_layered_earth(
            srcloc, rxlocs, depth_bg, res_bg, aniso_bg, frequency_bg, rx_direction="x"
        )
        data_hy = csem_layered_earth(
            srcloc,
            rxlocs,
            depth_bg,
            res_bg,
            aniso_bg,
            frequency_bg,
            rx_direction="y",
            rx_type="magnetic",
        )
        data_hy_bg = csem_layered_earth(
            srcloc,
            rxlocs,
            depth_bg,
            res_bg,
            aniso_bg,
            frequency_bg,
            rx_direction="y",
            rx_type="magnetic",
        )
        data = utils.EMArray(data_ex / data_hy)
        data_bg = utils.EMArray(data_ex_bg / data_hy_bg)

    if Complex == "Real":
        val = data.real
        val_bg = data_bg.real
    elif Complex == "Imag":
        val = data.imag
        val_bg = data_bg.imag
    elif Complex == "Amp":
        if Field == "Zxy":
            label = "Apparent Resistivity ($\Omega$m)"
            val = abs(data) ** 2 / (2 * frequency * np.pi * mu_0)
            val_bg = abs(data_bg) ** 2 / (2 * frequency_bg * np.pi * mu_0)
        else:
            val = data.amp()
            val_bg = data_bg.amp()
    elif Complex == "Phase":
        label = "Phase (degree)"
        val = data.pha()
        val_bg = data_bg.pha()

    plt.figure(figsize=(8 * 1.4, 3 * 1.4))
    ax = plt.subplot(111)
    if scale == "log":
        ax.plot(x, abs(val), "r")
        ax.plot(x, abs(val_bg), "k")
    elif scale == "linear":
        ax.plot(x, val, "r")
        ax.plot(x, val_bg, "k")
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(label)
    ax.set_yscale(scale)
    ax.legend(("Response", "Background"))
    if Complex == "Amp":
        if Field == "Zxy":
            title = "$\\rho_{xy}$"
        else:
            title = "|" + Field + Component + "|"
    else:
        if Field == "Zxy":
            title = Complex + "(" + Field + ")"
        else:
            title = Complex + "(" + Field + Component + ")"
    ax.set_title(title)
    if Fixed:
        ax.set_ylim(vmin, vmax)
    plt.show()


def DataApp():

    frequency = FloatText(value=0.5, description="f (Hz)")
    zsrc = FloatText(value=-950.0, description="src height (m)")
    zrx = FloatText(value=-1000.0, description="rx height (m)")
    rho0 = FloatText(value=1e8, description="$\\rho_{0} \ (\Omega m)$")
    rho1 = FloatText(value=0.3, description="$\\rho_{1} \ (\Omega m)$")
    rho2 = FloatText(value=1.0, description="$\\rho_{2} \ (\Omega m)$")
    rho3 = FloatText(value=100.0, description="$\\rho_{3} \ (\Omega m)$")
    rho4 = FloatText(value=1.0, description="$\\rho_{4} \ (\Omega m)$")
    rv_rh = FloatText(value=1.0, description="$\\rho_{2 \ v} / \\rho_{2 \ h}$")
    dz1 = FloatText(value=1000.0, description="dz1 (m)")
    dz2 = FloatText(value=1000.0, description="dz2 (m)")
    dz3 = FloatText(value=200.0, description="dz3 (m)")
    frequency_bg = FloatText(value=0.5, description="f$^{BG}$ (Hz)")
    zsrc_bg = FloatText(value=-950.0, description="src height$^{BG}$ (m)")
    zrx_bg = FloatText(value=-1000.0, description="rx height$^{BG}$ (m)")
    rho0_bg = FloatText(value=1e8, description="$\\rho_{0}^{BG} \ (\Omega m)$")
    rho1_bg = FloatText(value=0.3, description="$\\rho_{1}^{BG} \ (\Omega m)$")
    rho2_bg = FloatText(value=1.0, description="$\\rho_{2}^{BG} \ (\Omega m)$")
    rho3_bg = FloatText(value=1.0, description="$\\rho_{3}^{BG} \ (\Omega m)$")
    rho4_bg = FloatText(value=1.0, description="$\\rho_{4}^{BG} \ (\Omega m)$")
    rv_rh_bg = FloatText(
        value=1.0, description="$\\rho_{2 \ v}^{BG} / \\rho_{2 \ h}^{BG}$"
    )
    dz1_bg = FloatText(value=1000.0, description="dz1$^{BG}$ (m)")
    dz2_bg = FloatText(value=1000.0, description="dz2$^{BG}$ (m)")
    dz3_bg = FloatText(value=200.0, description="dz3$^{BG}$ (m)")
    Field = ToggleButtons(options=["E", "H", "Zxy"], value="E", description="Field")
    Plane = ToggleButtons(options=["XZ", "YZ"], value="XZ", description="Plane")
    Component = ToggleButtons(
        options=["x", "y", "z"], value="x", description="rx direction"
    )
    Complex = ToggleButtons(
        options=["Real", "Imag", "Amp", "Phase"], value="Amp", description="Complex"
    )
    scale = ToggleButtons(options=["log", "linear"], value="log", description="Scale")
    Fixed = widgets.widget_bool.Checkbox(value=False, description="Fixed")
    vmin = FloatText(value=None, description="vmin")
    vmax = FloatText(value=None, description="vmax")
    # __manual = True

    out = widgets.interactive_output(
        csem_data_app,
        {
            "frequency": frequency,
            "zsrc": zsrc,
            "zrx": zrx,
            "rho0": rho0,
            "rho1": rho1,
            "rho2": rho2,
            "rho3": rho3,
            "rho4": rho4,
            "rv_rh": rv_rh,
            "dz1": dz1,
            "dz2": dz2,
            "dz3": dz3,
            "frequency_bg": frequency_bg,
            "zsrc_bg": zsrc_bg,
            "zrx_bg": zrx_bg,
            "rho0_bg": rho0_bg,
            "rho1_bg": rho1_bg,
            "rho2_bg": rho2_bg,
            "rho3_bg": rho3_bg,
            "rho4_bg": rho4_bg,
            "rv_rh_bg": rv_rh_bg,
            "dz1_bg": dz1_bg,
            "dz2_bg": dz2_bg,
            "dz3_bg": dz3_bg,
            "Field": Field,
            "Plane": Plane,
            "Component": Component,
            "Complex": Complex,
            "scale": scale,
            "Fixed": Fixed,
            "vmin": vmin,
            "vmax": vmax,
        },
    )

    panel = VBox(
        [frequency, zsrc, zrx, rho0, rho1, rho2, rho3, rho4, rv_rh, dz1, dz2, dz3]
    )
    panel_bg = VBox(
        [
            frequency_bg,
            zsrc_bg,
            zrx_bg,
            rho0_bg,
            rho1_bg,
            rho2_bg,
            rho3_bg,
            rho4_bg,
            rv_rh_bg,
            dz1_bg,
            dz2_bg,
            dz3_bg,
        ]
    )
    panel_type = VBox([Field, Plane, Component, Complex, scale, Fixed, vmin, vmax])
    return VBox([HBox([panel, panel_bg]), panel_type, out])
