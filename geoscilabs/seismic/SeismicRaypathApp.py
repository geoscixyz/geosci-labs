import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from .SeismicRefraction import (
    viewTXdiagram,
    direct,
    reflection1,
    refraction1,
    refraction2,
)
from ipywidgets import widgets
import matplotlib

matplotlib.rcParams["font.size"] = 10


def direct_time_to_space(t, v1):
    """
    direct ray
    """
    return t * v1


def reflection1_time_to_space(t, v1, z1):
    t0 = 2.0 * z1 / v1
    x = np.sqrt((t ** 2 - t0 ** 2) * v1 ** 2)

    return x


def refraction1_time_to_space(t, v1, v2, z1):
    """
    refraction off of first interface
    """

    theta1 = np.arcsin(v1 / v2)
    ti1 = 2 * z1 * np.cos(theta1) / v1
    x = (t - ti1) * v2
    return x


def refraction2_time_to_space(t, v1, v2, v3, z1, z2):
    theta1 = np.arcsin(v1 / v3)
    theta2 = np.arcsin(v2 / v3)
    ti1 = 2 * z1 * np.cos(theta1) / v1
    ti2 = 2 * z2 * np.cos(theta2) / v2
    x = (t - (ti2 + ti1)) * v3
    return x


def refraction1_space_to_time(x, v1, v2, z1):
    """
    refraction off of first interface
    """

    theta1 = np.arcsin(v1 / v2)
    ti1 = 2 * z1 * np.cos(theta1) / v1
    ref1 = 1.0 / v2 * x + ti1
    return ref1


def refraction2_space_to_time(x, v1, v2, v3, z1, z2):
    theta1 = np.arcsin(v1 / v3)
    theta2 = np.arcsin(v2 / v3)
    ti1 = 2 * z1 * np.cos(theta1) / v1
    ti2 = 2 * z2 * np.cos(theta2) / v2
    ref2 = 1.0 / v3 * x + ti2 + ti1
    return ref2


def reflection1_space_to_time(x, v1, z1):
    t0 = 2.0 * z1 / v1
    refl1 = np.sqrt(t0 ** 2 + x ** 2 / v1 ** 2)
    return refl1


def direct_path(x_loc):
    x, y = [0, x_loc], [0, 0]
    return x, y


def reflection_path_1(x_loc, z1):
    x = [0, x_loc / 2.0, x_loc]
    y = [0, z1, 0]
    return x, y


def refraction_path_1(x_loc, v1, v2, z1):
    theta = np.arcsin(v1 / v2)
    d = np.tan(theta) * z1
    l = x_loc - 2 * d
    if l < 0:
        return None, None
    x = [0, d, d + l, x_loc]
    y = [0, z1, z1, 0]
    return x, y


def refraction_path_2(x_loc, v1, v2, v3, z1, z2):
    theta1 = np.arcsin(v1 / v3)
    theta2 = np.arcsin(v2 / v3)
    d1 = np.tan(theta1) * z1
    d2 = np.tan(theta2) * z2
    l = x_loc - 2 * (d1 + d2)
    if l < 0:
        return None, None
    x = [0, d1, d1 + d2, d1 + d2 + l, d1 + d2 + l + d2, x_loc]
    y = [0, z1, z2 + z1, z2 + z1, z1, 0]
    return x, y


def interact_refraction(v1, v2, v3, z1, z2, x_loc, t_star, show):
    z3 = 50
    plt.figure(figsize=(8 * 0.9, 10 * 0.9))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    xlim = [0, 100]

    x0 = 0.0
    x_int = x0 + np.arange(1000) * 0.1
    ax1.plot(xlim, [0, 0], "k-", lw=1, alpha=0.3)
    ax1.plot(xlim, [z1, z1], "k-", lw=1, alpha=0.3)
    ax1.plot(xlim, [z2 + z1, z2 + z1], "k-", lw=1, alpha=0.3)

    if (show == "direct") or (show == "all"):
        # direct
        x, y = direct_path(x_loc)
        f_direct = interp1d(x, y, bounds_error=False)
        ax1.plot(x, y, "r--", lw=1)

        x_star = direct_time_to_space(t_star, v1)
        y_direct = f_direct(x_int[x_int <= x_star])
        ax1.plot(x_int[x_int <= x_star], y_direct, "r-", lw=3)
        ax2.plot(x_int, direct(x_int, v1), "-r", linewidth=2.0)

    if (show == "reflection") or (show == "all"):
        # reflection
        x, y = reflection_path_1(x_loc, z1)
        ax1.plot(x, y, "k--", lw=1)
        f_reflection1 = interp1d(x, y, bounds_error=False)
        x_star = reflection1_time_to_space(t_star, v1, z1)
        y_reflection1 = f_reflection1(x_int[x_int <= x_star])
        ax1.plot(x_int[x_int <= x_star], y_reflection1, "k-", lw=3)
        ax2.plot(x_int, reflection1(x_int, v1, z1), "-k", linewidth=2.0)

    if (show == "refraction1") or (show == "all"):
        if v1 < v2:
            # refraction 1
            x, y = refraction_path_1(x_loc, v1, v2, z1)
            if y is not None:
                f_refraction1 = interp1d(x, y, bounds_error=False)
                x_star = refraction1_time_to_space(t_star, v1, v2, z1)
                y_refraction1 = f_refraction1(x_int[x_int <= x_star])
                ax1.plot(x_int[x_int <= x_star], y_refraction1, "b-", lw=3)
                ax1.plot(x, y, "b--", lw=1)
            ax2.plot(x_int, refraction1(x_int, v1, v2, z1), "-b", linewidth=2.0)
            ax2.plot(
                x_int, refraction1_space_to_time(x_int, v1, v2, z1), "--b", linewidth=1
            )

    if (show == "refraction2") or (show == "all"):
        if v2 < v3:
            # refraction 2
            x, y = refraction_path_2(x_loc, v1, v2, v3, z1, z2)
            if y is not None:
                f_refraction2 = interp1d(x, y, bounds_error=False)
                ax1.plot(x, y, "g--", lw=1)
                x_star = refraction2_time_to_space(t_star, v1, v2, v3, z1, z2)
                y_refraction2 = f_refraction2(x_int[x_int <= x_star])
                ax1.plot(x_int[x_int <= x_star], y_refraction2, "g-", lw=3)
            ax2.plot(x_int, refraction2(x_int, v1, v2, v3, z1, z2), "-g", linewidth=2.0)
            ax2.plot(
                x_int,
                refraction2_space_to_time(x_int, v1, v2, v3, z1, z2),
                "--g",
                linewidth=1,
            )

    ax1.plot(np.r_[x_loc, x_loc], np.r_[z3, 0], "k--o", lw=1, alpha=0.5)
    ax2.plot(np.r_[x_loc, x_loc], np.r_[0, 0.25], "k--o", lw=1, alpha=0.5)
    ax2.plot(np.r_[0, 100], np.r_[t_star, t_star], "k--o", lw=1, alpha=0.5)
    ax2.text(100 + 1, t_star + 0.003, ("%.3fs") % (t_star))
    ax2.text(x_loc - 3, 0.25 + 0.003, ("%.2fm") % (x_loc))
    ax1.text(100 + 1, z1 + 1, ("%.2fm") % (z1))
    ax1.text(100 + 1, z2 + z1 + 1, ("%.2fm") % (z2 + z1))
    ax1.plot(100, z1, "ko")
    ax1.plot(100, z2 + z1, "ko")

    #     if legend:
    #         ax2.legend(['direct', 'refraction1', 'refraction2'], loc='best')
    majorxtick = np.arange(0.0, 131.0, 20)
    minorxtick = np.arange(0.0, 131, 5.0)
    majorytick = np.arange(0.0, 0.26, 0.05)
    minorytick = np.arange(0.0, 0.26, 0.01)
    ax2.set_xticks(majorxtick)
    ax2.set_xticks(minorxtick, minor=True)

    ax1.set_xticks(majorxtick)
    ax1.set_xticks(minorxtick, minor=True)
    ax1.set_xlim(xlim)
    ax1.set_ylim(z3, -5.0)
    ax1.set_aspect(1)
    ax1.set_ylabel("Depth (m)")

    ax2.set_yticks(majorytick)
    ax2.set_yticks(minorytick, minor=True)
    ax2.set_xlim(0.0, 130.0)
    ax2.set_ylim(0.0, 0.25)
    # ax2.invert_yaxis()
    ax2.set_xlabel("Offset (m)")
    ax2.set_ylabel("Time (s)")
    ax2.set_xlim(0, 100)


def seismic_app():
    v1 = widgets.FloatSlider(
        description="v1", min=300, max=2000, step=1, continuous_update=False, value=400
    )
    v2 = widgets.FloatSlider(
        description="v2", min=300, max=5000, step=1, continuous_update=False, value=1000
    )
    v3 = widgets.FloatSlider(
        description="v3", min=300, max=5000, step=1, continuous_update=False, value=1500
    )
    z1 = widgets.FloatSlider(
        description="z1", min=5, max=50, step=1, continuous_update=False, value=5
    )
    z2 = widgets.FloatSlider(
        description="z2", min=5, max=50, step=1, continuous_update=False, value=10
    )
    x_loc = widgets.FloatSlider(
        description="offset",
        min=1,
        max=100,
        step=0.5,
        continuous_update=False,
        value=80,
    )
    t_star = widgets.FloatSlider(
        description="time",
        min=0.001,
        max=0.25,
        step=0.001,
        continuous_update=False,
        value=0.1,
        orientation="vertical",
        readout_format=".3f",
    )
    show = widgets.RadioButtons(
        description="plot",
        options=["direct", "reflection", "refraction1", "refraction2", "all"],
        value="all",
        disabled=False,
    )
    out = widgets.interactive_output(
        interact_refraction,
        {
            "v1": v1,
            "v2": v2,
            "v3": v3,
            "z1": z1,
            "z2": z2,
            "x_loc": x_loc,
            "t_star": t_star,
            "show": show,
        },
    )
    left = widgets.VBox(
        [t_star],
        layout=widgets.Layout(width="10%", height="400px", margin="300px 0px 0px 0px"),
    )
    right = widgets.VBox(
        [show, v1, v2, v3, z1, z2],
        layout=widgets.Layout(width="50%", height="400px", margin="20px 0px 0px 0px"),
    )
    image = widgets.VBox(
        [out, x_loc],
        layout=widgets.Layout(width="70%", height="600px", margin="0px 0px 0px 0px"),
    )
    return widgets.HBox([left, image, right])
