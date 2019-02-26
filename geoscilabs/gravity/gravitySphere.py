import numpy as np
import matplotlib.pyplot as plt
import matplotlib.contour as ctr
import scipy.io
from scipy.constants import G
import copy

from ipywidgets import (
    interactive,
    IntSlider,
    widget,
    FloatText,
    FloatSlider,
    ToggleButton,
    VBox,
    HBox,
    Output,
    Layout,
)

plotdata = []  # Storage the data that will print on the picture
colorList = [
    "red",
    "blue",
    "green",
    "orange",
    "black",
    "pink",
    "brown",
    "deepskyblue",
    "darkkhaki",
    "fuchsia",
    "midnightblue",
    "yellow",
    "gold",
    "lime",
]  # The Colors
index = 0
currentResult = []

# The main function and draw the first table
def drawfunction(delta_rho, a, z, stationSpacing, B):
    global plotdata
    global index
    global colorList
    global currentResult
    respEW, respNS, X, Y = datagenerator(delta_rho, a, z, stationSpacing)
    Dpi = 60
    plt.figure(figsize=(15, 20), dpi=Dpi)
    ax0 = plt.subplot2grid((20, 1), (1, 0), rowspan=4)
    ax2 = plt.subplot2grid((20, 1), (5, 0), rowspan=9)
    ax1 = plt.subplot2grid((20, 18), (14, 6), rowspan=4, colspan=6)
    textShow = []
    colors = []
    maxG = -100
    minG = 100
    if B:
        for each in plotdata:
            ax0.plot(each[0], each[1], "k.-", color=each[2])
            textShow.append(each[3])
            colors.append(each[2])
            if each[4] > maxG:
                maxG = each[4]
            elif each[4] < minG:
                minG = each[4]
    else:
        plotdata.clear()
        index = 0
        maxG = -100
        minG = 100
    ax0.plot(Y[:, 0], respNS, "k.-", color=colorList[index])
    currentResult = [Y[:, 0], respNS]
    showText = (
        r"$\Delta\rho$" + "=" + str(delta_rho) + ", a=" + str(a) + " ,z=" + str(z)
    )
    textShow.append(showText)
    colors.append(colorList[index])
    textLocation = max(max(respNS), maxG)
    maxG = max(max(respNS), maxG)
    minG = min(min(respNS), minG)
    textheight = 12 / 2.845 / 80 * (maxG - minG)
    for i in range(len(textShow)):
        ax0.text(
            -10,
            textLocation - i * textheight,
            textShow[i],
            color=colors[i],
            verticalalignment="top",
            fontsize=10,
        )
    ax0.grid(True)
    ax0.set_ylabel(r"$\Delta g_z$" + "(mGal)", fontsize=16)
    ax0.set_xlabel("x (m)", fontsize=16)
    printGrapha(delta_rho, a, z, ax1, stationSpacing)
    printCirCle(a, z, ax2)
    if B:
        plotdata.append(
            [
                copy.deepcopy(Y[:, 0]),
                copy.deepcopy(respNS),
                colorList[index],
                showText,
                textLocation,
            ]
        )
        index += 1
        if index > len(colorList) - 1:
            index = 0
    plt.tight_layout()
    plt.show()


# draw the figure of the Sphere
def printCirCle(a, z, axeToDraw):
    theta = np.linspace(0, 2 * np.pi, 800)
    x, y = np.cos(theta) * a, np.sin(theta) * a - z
    axeToDraw.text(a, -z, "The Sphere")
    axeToDraw.text(-0.6, 0.3, "Ground")
    axeToDraw.plot([-10, 10], [0, 0], color="black", linewidth=3.0)
    axeToDraw.plot([-5, 5], [-9, -9], color="white", linewidth=1.0)
    axeToDraw.plot([-5, 5], [4, 4], color="white", linewidth=1.0)
    axeToDraw.plot([-5, -5], [-9, 4], color="white", linewidth=1.0)
    axeToDraw.plot([5, 5], [4, -9], color="white", linewidth=1.0)
    axeToDraw.plot(x, y, color="blue", linewidth=2.0)
    axeToDraw.set_title("Position of the Sphere", fontsize=16)


# generate the data of table
def datagenerator(delta_rho, a, z, stationSpacing):
    respEW = 0
    gravity_change = []
    xmax = 10.0
    npts = int(1 / stationSpacing)
    x = np.linspace(-xmax, xmax, num=npts)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    for each in range(npts):
        gravity_change.append(calculategravity(delta_rho, a, z, x[each]))
    return respEW, gravity_change, X, Y


# get the value of the delta gravity
def calculategravity(delta_rho, a, z, x) -> float:
    partial = (4000 / 3 * np.pi * G) * delta_rho * a * a * a * z
    g = partial / float(np.power(x * x + z * z, 1.5))
    response = g * pow(10, 5)
    return response


# draw the third picture
def printGrapha(delta_rho, a, z, axeToDraw, stationSpacing):
    maxR = 5
    maxScala = 50
    axeToDraw.set_xlabel("X (m)", fontsize=16)
    axeToDraw.set_ylabel("Y (m)", fontsize=16)
    Step = pow(stationSpacing, 0.5)
    scalax, scalay, color = graphaDataGenerator(delta_rho, a, z, maxR, maxScala, Step)
    dat0 = axeToDraw.scatter(
        scalax, scalay, c=color, cmap="plasma", marker="s", s=450 * pow(Step, 0.5)
    )
    axeToDraw.set_title(r"$\Delta g_z$" + " (mGal)", fontsize=16)
    plt.colorbar(dat0, ax=axeToDraw)


# get the data of the third picture
def graphaDataGenerator(delta_rho, a, z, maxR, maxScala, Step):
    scalax = []
    scalay = []
    color = []
    for i in np.arange(-maxR, maxR + Step, Step):
        for j in np.arange(-maxR, maxR + Step, Step):
            scalax.append(i)
            scalay.append(j)
            color.append(
                calculategravity(delta_rho, a, z, pow(pow(i, 2) + pow(j, 2), 0.5))
            )
    return scalax, scalay, color


# draw the widgets
def interact_gravity_sphere():
    Q = interactive(
        drawfunction,
        delta_rho=FloatSlider(
            description=r"$\Delta\rho$",
            min=-5.0,
            max=5.0,
            step=0.1,
            value=1.0,
            continuous_update=False,
        ),
        a=FloatSlider(min=0.1, max=4.0, step=0.1, value=1.0, continuous_update=False),
        z=FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0, continuous_update=False),
        stationSpacing=FloatSlider(
            description="Step",
            min=0.005,
            max=0.1,
            step=0.005,
            value=0.01,
            continuous_update=False,
            readout_format=".3f",
        ),
        B=ToggleButton(
            value=True,
            description="keep previous plots",
            disabled=False,
            button_style="",
            tooltip="Click me",
            layout=Layout(width="20%"),
        ),
    )
    return Q


# Print the result of the last data
def printResult():
    global currentResult
    print("{0: ^10}{1: ^10}".format("X", "Δgz"))
    for i in range(len(currentResult[0])):
        print(
            "{0: ^10}{1: ^10}".format(
                "%.3f" % currentResult[0][i], "%.6f" % currentResult[1][i]
            )
        )
