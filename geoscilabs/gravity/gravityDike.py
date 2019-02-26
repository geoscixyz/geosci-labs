import numpy as np
import matplotlib.pyplot as plt
import matplotlib.contour as ctr
import scipy.io
import copy
from math import pi, tan, cos, acos, log, sin
from scipy.constants import G
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
    interactive_output,
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
def drawfunction(delta_rho, z1, z2, b, beta, stationSpacing, B):
    global plotdata
    global index
    global colorList
    global currentResult
    beta = beta * pi / 180
    respEW, respNS, X, Y = datagenerator(delta_rho, z1, z2, b, beta, stationSpacing)
    Dpi = 60
    plt.figure(figsize=(10, 22), dpi=Dpi)
    ax0 = plt.subplot2grid((22, 1), (1, 0), rowspan=6)
    ax2 = plt.subplot2grid((22, 1), (7, 0), rowspan=6)
    ax1 = plt.subplot2grid((22, 4), (13, 1), rowspan=4, colspan=2)
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
        r"$\Delta\rho$"
        + "="
        + str(delta_rho)
        + ", z1=%.2f" % z1
        + " ,z2=%.2f" % z2
        + " ,b=%d ," % b
        + r"$\beta=$%d" % (beta * 180 / pi)
        + " ,Step=%.3f" % stationSpacing
    )
    textShow.append(showText)
    colors.append(colorList[index])
    textLocation = max(max(respNS), maxG)
    minG = min(min(respNS), minG)
    maxG = max(max(respNS), maxG)
    textheight = 12 / 2.845 / 50 * (maxG - minG)
    for i in range(len(textShow)):
        ax0.text(
            -6,
            textLocation - i * textheight,
            textShow[i],
            color=colors[i],
            verticalalignment="top",
            fontsize=10,
        )
    ax0.grid(True)
    ax0.set_ylabel(r"$\Delta g_z$" + "(mGal)", fontsize=16)
    ax0.set_xlabel("x (m)", fontsize=16)
    printGrapha(delta_rho, z1, z2, b, beta, ax1, stationSpacing)
    printDike(ax2, z1, z2, b, beta, 5)
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


# draw the figure of the Dike
def printDike(axeToDraw, z1, z2, b, beta, x):
    axeToDraw.plot(0, 0, ".")
    axeToDraw.plot([-6, 6], [0, 0], color="black", linewidth=1.5)
    x1 = -z1 * float(np.tan(beta))
    x3 = b - z1 * float(np.tan(beta))
    x2 = -z2 * float(np.tan(beta))
    x4 = b - z2 * float(np.tan(beta))
    x1 = min(6, abs(x1)) * getSigned(x1)
    x2 = min(6, abs(x2)) * getSigned(x2)
    x3 = min(6, abs(x3)) * getSigned(x3)
    x4 = min(6, abs(x4)) * getSigned(x4)
    axeToDraw.plot([x1, x3], [-z1, -z1], color="black", linewidth=1.0)
    axeToDraw.plot([x2, x4], [-z2, -z2], color="black", linewidth=1.0)
    if beta != 0:
        axeToDraw.plot(
            [x1, x2],
            [x1 / np.tan(beta), x2 / np.tan(beta)],
            color="black",
            linewidth=1.0,
        )
        axeToDraw.plot(
            [x3, x4],
            [(x3 - b) / np.tan(beta), (x4 - b) / np.tan(beta)],
            color="black",
            linewidth=1.0,
        )
    else:
        axeToDraw.plot([x1, x2], [-z1, -z2], color="black", linewidth=1.0)
        axeToDraw.plot([x3, x4], [-z1, -z2], color="black", linewidth=1.0)
    axeToDraw.plot(
        [0, -z1 * float(np.tan(beta))], [0, -z1], ":", color="black", linewidth=1.0
    )
    axeToDraw.plot([-6, 6], [-6, -6], color="white", linewidth=1.0)
    axeToDraw.plot([-6, 6], [1, 1], color="white", linewidth=1.0)
    axeToDraw.plot([-6, -6], [-6, 1], color="white", linewidth=1.0)
    axeToDraw.plot([6, 6], [1, -6], color="white", linewidth=1.0)
    axeToDraw.set_title("Position of Block", fontsize=16)


# Get the sign of the input
def getSigned(a):
    if a > 0:
        return 1
    else:
        return -1


# generate the data of table
def datagenerator(delta_rho, z1, z2, b, beta, stationSpacing):
    respEW = 0
    gravity_change = []
    xmax = 6.0
    npts = int(1 / stationSpacing)
    x = np.linspace(-xmax, xmax, num=npts)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    for each in range(npts):
        gravity_change.append(calculategravity(delta_rho, z1, z2, b, beta, x[each]))
    return respEW, gravity_change, X, Y


# get the value of the delta gravity
def calculategravity(delta_rho, z1, z2, b, beta, x) -> float:
    r1 = pow(pow(x + z1 * tan(beta), 2) + pow(z1, 2), 0.5)
    r2 = pow(pow(x + z2 * tan(beta), 2) + pow(z2, 2), 0.5)
    r3 = pow(pow(x + z1 * tan(beta) - b, 2) + pow(z1, 2), 0.5)
    r4 = pow(pow(x + z2 * tan(beta) - b, 2) + pow(z2, 2), 0.5)
    theta1 = acos(z1 / r1)
    theta2 = acos(z2 / r2)
    theta3 = acos(z1 / r3)
    theta4 = acos(z2 / r4)
    if x < (-z2 * tan(beta)):
        theta2 = -theta2
    if x < (-z1 * tan(beta)):
        theta1 = -theta1
    if x < (b - z2 * tan(beta)):
        theta4 = -theta4
    if x < (b - z1 * tan(beta)):
        theta3 = -theta3
    part1 = z2 * (theta2 - theta4) - z1 * (theta1 - theta3)
    part2 = (
        sin(beta) * cos(beta) * (x * (theta2 - theta1) - (x - b) * (theta4 - theta3))
    )
    part3 = pow(cos(beta), 2) * (x * log(r2 / r1) - (x - b) * log(r4 / r3))
    g = 2000 * G * delta_rho * (part1 + part2 + part3)
    response = g * pow(10, 5)
    return response


# draw the third picture
def printGrapha(delta_rho, z1, z2, b, beta, axeToDraw, stationSpacing):
    maxR = 5
    maxScala = 50
    axeToDraw.set_xlabel("X (m)", fontsize=16)
    axeToDraw.set_ylabel("Y (m)", fontsize=16)
    Step = pow(stationSpacing, 0.5)
    scalax, scalay, color = graphaDataGenerator(
        delta_rho, z1, z2, b, beta, maxR, maxScala, Step
    )
    dat0 = axeToDraw.scatter(
        scalax, scalay, c=color, cmap="plasma", marker="s", s=450 * pow(Step, 0.5)
    )
    axeToDraw.set_title(r"$\Delta g_z$" + "(mGal)", fontsize=16)
    plt.colorbar(dat0, ax=axeToDraw)


# get the data of the third picture
def graphaDataGenerator(delta_rho, z1, z2, b, beta, maxR, maxScala, Step):
    scalax = []
    scalay = []
    color = []
    for i in np.arange(-maxR, maxR + Step, Step):
        g = calculategravity(delta_rho, z1, z2, b, beta, i)
        for j in np.arange(-maxR, maxR + Step, Step):
            scalax.append(i)
            scalay.append(j)
            color.append(g)
    return scalax, scalay, color


# draw the widgets
def interact_gravity_Dike():
    s1 = FloatSlider(
        description=r"$\Delta\rho$",
        min=-5.0,
        max=5.0,
        step=0.1,
        value=1.0,
        continuous_update=False,
    )
    s2 = FloatSlider(
        description=r"$z_1$",
        min=0.1,
        max=4.0,
        step=0.1,
        value=1 / 3,
        continuous_update=False,
    )
    s3 = FloatSlider(
        description=r"$z_2$",
        min=0.1,
        max=5.0,
        step=0.1,
        value=4 / 3,
        continuous_update=False,
    )
    s4 = FloatSlider(
        description="b", min=0.1, max=5.0, step=0.1, value=1.0, continuous_update=False
    )
    s5 = FloatSlider(
        description=r"$\beta$",
        min=-85,
        max=85,
        step=5,
        value=45,
        continuous_update=False,
    )
    s6 = FloatSlider(
        description="Step",
        min=0.005,
        max=0.10,
        step=0.005,
        value=0.01,
        continuous_update=False,
        readout_format=".3f",
    )
    b1 = ToggleButton(
        value=True,
        description="keep previous plots",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip="Click me",
        layout=Layout(width="20%"),
    )
    v1 = VBox([s1, s2, s3])
    v2 = VBox([s4, s5, s6])
    out1 = HBox([v1, v2, b1])
    out = interactive_output(
        drawfunction,
        {
            "delta_rho": s1,
            "z1": s2,
            "z2": s3,
            "b": s4,
            "beta": s5,
            "stationSpacing": s6,
            "B": b1,
        },
    )
    return VBox([out1, out])


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
