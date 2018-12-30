import numpy as np
from scipy.constants import mu_0, epsilon_0
import matplotlib.pyplot as plt
from PIL import Image
import warnings
from ipywidgets import (
    interact,
    interactive,
    IntSlider,
    widget,
    FloatText,
    FloatSlider,
    fixed,
)

from ..base import wiggle
from .Wiggle import PrimaryWave, ReflectedWave

import requests
from io import BytesIO


########################################
#           DOWNLOAD FUNCTIONS
########################################


def downloadRadargramImage(URL):

    urlObj = requests.get(URL)
    imgcmp = Image.open(BytesIO(urlObj.content))

    return imgcmp


########################################
#           WIDGETS
########################################


def PrimaryWidget(dataFile, timeFile):

    i = interact(
        PrimaryWidgetFcn,
        epsrL=(1, 10, 1),
        epsrH=(1, 20, 1),
        tinterpL=(0, 150, 2),
        tinterpH=(0, 150, 2),
        dFile=fixed(dataFile),
        tFile=fixed(timeFile),
    )

    return i


def PrimaryFieldWidget(radargramImage):

    i = interact(
        PrimaryFieldWidgetFcn,
        tinterp=(0, 80, 2),
        epsr=(1, 40, 1),
        radgramImg=fixed(radargramImage),
    )

    return i


def PipeWidget(radargramImage):

    i = interact(
        PipeWidgetFcn,
        epsr=(0, 100, 1),
        h=(0.1, 2.0, 0.1),
        xc=(0.0, 40.0, 0.2),
        r=(0.1, 3, 0.1),
        imgcmp=fixed(radargramImage),
    )

    return i


def WallWidget(radargramImagePath):

    i = interact(
        WallWidgetFcn,
        epsr=(0, 100, 1),
        h=(0.1, 2.0, 0.1),
        x1=(1, 35, 1),
        x2=(20, 40, 1),
        imgcmp=fixed(radargramImagePath),
    )

    return i


########################################
#           FUNCTIONS
########################################


def PrimaryWidgetFcn(tinterpL, epsrL, tinterpH, epsrH, dFile, tFile):
    data = np.load(dFile)
    time = np.load(tFile)
    dt = time[1] - time[0]
    v1 = 1.0 / np.sqrt(epsilon_0 * epsrL * mu_0)
    v2 = 1.0 / np.sqrt(epsilon_0 * epsrH * mu_0)
    dx = 0.3
    nano = 1e9
    xorig = np.arange(data.shape[0]) * dx
    out1 = PrimaryWave(xorig, v1, tinterpL / nano)
    out2 = ReflectedWave(xorig, v2, tinterpH / nano)

    kwargs = {"skipt": 1, "scale": 0.5, "lwidth": 0.1, "dx": dx, "sampr": dt * nano}

    extent = [0.0, 30, 300, 0]
    _, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    ax1.invert_yaxis()
    ax1.axis(extent)
    ax1.set_xlabel("Offset (m)")
    ax1.set_ylabel("Time (ns)")
    ax1.set_title("Shot Gather")
    wiggle(data, ax=ax1, **kwargs)
    ax1.plot(xorig, out1 * nano, "b", lw=2)
    ax1.plot(xorig, out2 * nano, "r", lw=2)

    plt.show()


def PrimaryFieldWidgetFcn(tinterp, epsr, radgramImg):
    imgcmp = Image.open(radgramImg)
    plt.figure(figsize=(6, 7))
    plt.imshow(imgcmp, extent=[0, 150, 150, 0])
    x = np.arange(81) * 0.1
    xconvert = x * 150.0 / 8.0
    v = 1.0 / np.sqrt(mu_0 * epsilon_0 * epsr)
    nano = 1e9
    # tinterp = 30
    y = (1.0 / v * x) * nano + tinterp
    plt.plot(xconvert, y, lw=2)
    plt.xticks(
        np.arange(11) * 15, np.arange(11) * 0.8 + 2.4
    )  # +2.4 for offset correction
    plt.xlim(0.0, 150.0)
    plt.ylim(146.0, 0.0)
    plt.ylabel("Time (ns)")
    plt.xlabel("Offset (m)")

    plt.show()


def PipeWidgetFcn(epsr, h, xc, r, imgcmp):

    # imgcmp = Image.open(dataImage)
    imgcmp = imgcmp.resize((600, 800))
    plt.figure(figsize=(9, 11))

    plt.imshow(imgcmp, extent=[0, 400, 250, 0])
    x = np.arange(41) * 1.0
    xconvert = x * 10.0
    v = 1.0 / np.sqrt(mu_0 * epsilon_0 * epsr)
    nano = 1e9
    time = (np.sqrt(((x - xc) ** 2 + 4 * h ** 2)) - r) / v
    plt.plot(xconvert, time * nano, "r--", lw=2)
    plt.xticks(np.arange(11) * 40, np.arange(11) * 4.0)
    plt.xlim(0.0, 400)
    plt.ylim(240.0, 0.0)
    plt.ylabel("Time (ns)")
    plt.xlabel("Survey line location (m)")

    plt.show()


def WallWidgetFcn(epsr, h, x1, x2, imgcmp):

    # imgcmp = Image.open(dataImage)
    imgcmp = imgcmp.resize((600, 800))
    plt.figure(figsize=(9, 11))

    plt.imshow(imgcmp, extent=[0, 400, 250, 0])
    x = np.arange(41) * 1.0
    ind1 = x <= x1
    ind2 = x >= x2
    scale = 10.0
    xconvert = x * scale
    v = 1.0 / np.sqrt(mu_0 * epsilon_0 * epsr)
    nano = 1e9

    def arrival(x, xc, h, v):
        return (np.sqrt(((x - xc) ** 2 + 4 * h ** 2))) / v

    plt.plot(xconvert[ind1], arrival(x[ind1], x1, h, v) * nano, "b--", lw=2)
    plt.plot(xconvert[ind2], arrival(x[ind2], x2, h, v) * nano, "b--", lw=2)
    plt.plot(
        np.r_[x1 * scale, x2 * scale],
        np.r_[2.0 * h / v, 2.0 * h / v] * nano,
        "b--",
        lw=2,
    )

    plt.xticks(np.arange(11) * 40, np.arange(11) * 4.0)
    plt.xlim(0.0, 400)
    plt.ylim(240.0, 0.0)
    plt.ylabel("Time (ns)")
    plt.xlabel("Survey line location (m)")

    plt.show()
