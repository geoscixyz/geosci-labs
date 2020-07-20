from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from ipywidgets import widgets
from discretize import CylMesh, TensorMesh
from SimPEG import maps, utils
from SimPEG.electromagnetics import frequency_domain as fdem
from pymatsolver import Pardiso

# from pymatsolver import PardisoSolver
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.constants import mu_0
import requests
from io import StringIO

from ..base import widgetify
from .DipoleWidgetFD import DisPosNegvalues
from .BiotSavart import BiotSavartFun


class HarmonicVMDCylWidget(object):
    """FDEMCylWidgete"""

    survey = None
    srcList = None
    mesh = None
    f = None
    activeCC = None
    srcLoc = None
    mesh2D = None
    mu = None

    def __init__(self):
        self.genMesh()
        self.getCoreDomain()
        # url = "http://em.geosci.xyz/_images/disc_dipole.png"
        # response = requests.get(url)
        # self.im = Image.open(StringIO(response.content))

    def mirrorArray(self, x, direction="x"):
        X = x.reshape((self.nx_core, self.ny_core), order="F")
        if direction == "x" or direction == "y":
            X2 = np.vstack((-np.flipud(X), X))
        else:
            X2 = np.vstack((np.flipud(X), X))
        return X2

    def genMesh(self, h=0.0, cs=3.0, ncx=15, ncz=30, npad=20):
        """
            Generate cylindrically symmetric mesh
        """
        # TODO: Make it adaptive due to z location
        hx = [(cs, ncx), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
        self.mesh = CylMesh([hx, 1, hz], "00C")

    def getCoreDomain(self, mirror=False, xmax=100, zmin=-100, zmax=100.0):

        self.activeCC = (self.mesh.gridCC[:, 0] <= xmax) & (
            np.logical_and(
                self.mesh.gridCC[:, 2] >= zmin, self.mesh.gridCC[:, 2] <= zmax
            )
        )
        self.gridCCactive = self.mesh.gridCC[self.activeCC, :][:, [0, 2]]

        xind = self.mesh.vectorCCx <= xmax
        yind = np.logical_and(self.mesh.vectorCCz >= zmin, self.mesh.vectorCCz <= zmax)
        self.nx_core = xind.sum()
        self.ny_core = yind.sum()

        # if self.mesh2D is None:
        hx = np.r_[self.mesh.hx[xind][::-1], self.mesh.hx[xind]]
        hz = self.mesh.hz[yind]
        self.mesh2D = TensorMesh([hx, hz], x0="CC")

    def getCoreModel(self, Type):

        if Type == "Layer":
            active = self.mesh2D.vectorCCy < self.z0
            ind1 = (self.mesh2D.vectorCCy < self.z0) & (
                self.mesh2D.vectorCCy >= self.z1
            )
            ind2 = (self.mesh2D.vectorCCy < self.z1) & (
                self.mesh2D.vectorCCy >= self.z2
            )
            mapping2D = maps.SurjectVertical1D(self.mesh2D) * maps.InjectActiveCells(
                self.mesh2D, active, self.sig0, nC=self.mesh2D.nCy
            )
            model2D = np.ones(self.mesh2D.nCy) * self.sig3
            model2D[ind1] = self.sig1
            model2D[ind2] = self.sig2
            model2D = model2D[active]
        elif Type == "Sphere":
            active = self.mesh2D.gridCC[:, 1] < self.z0
            ind1 = (self.mesh2D.gridCC[:, 1] < self.z1) & (
                self.mesh2D.gridCC[:, 1] >= self.z1 - self.h
            )
            ind2 = (
                np.sqrt(
                    (self.mesh2D.gridCC[:, 0]) ** 2
                    + (self.mesh2D.gridCC[:, 1] - self.z2) ** 2
                )
                <= self.R
            )

            mapping2D = maps.InjectActiveCells(
                self.mesh2D, active, self.sig0, nC=self.mesh2D.nC
            )
            model2D = np.ones(self.mesh2D.nC) * self.sigb
            model2D[ind1] = self.sig1
            model2D[ind2] = self.sig2
            model2D = model2D[active]

        return model2D, mapping2D

    def getBiotSavrt(self, rxLoc):
        """
            Compute Biot-Savart operator: Gz and Gx
        """
        self.Gz = BiotSavartFun(self.mesh, rxLoc, component="z")
        self.Gx = BiotSavartFun(self.mesh, rxLoc, component="x")

    def setThreeLayerParam(
        self, h1=12, h2=12, sig0=1e-8, sig1=1e-1, sig2=1e-2, sig3=1e-2, chi=0.0
    ):
        self.h1 = h1  # 1st layer thickness
        self.h2 = h2  # 2nd layer thickness
        self.z0 = 0.0
        self.z1 = self.z0 - h1
        self.z2 = self.z0 - h1 - h2
        self.sig0 = sig0  # 0th layer \sigma (assumed to be air)
        self.sig1 = sig1  # 1st layer \sigma
        self.sig2 = sig2  # 2nd layer \sigma
        self.sig3 = sig3  # 3rd layer \sigma

        active = self.mesh.vectorCCz < self.z0
        ind1 = (self.mesh.vectorCCz < self.z0) & (self.mesh.vectorCCz >= self.z1)
        ind2 = (self.mesh.vectorCCz < self.z1) & (self.mesh.vectorCCz >= self.z2)
        self.mapping = maps.SurjectVertical1D(self.mesh) * maps.InjectActiveCells(
            self.mesh, active, sig0, nC=self.mesh.nCz
        )
        model = np.ones(self.mesh.nCz) * sig3
        model[ind1] = sig1
        model[ind2] = sig2
        self.m = model[active]
        self.mu = np.ones(self.mesh.nC) * mu_0
        self.mu[self.mesh.gridCC[:, 2] < 0.0] = (1.0 + chi) * mu_0
        return self.m

    def setLayerSphereParam(
        self, d1=6, h=6, d2=16, R=4, sig0=1e-8, sigb=1e-2, sig1=1e-1, sig2=1.0, chi=0.0
    ):
        self.z0 = 0.0  # Surface elevation
        self.z1 = self.z0 - d1  # Depth to layer
        self.h = h  # Thickness of layer
        self.z2 = self.z0 - d2  # Depth to center of sphere
        self.R = R  # Radius of sphere
        self.sig0 = sig0  # Air conductivity
        self.sigb = sigb  # Background conductivity
        self.sig1 = sig1  # Layer conductivity
        self.sig2 = sig2  # Sphere conductivity

        active = self.mesh.gridCC[:, 2] < self.z0
        ind1 = (self.mesh.gridCC[:, 2] < self.z1) & (
            self.mesh.gridCC[:, 2] >= self.z1 - self.h
        )
        ind2 = (
            np.sqrt(
                (self.mesh.gridCC[:, 0]) ** 2 + (self.mesh.gridCC[:, 2] - self.z2) ** 2
            )
            <= self.R
        )

        self.mapping = maps.InjectActiveCells(self.mesh, active, sig0, nC=self.mesh.nC)
        model = np.ones(self.mesh.nC) * sigb
        model[ind1] = sig1
        model[ind2] = sig2
        self.m = model[active]
        self.mu = np.ones(self.mesh.nC) * mu_0
        self.mu[self.mesh.gridCC[:, 2] < 0.0] = (1.0 + chi) * mu_0
        return self.m

    def simulate(self, srcLoc, rxLoc, freqs):
        bzr = fdem.receivers.PointMagneticFluxDensitySecondary(
            rxLoc, orientation="z", component="real"
        )
        bzi = fdem.receivers.PointMagneticFluxDensitySecondary(
            rxLoc, orientation="z", component="imag"
        )
        self.srcList = [
            fdem.sources.MagDipole([bzr, bzi], freq, srcLoc, orientation="Z")
            for freq in freqs
        ]

        survey = fdem.survey.Survey(self.srcList)
        sim = fdem.Simulation3DMagneticFluxDensity(
            self.mesh, survey=survey, sigmaMap=self.mapping, mu=self.mu, solver=Pardiso
        )

        self.f = sim.fields(self.m)
        self.sim = sim
        dpred = sim.dpred(self.m, f=self.f)
        self.srcLoc = srcLoc
        self.rxLoc = rxLoc
        return dpred

    def getFields(self, bType="b", ifreq=0):
        src = self.srcList[ifreq]
        Pfx = self.mesh.getInterpolationMat(
            self.mesh.gridCC[self.activeCC, :], locType="Fx"
        )
        Pfz = self.mesh.getInterpolationMat(
            self.mesh.gridCC[self.activeCC, :], locType="Fz"
        )
        Ey = self.mesh.aveE2CC * self.f[src, "e"]
        Jy = utils.sdiag(self.sim.sigma) * Ey

        self.Ey = utils.mkvc(self.mirrorArray(Ey[self.activeCC], direction="y"))
        self.Jy = utils.mkvc(self.mirrorArray(Jy[self.activeCC], direction="y"))
        self.Bx = utils.mkvc(self.mirrorArray(Pfx * self.f[src, bType], direction="x"))
        self.Bz = utils.mkvc(self.mirrorArray(Pfz * self.f[src, bType], direction="z"))

    def getData(self, bType="b"):

        Pfx = self.mesh.getInterpolationMat(self.rxLoc, locType="Fx")
        Pfz = self.mesh.getInterpolationMat(self.rxLoc, locType="Fz")
        Pey = self.mesh.getInterpolationMat(self.rxLoc, locType="Ey")

        self.Ey = (Pey * self.f[:, "e"]).flatten()
        self.Bx = (Pfx * self.f[:, bType]).flatten()
        self.Bz = (Pfz * self.f[:, bType]).flatten()

    def plotField(
        self,
        Field="B",
        ComplexNumber="real",
        view="vec",
        scale="linear",
        Frequency=100,
        Geometry=True,
        Scenario=None,
    ):
        # Printout for null cases
        if (Field == "B") & (view == "y"):
            print("Think about the problem geometry. There is NO By in this case.")
        elif (Field == "E") & (view == "x") | (Field == "E") & (view == "z"):
            print(
                "Think about the problem geometry. There is NO Ex or Ez in this case. Only Ey."
            )
        elif (Field == "J") & (view == "x") | (Field == "J") & (view == "z"):
            print(
                "Think about the problem geometry. There is NO Jx or Jz in this case. Only Jy."
            )
        elif (Field == "E") & (view == "vec"):
            print(
                "Think about the problem geometry. E only has components along y. Vector plot not possible"
            )
        elif (Field == "J") & (view == "vec"):
            print(
                "Think about the problem geometry. J only has components along y. Vector plot not possible"
            )
        elif (view == "vec") & (ComplexNumber == "Amp") | (view == "vec") & (
            ComplexNumber == "Phase"
        ):
            print(
                "Cannot show amplitude or phase when vector plot selected. Set 'AmpDir=None' to see amplitude or phase."
            )
        elif Field == "Model":
            plt.figure(figsize=(7, 6))
            ax = plt.subplot(111)
            if Scenario == "Sphere":
                model2D, mapping2D = self.getCoreModel("Sphere")
            elif Scenario == "Layer":
                model2D, mapping2D = self.getCoreModel("Layer")

            out = self.mesh2D.plotImage(np.log10(mapping2D * model2D), ax=ax)
            cb = plt.colorbar(out[0], ax=ax, format="$10^{%.1f}$")
            cb.set_label("$\sigma$ (S/m)")
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Depth (m)")
            ax.set_title("Conductivity Model")
            plt.show()
        else:
            plt.figure(figsize=(10, 6))
            ax = plt.subplot(111)
            vec = False
            if view == "vec":
                tname = "Vector "
                title = tname + Field + "-field"
            elif view == "amp":
                tname = "|"
                title = tname + Field + "|-field"
            else:
                if ComplexNumber == "real":
                    tname = "Re("
                elif ComplexNumber == "imag":
                    tname = "Im("
                elif ComplexNumber == "amplitude":
                    tname = "Amp("
                elif ComplexNumber == "phase":
                    tname = "Phase("
                title = tname + Field + view + ")-field"

            if Field == "B":
                label = "Magnetic field (T)"
                if view == "vec":
                    vec = True
                    if ComplexNumber == "real":
                        val = np.c_[self.Bx.real, self.Bz.real]
                    elif ComplexNumber == "imag":
                        val = np.c_[self.Bx.imag, self.Bz.imag]
                    else:
                        return

                elif view == "x":
                    if ComplexNumber == "real":
                        val = self.Bx.real
                    elif ComplexNumber == "imag":
                        val = self.Bx.imag
                    elif ComplexNumber == "amplitude":
                        val = abs(self.Bx)
                    elif ComplexNumber == "phase":
                        val = np.angle(self.Bx)
                elif view == "z":
                    if ComplexNumber == "real":
                        val = self.Bz.real
                    elif ComplexNumber == "imag":
                        val = self.Bz.imag
                    elif ComplexNumber == "amplitude":
                        val = abs(self.Bz)
                    elif ComplexNumber == "phase":
                        val = np.angle(self.Bz)
                else:
                    return

            elif Field == "E":
                label = "Electric field (V/m)"
                if view == "y":
                    if ComplexNumber == "real":
                        val = self.Ey.real
                    elif ComplexNumber == "imag":
                        val = self.Ey.imag
                    elif ComplexNumber == "amplitude":
                        val = abs(self.Ey)
                    elif ComplexNumber == "phase":
                        val = np.angle(self.Ey)
                else:
                    return

            elif Field == "J":
                label = "Current density (A/m$^2$)"
                if view == "y":
                    if ComplexNumber == "real":
                        val = self.Jy.real
                    elif ComplexNumber == "imag":
                        val = self.Jy.imag
                    elif ComplexNumber == "amplitude":
                        val = abs(self.Jy)
                    elif ComplexNumber == "phase":
                        val = np.angle(self.Jy)
                else:
                    return

            out = utils.plot2Ddata(
                self.mesh2D.gridCC,
                val,
                vec=vec,
                ax=ax,
                contourOpts={"cmap": "viridis"},
                ncontour=200,
                scale=scale,
            )
            cb = plt.colorbar(out[0], ax=ax, format="%.2e")
            cb.set_label(label)
            xmax = self.mesh2D.gridCC[:, 0].max()
            if Geometry:
                if Scenario == "Layer":
                    ax.plot(np.r_[-xmax, xmax], np.ones(2) * self.srcLoc[2], "w-", lw=1)
                    ax.plot(np.r_[-xmax, xmax], np.ones(2) * self.z0, "w--", lw=1)
                    ax.plot(np.r_[-xmax, xmax], np.ones(2) * self.z1, "w--", lw=1)
                    ax.plot(np.r_[-xmax, xmax], np.ones(2) * self.z2, "w--", lw=1)
                    ax.plot(0, self.srcLoc[2], "ko", ms=4)
                    ax.plot(self.rxLoc[0, 0], self.srcLoc[2], "ro", ms=4)
                elif Scenario == "Sphere":
                    ax.plot(np.r_[-xmax, xmax], np.ones(2) * self.srcLoc[2], "k-", lw=1)
                    ax.plot(np.r_[-xmax, xmax], np.ones(2) * self.z0, "w--", lw=1)
                    ax.plot(np.r_[-xmax, xmax], np.ones(2) * self.z1, "w--", lw=1)
                    ax.plot(
                        np.r_[-xmax, xmax], np.ones(2) * (self.z1 - self.h), "w--", lw=1
                    )
                    Phi = np.linspace(0, 2 * np.pi, 41)
                    ax.plot(
                        self.R * np.cos(Phi),
                        self.z2 + self.R * np.sin(Phi),
                        "w--",
                        lw=1,
                    )
                    ax.plot(0, self.srcLoc[2], "ko", ms=4)
                    ax.plot(self.rxLoc[0, 0], self.srcLoc[2], "ro", ms=4)

            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Depth (m)")
            title = title + "\nf = " + "{:.2e}".format(Frequency) + " Hz"
            ax.set_title(title)
            plt.show()

    ######################################################
    # LAYER WIDGET
    ######################################################

    def InteractivePlane_Layer(self, scale="log", fieldvalue="B", compvalue="z"):
        def foo(
            Field,
            AmpDir,
            Component,
            ComplexNumber,
            Sigma0,
            Sigma1,
            Sigma2,
            Sigma3,
            Sus,
            h1,
            h2,
            Scale,
            rxOffset,
            z,
            ifreq,
            Geometry=True,
        ):

            Frequency = 10 ** ((ifreq - 1.0) / 3.0 - 2.0)

            if ComplexNumber == "Re":
                ComplexNumber = "real"
            elif ComplexNumber == "Im":
                ComplexNumber = "imag"
            elif ComplexNumber == "Amp":
                ComplexNumber = "amplitude"
            elif ComplexNumber == "Phase":
                ComplexNumber = "phase"

            if AmpDir == "Direction (B or Bsec)":
                # ComplexNumber = "real"
                Component = "vec"
            if Field == "Bsec":
                bType = "bSecondary"
                Field = "B"
            else:
                bType = "b"

            self.setThreeLayerParam(
                h1=h1,
                h2=h2,
                sig0=Sigma0,
                sig1=Sigma1,
                sig2=Sigma2,
                sig3=Sigma3,
                chi=Sus,
            )
            srcLoc = np.array([0.0, 0.0, z])
            rxLoc = np.array([[rxOffset, 0.0, z]])
            self.simulate(srcLoc, rxLoc, np.r_[Frequency])
            if Field != "Model":
                self.getFields(bType=bType)
            return self.plotField(
                Field=Field,
                ComplexNumber=ComplexNumber,
                view=Component,
                scale=Scale,
                Frequency=Frequency,
                Geometry=Geometry,
                Scenario="Layer",
            )

        out = widgetify(
            foo,
            Field=widgets.ToggleButtons(
                options=["E", "B", "Bsec", "J", "Model"], value="E"
            ),
            AmpDir=widgets.ToggleButtons(
                options=["None", "Direction (B or Bsec)"], value="None"
            ),
            Component=widgets.ToggleButtons(
                options=["x", "y", "z"], value="y", description="Comp."
            ),
            ComplexNumber=widgets.ToggleButtons(
                options=["Re", "Im", "Amp", "Phase"], value="Re", description="Re/Im"
            ),
            Sigma0=widgets.FloatText(
                value=1e-8, continuous_update=False, description="$\sigma_0$ (S/m)"
            ),
            Sigma1=widgets.FloatText(
                value=0.01, continuous_update=False, description="$\sigma_1$ (S/m)"
            ),
            Sigma2=widgets.FloatText(
                value=0.01, continuous_update=False, description="$\sigma_2$ (S/m)"
            ),
            Sigma3=widgets.FloatText(
                value=0.01, continuous_update=False, description="$\sigma_3$ (S/m)"
            ),
            Sus=widgets.FloatText(
                value=0.0, continuous_update=False, description="$\chi$"
            ),
            h1=widgets.FloatSlider(
                min=2.0,
                max=40.0,
                step=2.0,
                value=10.0,
                continuous_update=False,
                description="$h_1$ (m)",
            ),
            h2=widgets.FloatSlider(
                min=2.0,
                max=40.0,
                step=2.0,
                value=10.0,
                continuous_update=False,
                description="$h_2$ (m)",
            ),
            Scale=widgets.ToggleButtons(options=["log", "linear"], value="linear"),
            rxOffset=widgets.FloatSlider(
                min=0.0,
                max=50.0,
                step=2.0,
                value=10.0,
                continuous_update=False,
                description="$\Delta x$(m)",
            ),
            z=widgets.FloatSlider(
                min=0.0,
                max=50.0,
                step=2.0,
                value=0.0,
                continuous_update=False,
                description="$\Delta z$ (m)",
            ),
            ifreq=widgets.IntSlider(
                min=1,
                max=31,
                step=1,
                value=16,
                continuous_update=False,
                description="f index",
            ),
        )
        return out

    def InteractiveData_Layer(self, fieldvalue="B", compvalue="z", z=0.0):
        frequency = np.logspace(2, 5, 31)
        self.simulate(self.srcLoc, self.rxLoc, frequency)

        def foo(Field, Component, Scale):

            # Printout for null cases
            if (Field == "B") & (Component == "y") | (Field == "Bsec") & (
                Component == "y"
            ):
                print("Think about the problem geometry. There is NO By in this case.")
            elif (Field == "E") & (Component == "x") | (Field == "E") & (
                Component == "z"
            ):
                print(
                    "Think about the problem geometry. There is NO Ex or Ez in this case. Only Ey."
                )
            else:
                plt.figure()
                ax = plt.subplot(111)
                bType = "b"
                if (Field == "Bsec") or (Field == "B"):
                    if Field == "Bsec":
                        bType = "bSecondary"
                    Field = "B"
                    self.getData(bType=bType)
                    label = "Magnetic field (T)"
                    if Component == "x":
                        title = "Bx"
                        valr = self.Bx.real
                        vali = self.Bx.imag
                    elif Component == "z":
                        title = "Bz"
                        valr = self.Bz.real
                        vali = self.Bz.imag
                    else:
                        # ax.imshow(self.im)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        print(
                            "Think about the problem geometry. There is NO By in this case."
                        )

                elif Field == "E":
                    self.getData(bType=bType)
                    label = "Electric field (V/m)"
                    title = "Ey"
                    if Component == "y":
                        valr = self.Ey.real
                        vali = self.Ey.imag
                    else:
                        # ax.imshow(self.im)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        print(
                            "Think about the problem geometry. There is NO Ex or Ez in this case."
                        )

                elif Field == "J":
                    print(
                        "The conductivity at the location is 0. Therefore there is no electrical current here."
                    )

                if Scale == "log":
                    valr_p, valr_n = DisPosNegvalues(valr)
                    vali_p, vali_n = DisPosNegvalues(vali)
                    ax.plot(frequency, valr_p, "k-")
                    ax.plot(frequency, valr_n, "k--")
                    ax.plot(frequency, vali_p, "r-")
                    ax.plot(frequency, vali_n, "r--")
                    ax.legend(
                        ("Re (+)", "Re (-)", "Im (+)", "Im (-)"), loc=4, fontsize=10
                    )
                else:
                    ax.plot(frequency, valr, "k.-")
                    ax.plot(frequency, vali, "r.-")
                    ax.legend(("Re", "Im"), loc=4, fontsize=10)
                ax.set_xscale("log")
                ax.set_yscale(Scale)
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel(label)
                ax.set_title(title)
                ax.grid(True)
                plt.show()

        out = widgetify(
            foo,
            Field=widgets.ToggleButtons(options=["E", "B", "Bsec"], value=fieldvalue),
            Component=widgets.ToggleButtons(
                options=["x", "y", "z"], value=compvalue, description="Comp."
            ),
            Scale=widgets.ToggleButtons(options=["log", "linear"], value="log"),
        )
        return out

    ######################################################
    # SPHERE WIDGET
    ######################################################

    def InteractivePlane_Sphere(self, scale="log", fieldvalue="B", compvalue="z"):
        def foo(
            Field,
            AmpDir,
            Component,
            ComplexNumber,
            Sigma0,
            Sigmab,
            Sigma1,
            Sigma2,
            Sus,
            d1,
            h,
            d2,
            R,
            Scale,
            rxOffset,
            z,
            ifreq,
            Geometry=True,
        ):

            Frequency = 10 ** ((ifreq - 1.0) / 3.0 - 2.0)

            if ComplexNumber == "Re":
                ComplexNumber = "real"
            elif ComplexNumber == "Im":
                ComplexNumber = "imag"
            elif ComplexNumber == "Amp":
                ComplexNumber = "amplitude"
            elif ComplexNumber == "Phase":
                ComplexNumber = "phase"

            if AmpDir == "Direction (B or Bsec)":
                # ComplexNumber = "real"
                Component = "vec"
            if Field == "Bsec":
                bType = "bSecondary"
                Field = "B"
            else:
                bType = "b"

            self.setLayerSphereParam(
                d1=d1,
                h=h,
                d2=d2,
                R=R,
                sig0=Sigma0,
                sigb=Sigmab,
                sig1=Sigma1,
                sig2=Sigma2,
                chi=Sus,
            )
            srcLoc = np.array([0.0, 0.0, z])
            rxLoc = np.array([[rxOffset, 0.0, z]])
            self.simulate(srcLoc, rxLoc, np.r_[Frequency])
            if Field != "Model":
                self.getFields(bType=bType)
            return self.plotField(
                Field=Field,
                ComplexNumber=ComplexNumber,
                Frequency=Frequency,
                view=Component,
                scale=Scale,
                Geometry=Geometry,
                Scenario="Sphere",
            )

        out = widgetify(
            foo,
            Field=widgets.ToggleButtons(
                options=["E", "B", "Bsec", "J", "Model"], value="E"
            ),
            AmpDir=widgets.ToggleButtons(
                options=["None", "Direction (B or Bsec)"], value="None"
            ),
            Component=widgets.ToggleButtons(
                options=["x", "y", "z"], value="y", description="Comp."
            ),
            ComplexNumber=widgets.ToggleButtons(
                options=["Re", "Im", "Amp", "Phase"], value="Re", description="Re/Im"
            ),
            Sigma0=widgets.FloatText(
                value=1e-8, continuous_update=False, description="$\sigma_0$ (S/m)"
            ),
            Sigmab=widgets.FloatText(
                value=0.01, continuous_update=False, description="$\sigma_b$ (S/m)"
            ),
            Sigma1=widgets.FloatText(
                value=0.01, continuous_update=False, description="$\sigma_1$ (S/m)"
            ),
            Sigma2=widgets.FloatText(
                value=0.01, continuous_update=False, description="$\sigma_2$ (S/m)"
            ),
            Sus=widgets.FloatText(
                value=0.0, continuous_update=False, description="$\chi$"
            ),
            d1=widgets.FloatSlider(
                min=0.0,
                max=50.0,
                step=2.0,
                value=0.0,
                continuous_update=False,
                description="$d_1$ (m)",
            ),
            h=widgets.FloatSlider(
                min=2.0,
                max=20.0,
                step=2.0,
                value=10.0,
                continuous_update=False,
                description="$h$ (m)",
            ),
            d2=widgets.FloatSlider(
                min=10.0,
                max=50.0,
                step=2.0,
                value=30.0,
                continuous_update=False,
                description="$d_2$ (m)",
            ),
            R=widgets.FloatSlider(
                min=2.0,
                max=20.0,
                step=2.0,
                value=10.0,
                continuous_update=False,
                description="$R$ (m)",
            ),
            Scale=widgets.ToggleButtons(options=["log", "linear"], value="linear"),
            rxOffset=widgets.FloatSlider(
                min=0.0,
                max=50.0,
                step=2.0,
                value=10.0,
                continuous_update=False,
                description="$\Delta x$(m)",
            ),
            z=widgets.FloatSlider(
                min=0.0,
                max=50.0,
                step=2.0,
                value=0.0,
                continuous_update=False,
                description="$\Delta z$ (m)",
            ),
            ifreq=widgets.IntSlider(
                min=1,
                max=31,
                step=1,
                value=16,
                continuous_update=False,
                description="f index",
            ),
        )
        return out

    def InteractiveData_Sphere(self, fieldvalue="B", compvalue="z", z=0.0):
        frequency = np.logspace(2, 5, 31)
        self.simulate(self.srcLoc, self.rxLoc, frequency)

        def foo(Field, Component, Scale):
            # Printout for null cases
            if (Field == "B") & (Component == "y") | (Field == "Bsec") & (
                Component == "y"
            ):
                print("Think about the problem geometry. There is NO By in this case.")
            elif (Field == "E") & (Component == "x") | (Field == "E") & (
                Component == "z"
            ):
                print(
                    "Think about the problem geometry. There is NO Ex or Ez in this case. Only Ey."
                )
            else:
                plt.figure()
                ax = plt.subplot(111)
                bType = "b"
                if (Field == "Bsec") or (Field == "B"):
                    if Field == "Bsec":
                        bType = "bSecondary"
                    Field = "B"
                    self.getData(bType=bType)
                    label = "Magnetic field (T)"
                    if Component == "x":
                        title = "Bx"
                        valr = self.Bx.real
                        vali = self.Bx.imag
                    elif Component == "z":
                        title = "Bz"
                        valr = self.Bz.real
                        vali = self.Bz.imag
                    else:
                        # ax.imshow(self.im)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        print(
                            "Think about the problem geometry. There is NO By in this case."
                        )

                elif Field == "E":
                    self.getData(bType=bType)
                    label = "Electric field (V/m)"
                    title = "Ey"
                    if Component == "y":
                        valr = self.Ey.real
                        vali = self.Ey.imag
                    else:
                        # ax.imshow(self.im)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        print(
                            "Think about the problem geometry. There is NO Ex or Ez in this case."
                        )

                elif Field == "J":
                    print(
                        "The conductivity at the location is 0. Therefore there is no electrical current here."
                    )

                if Scale == "log":
                    valr_p, valr_n = DisPosNegvalues(valr)
                    vali_p, vali_n = DisPosNegvalues(vali)
                    ax.plot(frequency, valr_p, "k-")
                    ax.plot(frequency, valr_n, "k--")
                    ax.plot(frequency, vali_p, "r-")
                    ax.plot(frequency, vali_n, "r--")
                    ax.legend(
                        ("Re (+)", "Re (-)", "Im (+)", "Im (-)"), loc=4, fontsize=10
                    )
                else:
                    ax.plot(frequency, valr, "k.-")
                    ax.plot(frequency, vali, "r.-")
                    ax.legend(("Re", "Im"), loc=4, fontsize=10)
                ax.set_xscale("log")
                ax.set_yscale(Scale)
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel(label)
                ax.set_title(title)
                ax.grid(True)
                plt.show()

        out = widgetify(
            foo,
            Field=widgets.ToggleButtons(options=["E", "B", "Bsec"], value=fieldvalue),
            Component=widgets.ToggleButtons(
                options=["x", "y", "z"], value=compvalue, description="Comp."
            ),
            Scale=widgets.ToggleButtons(options=["log", "linear"], value="log"),
        )
        return out
