from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from ipywidgets import *
from SimPEG import Mesh, Maps, EM, Utils
# from pymatsolver import PardisoSolver
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.constants import mu_0
import requests
from io import StringIO

from .Base import widgetify
from .DipoleWidgetFD import DisPosNegvalues
from .BiotSavart import BiotSavartFun


class TransientVMDCylWidget(object):
    """FDEMCylWidgete"""

    survey = None
    srcList = None
    mesh = None
    f = None
    activeCC = None
    srcLoc = None
    mesh2D = None
    mu = None
    counter = 0

    def __init__(self):
        self.genMesh()
        self.getCoreDomain()
        # url = "http://em.geosci.xyz/_images/disc_dipole.png"
        # response = requests.get(url)
        # self.im = Image.open(StringIO(response.content))
        self.time = np.logspace(-5, -2, 41)

    def mirrorArray(self, x, direction="x"):
        X = x.reshape((self.nx_core, self.ny_core), order="F")
        if direction == "x" or direction == "y" :
            X2 = np.vstack((-np.flipud(X), X))
        else:
            X2 = np.vstack((np.flipud(X), X))
        return X2

    def genMesh(self, h=0., cs=3., ncx=15, ncz=30, npad=20):
        """
            Generate cylindrically symmetric mesh
        """
        #TODO: Make it adaptive due to z location
        hx = [(cs, ncx), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
        self.mesh = Mesh.CylMesh([hx, 1, hz], '00C')

    def getCoreDomain(self, mirror=False, xmax=95, zmin=-95, zmax=95.):

        self.activeCC = (self.mesh.gridCC[:,0] <= xmax) & (np.logical_and(self.mesh.gridCC[:,2] >= zmin, self.mesh.gridCC[:,2] <= zmax))
        self.gridCCactive = self.mesh.gridCC[self.activeCC,:][:,[0, 2]]

        xind = (self.mesh.vectorCCx <= xmax)
        yind = np.logical_and(self.mesh.vectorCCz >= zmin, self.mesh.vectorCCz <= zmax)
        self.nx_core = xind.sum()
        self.ny_core = yind.sum()

        # if self.mesh2D is None:
        hx = np.r_[self.mesh.hx[xind][::-1], self.mesh.hx[xind]]
        hz = self.mesh.hz[yind]
        self.mesh2D = Mesh.TensorMesh([hx, hz], x0="CC")

    def getBiotSavrt(self, rxLoc):
        """
            Compute Biot-Savart operator: Gz and Gx
        """
        self.Gz = BiotSavartFun(self.mesh, rxLoc, component='z')
        self.Gx = BiotSavartFun(self.mesh, rxLoc, component='x')


    def setThreeLayerParam(self, h1=12, h2=12, sig0=1e-8, sig1=1e-2, sig2=1e-2, sig3=1e-2, chi=0.):
        self.h1 = h1      # 1st layer thickness
        self.h2 = h2      # 2nd layer thickness
        self.z0 = 0.
        self.z1 = self.z0-h1
        self.z2 = self.z0-h1-h2
        self.sig0 = sig0  # 0th layer \sigma (assumed to be air)
        self.sig1 = sig1  # 1st layer \sigma
        self.sig2 = sig2  # 2nd layer \sigma
        self.sig3 = sig3  # 3rd layer \sigma

        active = self.mesh.vectorCCz < self.z0
        ind1 = (self.mesh.vectorCCz < self.z0) & (self.mesh.vectorCCz >= self.z1)
        ind2 = (self.mesh.vectorCCz < self.z1) & (self.mesh.vectorCCz >= self.z2)
        self.mapping = Maps.SurjectVertical1D(self.mesh) * Maps.InjectActiveCells(self.mesh, active, sig0, nC=self.mesh.nCz)
        model = np.ones(self.mesh.nCz) * sig3
        model[ind1] = sig1
        model[ind2] = sig2
        self.m = model[active]
        self.mu = np.ones(self.mesh.nC)*mu_0
        self.mu[self.mesh.gridCC[:, 2]<0.] = (1.+chi)*mu_0
        return self.m

    def simulate(self, srcLoc, rxLoc, time, radius=1.):

        bz = EM.TDEM.Rx.Point_b(rxLoc, time, orientation='z')
        dbzdt = EM.TDEM.Rx.Point_dbdt(rxLoc, time, orientation='z')
        src = EM.TDEM.Src.CircularLoop([bz],
                                       waveform=EM.TDEM.Src.StepOffWaveform(),
                                       loc=srcLoc, radius=radius)
        self.srcList = [src]
        prb = EM.TDEM.Problem3D_b(self.mesh, sigmaMap=self.mapping)
        prb.timeSteps = [(1e-06, 10), (5e-06, 10), (1e-05, 10), (5e-5, 10), (1e-4, 10), (5e-4, 10), (1e-3, 10)]
        survey = EM.TDEM.Survey(self.srcList)
        prb.pair(survey)
        self.f = prb.fields(self.m)
        self.prb = prb
        dpred = survey.dpred(self.m, f=self.f)
        return dpred

    def getFields(self, itime):
        src = self.srcList[0]
        Pfx = self.mesh.getInterpolationMat(self.mesh.gridCC[self.activeCC,:], locType="Fx")
        Pfz = self.mesh.getInterpolationMat(self.mesh.gridCC[self.activeCC,:], locType="Fz")
        Ey = self.mesh.aveE2CC*self.f[src, "e", itime]
        Jy = Utils.sdiag(self.prb.sigma) * Ey

        self.Ey = Utils.mkvc(self.mirrorArray(Ey[self.activeCC], direction="y"))
        self.Jy = Utils.mkvc(self.mirrorArray(Jy[self.activeCC], direction="y"))
        self.Bx = Utils.mkvc(self.mirrorArray(Pfx*self.f[src, "b", itime], direction="x"))
        self.Bz = Utils.mkvc(self.mirrorArray(Pfz*self.f[src, "b", itime], direction="z"))
        self.dBxdt = Utils.mkvc(self.mirrorArray(-Pfx*self.mesh.edgeCurl*self.f[src, "e", itime], direction="x"))
        self.dBzdt = Utils.mkvc(self.mirrorArray(-Pfz*self.mesh.edgeCurl*self.f[src, "e", itime], direction="z"))

    def getData(self):
        src = self.srcList[0]
        Pfx = self.mesh.getInterpolationMat(self.rxLoc, locType="Fx")
        Pfz = self.mesh.getInterpolationMat(self.rxLoc, locType="Fz")
        Pey = self.mesh.getInterpolationMat(self.rxLoc, locType="Ey")

        self.Ey = (Pey*self.f[src, "e", :]).flatten()
        self.Bx = (Pfx*self.f[src, "b", :]).flatten()
        self.Bz = (Pfz*self.f[src, "b", :]).flatten()
        self.dBxdt = (-Pfx*self.mesh.edgeCurl*self.f[src, "e", :]).flatten()
        self.dBzdt = (-Pfz*self.mesh.edgeCurl*self.f[src, "e", :]).flatten()


    def plotField(self, Field='B', view="vec", scale="linear", itime=0, Geometry=True):
        fig = plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)
        vec = False
        if view == "vec":
            tname = "Vector "
            title = tname+Field+"-field"
        elif  view == "amp":
            tname = "|"
            title = tname+Field+"|-field"
        else:
            title = Field + view+"-field"

        if Field == "B":
            label = "Magnetic field (T)"
            if view == "vec":
                vec = True
                val = np.c_[self.Bx, self.Bz]
            elif view == "x":
                val = self.Bx
            elif view == "z":
                val = self.Bz
            else:
                # ax.imshow(self.im)
                ax.set_xticks([])
                ax.set_yticks([])
                return "Dude, think twice ... no By for VMD"

        elif Field == "dBdt":
            label = "Time derivative of magnetic field (T/s)"
            if view == "vec":
                vec = True
                val = np.c_[self.dBxdt, self.dBzdt]
            elif view == "x":
                val = self.dBxdt
            elif view == "z":
                val = self.dBzdt
            else:
                # ax.imshow(self.im)
                ax.set_xticks([])
                ax.set_yticks([])
                return "Dude, think twice ... no dBydt for VMD"

        elif Field == "E":
            label = "Electric field (V/m)"
            if view == "y":
                val = self.Ey
            else:
                # ax.imshow(self.im)
                ax.set_xticks([])
                ax.set_yticks([])
                return "Dude, think twice ... only Ey for VMD"

        elif Field == "J":
            label = "Current density (A/m$^2$)"
            if view=="y":
                val = self.Jy
            else:
                # ax.imshow(self.im)
                ax.set_xticks([])
                ax.set_yticks([])
                return "Dude, think twice ... only Jy for VMD"

        out = Utils.plot2Ddata(self.mesh2D.gridCC, val, vec=vec, ax=ax, contourOpts={"cmap":"viridis"}, ncontour=50, scale=scale)
        if scale == "linear":
            cb = plt.colorbar(out[0], ax=ax, ticks=np.linspace(out[0].vmin, out[0].vmax, 3), format="%.1e")
        elif scale == "log":
            cb = plt.colorbar(out[0], ax=ax, ticks=np.linspace(out[0].vmin, out[0].vmax, 3), format="$10^{%.1f}$")
        else:
            raise Exception("We consdier only linear and log scale!")
        cb.set_label(label)
        xmax = self.mesh2D.gridCC[:, 0].max()
        if Geometry:
            ax.plot(np.r_[-xmax, xmax], np.ones(2)*self.srcLoc[2], 'k-', lw=0.5)
            ax.plot(np.r_[-xmax, xmax], np.ones(2)*self.z0, 'k--', lw=0.5)
            ax.plot(np.r_[-xmax, xmax], np.ones(2)*self.z1, 'k--', lw=0.5)
            ax.plot(np.r_[-xmax, xmax], np.ones(2)*self.z2, 'k--', lw=0.5)
            ax.plot(0, self.srcLoc[2], 'ko', ms=4)
            ax.plot(self.rxLoc[0, 0], self.srcLoc[2], 'ro', ms=4)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(title)
        ax.text(-85, 90, ("Time at %.3f ms")%(self.prb.times[itime]*1e3), fontsize = 12)
        plt.show()

    def InteractivePlane(self, scale="log", fieldvalue="E", compvalue="y", sig0=1e-8, sig1=0.01, sig2=0.01, sig3=0.01,
                         radius=1., z0=0., x0=10.):
        def foo(Update, Field, AmpDir, Component, itime, Sigma0, Sigma1, Sigma2, Sigma3, Sus, z, h1, h2, Scale, rxOffset, radius, Geometry=True):

            if AmpDir == "Direction":
                Component = "vec"
            m = self.setThreeLayerParam(h1=h1, h2=h2, sig0=Sigma0, sig1=Sigma1, sig2=Sigma2, sig3=Sigma3, chi=Sus)
            self.srcLoc = np.array([0., 0., z])
            self.rxLoc = np.array([[rxOffset, 0., z]])
            self.radius = radius
            if Update =="True":
                dpred = self.simulate(self.srcLoc, self.rxLoc, self.time, self.radius)
            self.getFields(itime)
            return self.plotField(Field=Field, view=Component, scale=Scale, Geometry=Geometry, itime=itime)

        out = widgetify(foo
                        ,Update=widgets.ToggleButtons(options=["True", "False"], value="True") \
                        ,Field=widgets.ToggleButtons(options=["E", "B", "dBdt", "J"], value=fieldvalue) \
                        ,AmpDir=widgets.ToggleButtons(options=['None','Direction'], value="None") \
                        ,Component=widgets.ToggleButtons(options=['x','y','z'], value=compvalue, description='Comp.') \
                        ,itime=widgets.IntSlider(min=1, max=70, step=1, value=1, continuous_update=False, description='Time index') \
                        ,Sigma0=widgets.FloatText(value=sig0, continuous_update=False, description='$\sigma_0$ (S/m)') \
                        ,Sigma1=widgets.FloatText(value=sig1, continuous_update=False, description='$\sigma_1$ (S/m)') \
                        ,Sigma2=widgets.FloatText(value=sig2, continuous_update=False, description='$\sigma_2$ (S/m)') \
                        ,Sigma3=widgets.FloatText(value=sig3, continuous_update=False, description='$\sigma_3$ (S/m)') \
                        ,Sus=widgets.FloatText(value=0., continuous_update=False, description='$\chi$') \
                        ,z=widgets.FloatSlider(min=0., max=48., step=3., value=z0, continuous_update=False, description='$z$ (m)') \
                        ,h1=widgets.FloatSlider(min=3., max=48., step=3., value=6., continuous_update=False, description='$h_1$ (m)') \
                        ,h2=widgets.FloatSlider(min=3., max=48., step=3., value=6., continuous_update=False, description='$h_2$ (m)') \
                        ,Scale=widgets.ToggleButtons(options=['log','linear'], value="linear") \
                        ,rxOffset=widgets.FloatText(value=x0, continuous_update=False, description='x (m)') \
                        ,radius=widgets.FloatText(value=radius, continuous_update=False, description='Tx radius (m)') \
                        )
        return out

    def InteractiveData(self, fieldvalue="B", compvalue="z"):
        # dpred = self.simulate(self.srcLoc, self.rxLoc, frequency, radius=radius)
        def foo(Field, Component, Scale):
            fig = plt.figure()
            ax = plt.subplot(111)
            bType = "b"
            self.getData()
            if Field == "B":
                label = "Magnetic field (T)"
                if Component == "x":
                    title = "Bx"
                    val = self.Bx
                elif Component == "z":
                    title = "Bz"
                    val = self.Bz
                else:
                    # ax.imshow(self.im)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.show()
                    return "Dude, think twice ... no By for VMD"

            elif Field == "dBdt":
                label = "Time dervative of magnetic field (T/s)"
                if Component == "x":
                    title = "dBx/dt"
                    val = self.dBxdt
                elif Component == "z":
                    title = "dBz/dt"
                    val = self.dBzdt
                else:
                    # ax.imshow(self.im)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.show()
                    return "Dude, think twice ... no dBydt for VMD"

            else:
                label = "Electric field (V/m)"
                title = "Ey"
                if Component == "y":
                    val = self.Ey
                else:
                    # ax.imshow(self.im)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.show()
                    return "Dude, think twice ... only Ey for VMD"

            if Scale == "log":
                val_p, val_n = DisPosNegvalues(val)
                ax.plot(self.prb.times[10:]*1e3, val_p[10:], 'k-')
                ax.plot(self.prb.times[10:]*1e3, val_n[10:], 'k--')
                ax.legend(("(+)", "(-)"), loc=1, fontsize = 10)
            else:
                ax.plot(self.prb.times[10:]*1e3, val[10:], 'k.-')

            ax.set_xscale("log")
            ax.set_yscale(Scale)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel(label)
            ax.set_title(title)
            ax.grid(True)
            plt.show()


        out = widgetify(foo
                        ,Field=widgets.ToggleButtons(options=["E", "B", "dBdt"], value=fieldvalue) \
                        ,Component=widgets.ToggleButtons(options=['x','y','z'], value=compvalue, description='Comp.') \
                        ,Scale=widgets.ToggleButtons(options=['log','linear'], value="log") \
                        )
        return out
