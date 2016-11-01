from ipywidgets import *
from SimPEG import Mesh, Maps, EM, Utils
from pymatsolver import PardisoSolver
import matplotlib.pyplot as plt
import numpy as np
from DipoleWidgetFD import DisPosNegvalues

class HarmonicVMDCylWidget(object):
    """FDEMCylWidgete"""

    survey = None
    srcList = None
    mesh = None
    f = None
    activeCC = None
    srcLoc = None

    def __init__(self):
        self.genMesh()
        self.getCoreDomain()

    def genMesh(self, h=0., cs=3., ncx=15, ncz=30, npad=20):
        """
            Generate cylindrically symmetric mesh
        """
        #TODO: Make it adaptive due to z location
        hx = [(cs, ncx), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
        self.mesh = Mesh.CylMesh([hx, 1, hz], '00C')

    def getCoreDomain(self, mirror=False, xmax=100, zmin=-100, zmax=100.):
        self.activeCC = (self.mesh.gridCC[:,0] <= xmax) & (np.logical_and(self.mesh.gridCC[:,2] >= zmin, self.mesh.gridCC[:,2] <= zmax))
        self.gridCCactive = self.mesh.gridCC[self.activeCC,:][:,[0, 2]]

    def setThreeLayerParam(self, h1=12, h2=12, sig0=1e-8, sig1=1e-1, sig2=1e-2, sig3=1e-2):
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
        return self.m

    def simulate(self, srcLoc, rxLoc, freqs):
        bzr = EM.FDEM.Rx.Point_bSecondary(
            rxLoc,
            orientation='z',
            component='real'
        )
        bzi = EM.FDEM.Rx.Point_bSecondary(
            rxLoc,
            orientation='z',
            component='imag'
        )
        self.srcList = [EM.FDEM.Src.MagDipole([bzr, bzi], freq, srcLoc, orientation='Z')
                   for freq in freqs]
        prb = EM.FDEM.Problem3D_b(self.mesh, sigmaMap=self.mapping, Solver=PardisoSolver)
        survey = EM.FDEM.Survey(self.srcList)
        prb.pair(survey)
        self.f = prb.fields(self.m)
        self.prb = prb
        dpred = survey.dpred(self.m, f=self.f)
        self.srcLoc = srcLoc
        self.rxLoc = rxLoc
        return dpred

    def getFields(self, bType = "b"):
        src = self.srcList[0]
        Pfx = self.mesh.getInterpolationMat(self.mesh.gridCC[self.activeCC,:], locType="Fx")
        Pfz = self.mesh.getInterpolationMat(self.mesh.gridCC[self.activeCC,:], locType="Fz")
        Ey = self.mesh.aveE2CC*self.f[src, "e"]
        Jy = Utils.sdiag(self.prb.sigma) * Ey
        self.Ey = Ey[self.activeCC]
        self.Jy = Jy[self.activeCC]
        self.Bx = Pfx*self.f[src, bType]
        self.Bz = Pfz*self.f[src, bType]

    def getData(self, bType = "b"):

        Pfx = self.mesh.getInterpolationMat(self.rxLoc, locType="Fx")
        Pfz = self.mesh.getInterpolationMat(self.rxLoc, locType="Fz")
        Pey = self.mesh.getInterpolationMat(self.rxLoc, locType="Ey")

        self.Ey = (Pey*self.f[:, "e"]).flatten()
        self.Bx = (Pfx*self.f[:, bType]).flatten()
        self.Bz = (Pfz*self.f[:, bType]).flatten()

    def plotField(self, Field='B', ComplexNumber="real", view="vec", scale="linear", ifreq=0):
        fig = plt.figure(figsize=(5, 6))
        ax = plt.subplot(111)
        vec = False
        if view == "vec":
            tname = "Vector "
            title = tname+Field+"-field"
        elif  view == "amp":
            tname = "|"
            title = tname+Field+"|-field"
        else:
            if ComplexNumber == "real":
                tname = "Re("
            elif ComplexNumber == "imag":
                tname = "Im("
            elif ComplexNumber == "amplitude":
                tname = "Amp("
            elif ComplexNumber == "phase":
                tname = "Phase("
            title = tname + Field + view+")-field"

        if Field == "B":
            label = "Magnetic field (T)"
            if view == "vec":
                vec = True
                if ComplexNumber == "real":
                    val = np.c_[self.Bx.real, self.Bz.real]
                elif ComplexNumber == "imag":
                    val = np.c_[self.Bx.imag, self.Bz.imag]

            elif view=="x":
                if ComplexNumber == "real":
                    val = self.Bx.real
                elif ComplexNumber == "imag":
                    val = self.Bx.imag
                elif ComplexNumber == "amplitude":
                    val = abs(self.Bx)
                elif ComplexNumber == "phase":
                    val = np.angle(self.Bx)
            elif view=="z":
                if ComplexNumber == "real":
                    val = self.Bz.real
                elif ComplexNumber == "imag":
                    val = self.Bz.imag
                elif ComplexNumber == "amplitude":
                    val = abs(self.Bz)
                elif ComplexNumber == "phase":
                    val = np.angle(self.Bz)
            else:
                raise Exception("XXX")

        elif Field == "E":
            label = "Electric field (V/m)"
            if view=="y":
                if ComplexNumber == "real":
                    val = self.Ey.real
                elif ComplexNumber == "imag":
                    val = self.Ey.imag
                elif ComplexNumber == "amplitude":
                    val = abs(self.Ey)
                elif ComplexNumber == "phase":
                    val = np.angle(self.Ey)
            else:
                raise Exception("XXX")

        elif Field == "J":
            label = "Current density (A/m$^2$)"
            if view=="y":
                if ComplexNumber == "real":
                    val = self.Jy.real
                elif ComplexNumber == "imag":
                    val = self.Jy.imag
                elif ComplexNumber == "amplitude":
                    val = abs(self.Jy)
                elif ComplexNumber == "phase":
                    val = np.angle(self.Jy)
        out = Utils.plot2Ddata(self.gridCCactive, val, vec=vec, ax=ax, contourOpts={"cmap":"viridis"}, ncontour=50, scale=scale)
        if scale == "linear":
            cb = plt.colorbar(out[0], ax=ax, ticks=np.linspace(out[0].vmin, out[0].vmax, 3), format="%.1e")
        elif scale == "log":
            cb = plt.colorbar(out[0], ax=ax, ticks=np.linspace(out[0].vmin, out[0].vmax, 3), format="$10^{%.1f}$")
        else:
            raise Exception("XXX")
        cb.set_label(label)
        xmax = self.gridCCactive[:,0].max()
        ax.plot(np.r_[0, xmax], np.ones(2)*self.srcLoc[2], 'k-', lw=0.5)
        ax.plot(np.r_[0, xmax], np.ones(2)*self.z0, 'k--', lw=0.5)
        ax.plot(np.r_[0, xmax], np.ones(2)*self.z1, 'k--', lw=0.5)
        ax.plot(np.r_[0, xmax], np.ones(2)*self.z2, 'k--', lw=0.5)
        ax.plot(0, self.srcLoc[2], 'ko', ms=4)
        ax.plot(self.rxLoc[0, 0], self.srcLoc[2], 'ro', ms=4)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(title)

    def InteractivePlane(self, scale="log", fieldvalue="B", compvalue="z"):

        def foo(Field, AmpDir, Component, ComplexNumber, Frequency, Sigma0, Sigma1, Sigma2, Sigma3, z, h1, h2, Scale, rxOffset):

            if ComplexNumber == "Re":
                ComplexNumber = "real"
            elif ComplexNumber == "Im":
                ComplexNumber = "imag"
            elif ComplexNumber == "Amp":
                ComplexNumber = "amplitude"
            elif ComplexNumber == "Phase":
                ComplexNumber = "phase"

            if AmpDir == "Direction":
                # ComplexNumber = "real"
                Component = "vec"
            if Field == "Bsec":
                bType = "bSecondary"
                Field = "B"
            else:
                bType = "b"

            m = self.setThreeLayerParam(h1=h1, h2=h2, sig0=Sigma0, sig1=Sigma1, sig2=Sigma2, sig3=Sigma3)
            srcLoc = np.array([0., 0., z])
            rxLoc = np.array([[rxOffset, 0., z]])
            dpred = self.simulate(srcLoc, rxLoc, np.r_[Frequency])
            self.getFields(bType=bType)
            return self.plotField(Field=Field, ComplexNumber=ComplexNumber, view=Component, scale=Scale)

        out = widgets.interactive (foo
                        ,Field=widgets.ToggleButtons(options=["E", "B", "Bsec", "J"], value=fieldvalue) \
                        ,AmpDir=widgets.ToggleButtons(options=['None','Direction'], value="None") \
                        ,Component=widgets.ToggleButtons(options=['x','y','z'], value=compvalue, description='Comp.') \
                        ,ComplexNumber=widgets.ToggleButtons(options=['Re','Im','Amp', 'Phase']) \
                        ,Frequency=widgets.FloatText(value=100., continuous_update=False, description='f (Hz)') \
                        ,Sigma0=widgets.FloatText(value=1e-8, continuous_update=False, description='$\sigma_0$ (S/m)') \
                        ,Sigma1=widgets.FloatText(value=0.01, continuous_update=False, description='$\sigma_1$ (S/m)') \
                        ,Sigma2=widgets.FloatText(value=0.01, continuous_update=False, description='$\sigma_2$ (S/m)') \
                        ,Sigma3=widgets.FloatText(value=0.01, continuous_update=False, description='$\sigma_3$ (S/m)') \
                        ,z=widgets.FloatSlider(min=0., max=48., step=3., value=0., continuous_update=False, description='$z$ (m)') \
                        ,h1=widgets.FloatSlider(min=3., max=48., step=3., value=6., continuous_update=False, description='$h_1$ (m)') \
                        ,h2=widgets.FloatSlider(min=3., max=48., step=3., value=6., continuous_update=False, description='$h_2$ (m)') \
                        ,Scale=widgets.ToggleButtons(options=['log','linear'], value="log") \
                        ,rxOffset=widgets.FloatText(value=10., continuous_update=False, description='x (m)') \
                        )
        return out

    def InteractiveData(self, fieldvalue="B", compvalue="x", z=0.):
        # srcLoc = np.array([0., 0., z])
        # rxLoc = np.array([[rxOffset, 0., z]])
        frequency = np.logspace(2, 5, 31)
        # m = self.setThreeLayerParam(h1=h1, h2=h2, sig0=Sigma0, sig1=Sigma1, sig2=Sigma2, sig3=Sigma3)
        dpred = self.simulate(self.srcLoc, self.rxLoc, frequency)
        def foo(Field, Component, Scale):
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
                    raise Exception("XXX")
            else:
                self.getData(bType=bType)
                label = "Electric field (V/m)"
                title = "Ey"
                if Component == "y":
                    valr = self.Ey.real
                    vali = self.Ey.imag
                else:
                    raise Exception("XXX")

            fig = plt.figure()
            ax = plt.subplot(111)
            if Scale == "log":
                valr_p, valr_n = DisPosNegvalues(valr)
                vali_p, vali_n = DisPosNegvalues(vali)
                ax.plot(frequency, valr_p, 'k-')
                ax.plot(frequency, valr_n, 'k--')
                ax.plot(frequency, vali_p, 'r-')
                ax.plot(frequency, vali_n, 'r--')
                ax.legend(("Re (+)", "Re (-)", "Im (+)", "Im (-)"), loc=4, fontsize = 10)
            else:
                ax.plot(frequency, valr, 'k.-')
                ax.plot(frequency, vali, 'r.-')
                ax.legend(("Re", "Im"), loc=4, fontsize = 10)
            ax.set_xscale("log")
            ax.set_yscale(Scale)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(label)
            ax.set_title(title)
            ax.grid(True)


        out = widgets.interactive (foo
                        ,Field=widgets.ToggleButtons(options=["E", "B", "Bsec"], value=fieldvalue) \
                        ,Component=widgets.ToggleButtons(options=['x','y','z'], value=compvalue, description='Comp.') \
                        ,Scale=widgets.ToggleButtons(options=['log','linear'], value="log") \
                        )
        return out
