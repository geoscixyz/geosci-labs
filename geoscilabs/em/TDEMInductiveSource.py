import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
from discretize import TensorMesh
from SimPEG import utils
import tarfile
import os

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


def download_and_unzip_data(
    url="https://storage.googleapis.com/simpeg/em_examples/tdem_inductivesource/tdem_inductivesource.tar",
):
    """
    Download the data from the storage bucket, unzip the tar file, return
    the directory where the data are
    """
    # download the data
    downloads = utils.download(url)

    # directory where the downloaded files are
    directory = downloads.split(".")[0]

    # unzip the tarfile
    tar = tarfile.open(downloads, "r")
    tar.extractall()
    tar.close()

    return downloads, directory


use_computed_results = True


def load_or_run_results(re_run=False, fname=None, src_type="VMD", sigma_halfspace=0.01):
    if re_run:
        run_simulation(fname=fname, sigma_halfspace=sigma_halfspace, src_type=src_type)
    else:
        downloads, directory = download_and_unzip_data()
        fname = os.path.sep.join([directory, fname])

    simulation_results = dd.io.load(fname)
    mesh = TensorMesh(
        simulation_results["mesh"]["h"], x0=simulation_results["mesh"]["x0"]
    )
    sigma = simulation_results["sigma"]
    times = simulation_results["time"]
    E = simulation_results["E"]
    B = simulation_results["B"]
    J = simulation_results["J"]
    output = {"mesh": mesh, "sigma": sigma, "times": times, "E": E, "B": B, "J": J}
    return output


def choose_source(src_type):
    known_names = ["tdem_vmd.h5", "tdem_hmd.h5"]
    if src_type == "VMD":
        ind = 0
    elif src_type == "HMD":
        ind = 1
    return known_names[ind]


def run_simulation(fname="tdem_vmd.h5", sigma_halfspace=0.01, src_type="VMD"):
    from SimPEG.electromagnetics import time_domain
    from scipy.constants import mu_0
    import numpy as np
    from SimPEG import maps
    from pymatsolver import Pardiso

    cs = 20.0
    ncx, ncy, ncz = 5, 3, 4
    npad = 10
    npadz = 10
    pad_rate = 1.3
    hx = [(cs, npad, -pad_rate), (cs, ncx), (cs, npad, pad_rate)]
    hy = [(cs, npad, -pad_rate), (cs, ncy), (cs, npad, pad_rate)]
    hz = utils.meshTensor([(cs, npadz, -1.3), (cs / 2.0, ncz), (cs, 5, 2)])
    mesh = TensorMesh([hx, hy, hz], x0=["C", "C", -hz[: int(npadz + ncz / 2)].sum()])
    sigma = np.ones(mesh.nC) * sigma_halfspace
    sigma[mesh.gridCC[:, 2] > 0.0] = 1e-8

    xmin, xmax = -600.0, 600.0
    ymin, ymax = -600.0, 600.0
    zmin, zmax = -600, 100.0

    times = np.logspace(-5, -2, 21)
    rxList = time_domain.receivers.PointMagneticFluxTimeDerivative(
        np.r_[10.0, 0.0, 30.0], times, orientation="z"
    )
    if src_type == "VMD":
        src = time_domain.sources.CircularLoop(
            [rxList],
            loc=np.r_[0.0, 0.0, 30.0],
            orientation="Z",
            waveform=time_domain.sources.StepOffWaveform(),
            radius=13.0,
        )
    elif src_type == "HMD":
        src = time_domain.sources.MagDipole(
            [rxList],
            loc=np.r_[0.0, 0.0, 30.0],
            orientation="X",
            waveform=time_domain.sources.StepOffWaveform(),
        )
    SrcList = [src]
    survey = time_domain.Survey(SrcList)
    sig = 1e-2
    sigma = np.ones(mesh.nC) * sig
    sigma[mesh.gridCC[:, 2] > 0] = 1e-8
    prb = time_domain.Simulation3DMagneticFluxDensity(
        mesh,
        sigmaMap=maps.IdentityMap(mesh),
        verbose=True,
        survey=survey,
        solver=Pardiso,
    )
    prb.time_steps = [
        (1e-06, 5),
        (5e-06, 5),
        (1e-05, 10),
        (5e-05, 10),
        (1e-4, 15),
        (5e-4, 16),
    ]

    f = prb.fields(sigma)

    xyzlim = np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])
    actinds, meshCore = utils.ExtractCoreMesh(xyzlim, mesh)
    Pex = mesh.getInterpolationMat(meshCore.gridCC, locType="Ex")
    Pey = mesh.getInterpolationMat(meshCore.gridCC, locType="Ey")
    Pez = mesh.getInterpolationMat(meshCore.gridCC, locType="Ez")
    Pfx = mesh.getInterpolationMat(meshCore.gridCC, locType="Fx")
    Pfy = mesh.getInterpolationMat(meshCore.gridCC, locType="Fy")
    Pfz = mesh.getInterpolationMat(meshCore.gridCC, locType="Fz")

    sigma_core = sigma[actinds]

    def getEBJcore(src0):
        B0 = np.r_[Pfx * f[src0, "b"], Pfy * f[src0, "b"], Pfz * f[src0, "b"]]
        E0 = np.r_[Pex * f[src0, "e"], Pey * f[src0, "e"], Pez * f[src0, "e"]]
        J0 = utils.sdiag(np.r_[sigma_core, sigma_core, sigma_core]) * E0
        return E0, B0, J0

    E, B, J = getEBJcore(src)
    tdem_is = {
        "E": E,
        "B": B,
        "J": J,
        "sigma": sigma_core,
        "mesh": meshCore.serialize(),
        "time": prb.times,
    }
    dd.io.save(fname, tdem_is)


# ------------------------------------------------------------------- #
# For visualizations
# ------------------------------------------------------------------- #


class PlotTDEM(object):
    """docstring for PlotTDEM"""

    mesh = None
    sigma = None
    times = None
    input_currents = None
    E = None
    B = None
    J = None

    def __init__(self, **kwargs):
        super(PlotTDEM, self).__init__()
        utils.setKwargs(self, **kwargs)
        self.xmin, self.xmax = self.mesh.vectorCCx.min(), self.mesh.vectorCCx.max()
        self.ymin, self.ymax = self.mesh.vectorCCy.min(), self.mesh.vectorCCy.max()
        self.zmin, self.zmax = self.mesh.vectorCCz.min(), self.mesh.vectorCCz.max()

    def show_3d_survey_geometry(self, elev, azim):
        X1, X2 = -250.0, 250.0
        Y1, Y2 = -250.0, 250.0
        Z1, Z2 = -400.0, 50.0

        def polyplane(verts, alpha=0.5, color="green"):
            poly = Poly3DCollection(verts)
            poly.set_alpha(alpha)
            poly.set_facecolor(color)
            return poly

        x = np.r_[X1, X2, X2, X1, X1]
        y = np.ones(5) * 0.0
        z = np.r_[Z1, Z1, Z2, Z2, Z1]
        verts = [list(zip(x, y, z))]
        polya = polyplane(verts, color="green")
        x = np.r_[X1, X2, X2, X1, X1]
        y = np.r_[Y1, Y1, Y2, Y2, Y1]
        z = np.ones(5) * 0.0
        verts = [list(zip(x, y, z))]
        polyb = polyplane(verts, color="grey")
        x = np.r_[X1, X2, X2, X1, X1]
        y = np.r_[Y1, Y1, Y2, Y2, Y1]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca(projection="3d")
        ax.plot3D(np.r_[0, 0], np.r_[0, 0], np.r_[1, 1] * 30.0, "ro", ms=5)
        ax.legend(("Tx",), loc=1)
        ax.plot3D(np.r_[X1, X2], np.r_[0, 0], np.r_[0, 0], "k-")
        ax.plot3D(np.r_[X1, X2], np.r_[0, 0], np.r_[30, 30], "k--")
        ax.add_collection3d(polya)
        ax.add_collection3d(polyb)

        ax.set_xlim(X1, X2)
        ax.set_ylim(Y1, Y2)
        ax.set_zlim(Z1, Z2)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Depth (m)")
        # ax.set_aspect("equal")
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        plt.show()

    def plot_input_currents(self, itime, scale):
        plt.figure()

        plt.plot(self.times * 1e3, np.zeros_like(self.times), "k|-")
        plt.plot(self.times[itime] * 1e3, np.zeros_like(self.times)[itime], "ro")
        plt.xlabel("Time (ms)")
        plt.ylabel("Normalized current")
        plt.xscale(scale)

    def getSlices(self, mesh, vec, itime, normal="Z", loc=0.0, isz=False, isy=False):
        VEC = vec[:, itime].reshape((mesh.nC, 3), order="F")
        if normal == "Z":
            ind = np.argmin(abs(mesh.vectorCCz - loc))
            vx = VEC[:, 0].reshape((mesh.nCx, mesh.nCy, mesh.nCz), order="F")[:, :, ind]
            vy = VEC[:, 1].reshape((mesh.nCx, mesh.nCy, mesh.nCz), order="F")[:, :, ind]
            vz = VEC[:, 2].reshape((mesh.nCx, mesh.nCy, mesh.nCz), order="F")[:, :, ind]
            xy = utils.ndgrid(mesh.vectorCCx, mesh.vectorCCy)
            if isz:
                return utils.mkvc(vz), xy
            else:
                return np.c_[utils.mkvc(vx), utils.mkvc(vy)], xy

        elif normal == "Y":
            ind = np.argmin(abs(mesh.vectorCCx - loc))
            vx = VEC[:, 0].reshape((mesh.nCx, mesh.nCy, mesh.nCz), order="F")[:, ind, :]
            vy = VEC[:, 1].reshape((mesh.nCx, mesh.nCy, mesh.nCz), order="F")[:, ind, :]
            vz = VEC[:, 2].reshape((mesh.nCx, mesh.nCy, mesh.nCz), order="F")[:, ind, :]
            xz = utils.ndgrid(mesh.vectorCCx, mesh.vectorCCz)
            if isz:
                return utils.mkvc(vz), xz
            elif isy:
                return utils.mkvc(vy), xz
            else:
                return np.c_[utils.mkvc(vx), utils.mkvc(vz)], xz

        elif normal == "X":
            ind = np.argmin(abs(mesh.vectorCCy - loc))
            vx = VEC[:, 0].reshape((mesh.nCx, mesh.nCy, mesh.nCz), order="F")[ind, :, :]
            vy = VEC[:, 1].reshape((mesh.nCx, mesh.nCy, mesh.nCz), order="F")[ind, :, :]
            vz = VEC[:, 2].reshape((mesh.nCx, mesh.nCy, mesh.nCz), order="F")[ind, :, :]
            yz = utils.ndgrid(mesh.vectorCCy, mesh.vectorCCz)
            if isz:
                return utils.mkvc(vy), yz
            elif isy:
                return utils.mkvc(vy), yz
            else:
                return np.c_[utils.mkvc(vy), utils.mkvc(vz)], yz

    def plot_electric_currents(self, itime):
        exy, xy = self.getSlices(self.mesh, self.J, itime, normal="Z", loc=-100.5)
        exz, xz = self.getSlices(
            self.mesh, self.J, itime, normal="Y", loc=0.0, isy=True
        )
        label = "Current density (A/m$^2$)"
        plt.figure(figsize=(12, 5))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        out_xz = utils.plot2Ddata(
            xz, exz, vec=False, ncontour=300, contourOpts={"cmap": "viridis"}, ax=ax2
        )
        vmin_xz, vmax_xz = out_xz[0].get_clim()
        out_xy = utils.plot2Ddata(
            xy, exy, vec=True, ncontour=300, contourOpts={"cmap": "viridis"}, ax=ax1
        )
        vmin_xy, vmax_xy = out_xy[0].get_clim()
        ax1.set_aspect("equal", adjustable="box")
        ax2.set_aspect("equal", adjustable="box")

        plt.colorbar(
            out_xy[0],
            ax=ax1,
            format="%.1e",
            ticks=np.linspace(vmin_xy, vmax_xy, 5),
            fraction=0.02,
        )
        cb = plt.colorbar(
            out_xz[0],
            ax=ax2,
            format="%.1e",
            ticks=np.linspace(vmin_xz, vmax_xz, 5),
            fraction=0.02,
        )
        cb.set_label(label)
        ax1.set_title("")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Z (m)")
        ax1.set_xlim(self.xmin, self.xmax)
        ax1.set_ylim(self.ymin, self.ymax)
        ax2.set_xlim(self.xmin, self.xmax)
        ax2.set_ylim(self.zmin, self.zmax)

        ax1.plot(ax1.get_xlim(), np.zeros(2), "k--", lw=1)
        ax2.plot(ax1.get_xlim(), np.zeros(2) - 100.0, "k--", lw=1)
        ax2.plot(ax2.get_xlim(), np.zeros(2), "k-", lw=1)
        ax2.plot(0, 30, "ro", ms=5)
        title = ("Time at %.2f ms") % ((self.times[itime]) * 1e3)
        ax1.set_title(title)
        plt.tight_layout()

    def plot_magnetic_flux(self, itime):
        bxy, xy = self.getSlices(
            self.mesh, self.B, itime, normal="Z", loc=-100.5, isz=True
        )
        bxz, xz = self.getSlices(self.mesh, self.B, itime, normal="Y", loc=0.0)
        label = "Magnetic flux density (T)"
        plt.figure(figsize=(12, 5))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

        out_xy = utils.plot2Ddata(
            xy, bxy, vec=False, ncontour=300, contourOpts={"cmap": "viridis"}, ax=ax1
        )
        vmin_xy, vmax_xy = out_xy[0].get_clim()
        out_xz = utils.plot2Ddata(
            xz, bxz, vec=True, ncontour=300, contourOpts={"cmap": "viridis"}, ax=ax2
        )
        vmin_xz, vmax_xz = out_xz[0].get_clim()
        ax1.set_aspect("equal", adjustable="box")
        ax2.set_aspect("equal", adjustable="box")

        plt.colorbar(
            out_xy[0],
            ax=ax1,
            format="%.1e",
            ticks=np.linspace(vmin_xy, vmax_xy, 5),
            fraction=0.02,
        )
        cb = plt.colorbar(
            out_xz[0],
            ax=ax2,
            format="%.1e",
            ticks=np.linspace(vmin_xz, vmax_xz, 5),
            fraction=0.02,
        )
        cb.set_label(label)
        ax1.set_title("")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Z (m)")
        ax1.set_xlim(self.xmin, self.xmax)
        ax1.set_ylim(self.ymin, self.ymax)
        ax2.set_xlim(self.xmin, self.xmax)
        ax2.set_ylim(self.zmin, self.zmax)
        ax1.plot(ax1.get_xlim(), np.zeros(2), "k--", lw=1)
        ax2.plot(ax1.get_xlim(), np.zeros(2) - 100.0, "k--", lw=1)
        ax2.plot(ax2.get_xlim(), np.zeros(2), "k-", lw=1)
        ax2.plot(0, 30, "ro", ms=5)
        title = ("Time at %.2f ms") % ((self.times[itime]) * 1e3)
        ax1.set_title(title)
        plt.tight_layout()
