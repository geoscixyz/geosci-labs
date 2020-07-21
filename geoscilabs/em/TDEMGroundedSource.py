import numpy as np
import matplotlib
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
    url="https://storage.googleapis.com/simpeg/em_examples/tdem_groundedsource/tdem_groundedsource.tar",
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


def load_or_run_results(
    re_run=False, fname=None, sigma_block=0.01, sigma_halfspace=0.01
):
    if re_run:
        run_simulation(
            fname=fname, sigma_block=sigma_block, sigma_halfspace=sigma_halfspace
        )
    else:
        downloads, directory = download_and_unzip_data()
        fname = os.path.sep.join([directory, fname])

    simulation_results = dd.io.load(fname)
    mesh = TensorMesh(
        simulation_results["mesh"]["h"], x0=simulation_results["mesh"]["x0"]
    )
    sigma = simulation_results["sigma"]
    times = simulation_results["time"]
    input_currents = simulation_results["input_currents"]
    E = simulation_results["E"]
    B = simulation_results["B"]
    J = simulation_results["J"]
    output = {
        "mesh": mesh,
        "sigma": sigma,
        "times": times,
        "input_currents": input_currents,
        "E": E,
        "B": B,
        "J": J,
    }
    return output


def choose_model(model):
    known_names = ["tdem_gs_half.h5", "tdem_gs_conductor.h5", "tdem_gs_resistor.h5"]
    if model == "halfspace":
        ind = 0
    elif model == "conductor":
        ind = 1
    elif model == "resistor":
        ind = 2
    return known_names[ind]


def run_simulation(fname="tdem_gs_half.h5", sigma_block=0.01, sigma_halfspace=0.01):
    from SimPEG.electromagnetics import time_domain as tdem
    from SimPEG.electromagnetics.utils import waveform_utils
    from scipy.constants import mu_0
    import numpy as np
    from SimPEG import maps, utils
    from pymatsolver import Pardiso

    cs = 20
    ncx, ncy, ncz = 20, 20, 20
    npad = 10
    hx = [(cs, npad, -1.5), (cs, ncx), (cs, npad, 1.5)]
    hy = [(cs, npad, -1.5), (cs, ncy), (cs, npad, 1.5)]
    hz = [(cs, npad, -1.5), (cs, ncz), (cs, npad, 1.5)]
    mesh = TensorMesh([hx, hy, hz], "CCC")
    sigma = np.ones(mesh.nC) * sigma_halfspace
    blk_ind = utils.ModelBuilder.getIndicesBlock(
        np.r_[-40, -40, -160], np.r_[40, 40, -80], mesh.gridCC
    )
    sigma[mesh.gridCC[:, 2] > 0.0] = 1e-8
    sigma[blk_ind] = sigma_block

    xmin, xmax = -200.0, 200.0
    ymin, ymax = -200.0, 200.0
    x = mesh.vectorCCx[np.logical_and(mesh.vectorCCx > xmin, mesh.vectorCCx < xmax)]
    y = mesh.vectorCCy[np.logical_and(mesh.vectorCCy > ymin, mesh.vectorCCy < ymax)]
    xyz = utils.ndgrid(x, y, np.r_[-1.0])

    px = np.r_[-200.0, 200.0]
    py = np.r_[0.0, 0.0]
    pz = np.r_[0.0, 0.0]
    srcLoc = np.c_[px, py, pz]

    from scipy.interpolate import interp1d

    t0 = 0.01 + 1e-4
    times = np.logspace(-4, -2, 21)
    rx_ex = tdem.receivers.PointElectricField(xyz, times + t0, orientation="x")
    rx_ey = tdem.receivers.PointElectricField(xyz, times + t0, orientation="y")
    rx_by = tdem.receivers.PointElectricField(xyz, times + t0, orientation="y")

    rxList = [rx_ex, rx_ey, rx_by]

    sim = tdem.Simulation3DMagneticFluxDensity(mesh, sigma=sigma, verbose=True)
    sim.Solver = Pardiso
    sim.solverOpts = {"is_symmetric": False}
    sim.time_steps = [(1e-3, 10), (2e-5, 10), (1e-4, 10), (5e-4, 10), (1e-3, 10)]
    t0 = 0.01 + 1e-4
    out = waveform_utils.VTEMFun(sim.times, 0.01, t0, 200)
    wavefun = interp1d(sim.times, out)
    waveform = tdem.sources.RawWaveform(offTime=t0, waveFct=wavefun)

    src = tdem.sources.LineCurrent(rxList, loc=srcLoc, waveform=waveform)
    survey = tdem.survey.Survey([src])
    sim.survey = survey
    input_currents = wavefun(sim.times)

    f = sim.fields(sigma)

    xyzlim = np.array([[xmin, xmax], [ymin, ymax], [-400, 0.0]])
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
    tdem_gs = {
        "E": E,
        "B": B,
        "J": J,
        "sigma": sigma_core,
        "mesh": meshCore.serialize(),
        "time": sim.times - t0,
        "input_currents": input_currents,
    }
    dd.io.save(fname, tdem_gs)


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

    def show_3d_survey_geometry(self, elev, azim, show_block=False):
        X1, X2 = -250.0, 250.0
        Y1, Y2 = -250.0, 250.0
        Z1, Z2 = -400.0, 0.0

        def polyplane(verts, alpha=0.1, color="green"):
            poly = Poly3DCollection(verts)
            poly.set_alpha(alpha)
            poly.set_facecolor(color)
            return poly

        z1 = -100.0
        x = np.r_[X1, X2, X2, X1, X1]
        y = np.ones(5) * 0.0
        z = np.r_[Z1, Z1, Z2, Z2, Z1]
        verts = [list(zip(x, y, z))]
        polyplane(verts, color="green")
        x = np.r_[X1, X2, X2, X1, X1]
        y = np.r_[Y1, Y1, Y2, Y2, Y1]
        z = np.ones(5) * 0.0
        verts = [list(zip(x, y, z))]
        polyplane(verts, color="grey")
        x = np.r_[X1, X2, X2, X1, X1]
        y = np.r_[Y1, Y1, Y2, Y2, Y1]
        z = np.ones(5) * z1
        verts = [list(zip(x, y, z))]
        polyplane(verts, color="grey")

        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca(projection="3d")
        ax.plot3D(np.r_[-200, 200], np.r_[0, 0], np.r_[1, 1] * 0.0, "r-", lw=3)
        ax.plot3D(
            self.mesh.gridCC[:, 0],
            self.mesh.gridCC[:, 1],
            np.zeros_like(self.mesh.gridCC[:, 0]),
            "k.",
        )
        ax.legend(("Tx", "Rx"), loc=1)

        if show_block:

            xc, yc, zc = 0, 0, 0
            x1, x2 = -40, 40
            y1, y2 = -40, 40
            z1, z2 = -160, -80
            x = np.r_[x1, x2, x2, x1, x1]
            y = np.ones(5) * 0.0
            z = np.r_[z1, z1, z2, z2, z1]
            ax.plot3D(x, y, z, "k--")
            x = np.r_[x1, x2, x2, x1, x1]
            y = np.r_[y1, y1, y2, y2, y1]
            z = np.ones(5) * (z1 + z2) / 2.0
            ax.plot3D(x, y, z, "k--")

            block_xyz = np.asarray(
                [
                    [x1, x1, x2, x2, x1, x1, x2, x2],
                    [y1, y2, y2, y1, y1, y2, y2, y1],
                    [z1, z1, z1, z1, z2, z2, z2, z2],
                ]
            )
            xyz = block_xyz.T
            # Face 1
            ax.add_collection3d(
                Poly3DCollection(
                    [list(zip(xyz[:4, 0] + xc, xyz[:4, 1] + yc, xyz[:4, 2] + zc))],
                    facecolors="k",
                    alpha=0.5,
                )
            )

            # Face 2
            ax.add_collection3d(
                Poly3DCollection(
                    [list(zip(xyz[4:, 0] + xc, xyz[4:, 1] + yc, xyz[4:, 2] + zc))],
                    facecolors="k",
                    alpha=0.5,
                )
            )

            # Face 3
            ax.add_collection3d(
                Poly3DCollection(
                    [
                        list(
                            zip(
                                xyz[[0, 1, 5, 4], 0] + xc,
                                xyz[[0, 1, 5, 4], 1] + yc,
                                xyz[[0, 1, 5, 4], 2] + zc,
                            )
                        )
                    ],
                    facecolors="k",
                    alpha=0.5,
                )
            )

            # Face 4
            ax.add_collection3d(
                Poly3DCollection(
                    [
                        list(
                            zip(
                                xyz[[3, 2, 6, 7], 0] + xc,
                                xyz[[3, 2, 6, 7], 1] + yc,
                                xyz[[3, 2, 6, 7], 2] + zc,
                            )
                        )
                    ],
                    facecolors="k",
                    alpha=0.5,
                )
            )

            # Face 5
            ax.add_collection3d(
                Poly3DCollection(
                    [
                        list(
                            zip(
                                xyz[[0, 4, 7, 3], 0] + xc,
                                xyz[[0, 4, 7, 3], 1] + yc,
                                xyz[[0, 4, 7, 3], 2] + zc,
                            )
                        )
                    ],
                    facecolors="k",
                    alpha=0.5,
                )
            )

            # Face 6
            ax.add_collection3d(
                Poly3DCollection(
                    [
                        list(
                            zip(
                                xyz[[1, 5, 6, 2], 0] + xc,
                                xyz[[1, 5, 6, 2], 1] + yc,
                                xyz[[1, 5, 6, 2], 2] + zc,
                            )
                        )
                    ],
                    facecolors="k",
                    alpha=0.5,
                )
            )

        ax.set_xlim(X1, X2)
        ax.set_ylim(Y1, Y2)
        ax.set_zlim(Z1, Z2)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Depth (m)")
        # ax.set_aspect("equal")
        ax.view_init(elev=elev, azim=azim)
        plt.show()

    def plot_input_currents(self, itime, scale):
        plt.figure()

        plt.plot(self.times * 1e3, self.input_currents, "k|-")
        plt.plot(self.times[itime] * 1e3, self.input_currents[itime], "ro")
        plt.xlabel("Time (ms)")
        plt.ylabel("Normalized current")
        plt.xscale(scale)

    def getSlices(self, mesh, vec, itime, normal="Z", loc=0.0, isz=False):
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
                return utils.mkvc(vz), xy
            else:
                return np.c_[utils.mkvc(vx), utils.mkvc(vz)], xz

        elif normal == "X":
            ind = np.argmin(abs(mesh.vectorCCy - loc))
            vx = VEC[:, 0].reshape((mesh.nCx, mesh.nCy, mesh.nCz), order="F")[ind, :, :]
            vy = VEC[:, 1].reshape((mesh.nCx, mesh.nCy, mesh.nCz), order="F")[ind, :, :]
            vz = VEC[:, 2].reshape((mesh.nCx, mesh.nCy, mesh.nCz), order="F")[ind, :, :]
            yz = utils.ndgrid(mesh.vectorCCy, mesh.vectorCCz)
            if isz:
                return utils.mkvc(vz), xy
            else:
                return np.c_[utils.mkvc(vy), utils.mkvc(vz)], yz

    def plot_electric_currents(self, itime):
        exy, xy = self.getSlices(self.mesh, self.J, itime, normal="Z", loc=-100.5)
        exz, xz = self.getSlices(self.mesh, self.J, itime, normal="Y", loc=0.0)
        label = "Current density (A/m$^2$)"
        plt.figure(figsize=(12, 5))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        vmin, vmax = abs(np.r_[exz]).min(), abs(np.r_[exz]).max()
        out_xz = utils.plot2Ddata(
            xz, exz, vec=True, ncontour=20, contourOpts={"cmap": "viridis"}, ax=ax2
        )
        vmin, vmax = out_xz[0].get_clim()
        utils.plot2Ddata(
            xy,
            exy,
            vec=True,
            ncontour=20,
            contourOpts={"cmap": "viridis", "vmin": vmin, "vmax": vmax},
            ax=ax1,
        )
        ax1.set_aspect("equal", adjustable="box")
        ax2.set_aspect("equal", adjustable="box")
        plt.colorbar(
            out_xz[0],
            ax=ax1,
            format="%.1e",
            ticks=np.linspace(vmin, vmax, 5),
            fraction=0.02,
        )
        cb = plt.colorbar(
            out_xz[0],
            ax=ax2,
            format="%.1e",
            ticks=np.linspace(vmin, vmax, 5),
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
        title = ("Time at %.2f ms") % ((self.times[itime]) * 1e3)
        ax1.set_title(title)
        plt.tight_layout()

    def plot_magnetic_flux(self, itime):
        bxy, xy = self.getSlices(self.mesh, self.B, itime, normal="Z", loc=-100.5)
        byz, yz = self.getSlices(self.mesh, self.B, itime, normal="X", loc=0.0)
        label = "Magnetic flux density (T)"
        plt.figure(figsize=(12, 5))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        vmin, vmax = abs(np.r_[byz]).min(), abs(np.r_[byz]).max()
        out_yz = utils.plot2Ddata(
            yz, byz, vec=True, ncontour=20, contourOpts={"cmap": "viridis"}, ax=ax2
        )
        vmin, vmax = out_yz[0].get_clim()
        utils.plot2Ddata(
            xy,
            bxy,
            vec=True,
            ncontour=20,
            contourOpts={"cmap": "viridis", "vmin": vmin, "vmax": vmax},
            ax=ax1,
        )
        ax1.set_aspect("equal", adjustable="box")
        ax2.set_aspect("equal", adjustable="box")
        plt.colorbar(
            out_yz[0],
            ax=ax1,
            format="%.1e",
            ticks=np.linspace(vmin, vmax, 5),
            fraction=0.02,
        )
        cb = plt.colorbar(
            out_yz[0],
            ax=ax2,
            format="%.1e",
            ticks=np.linspace(vmin, vmax, 5),
            fraction=0.02,
        )
        cb.set_label(label)
        ax1.set_title("")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax2.set_xlabel("Y (m)")
        ax2.set_ylabel("Z (m)")
        ax1.set_xlim(self.xmin, self.xmax)
        ax1.set_ylim(self.ymin, self.ymax)
        ax2.set_xlim(self.xmin, self.xmax)
        ax2.set_ylim(self.zmin, self.zmax)
        ax1.plot(ax1.get_xlim(), np.zeros(2), "k--", lw=1)
        ax2.plot(ax1.get_xlim(), np.zeros(2) - 100.0, "k--", lw=1)
        title = ("Time at %.2f ms") % ((self.times[itime]) * 1e3)
        ax1.set_title(title)
        plt.tight_layout()
