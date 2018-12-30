from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy

matplotlib.rcParams["font.size"] = 13


def phase(z):
    val = np.angle(z)
    # val = np.rad2deg(np.unwrap(np.angle((z))))
    return val


class DataView(object):
    """
        Provides viewingtions for Data
        This can be inherited by XXX
    """

    def set_xyz(self, x, y, z, normal="Z", geometry="grid"):
        self.normal = normal
        self.geometry = geometry

        if geometry.upper() == "GRID":
            if normal.upper() == "X":
                self.x, self.y, self.z = x, y, z
                self.ncx, self.ncy, self.ncz = 1, y.size, z.size
                self.Y, self.Z = np.meshgrid(y, z)
                self.xyz = np.c_[
                    x*np.ones(self.ncy*self.ncz),
                    self.Y.flatten(),
                    self.Z.flatten()
                ]

            elif normal.upper() == "Y":
                self.x, self.y, self.z = x, y, z
                self.ncx, self.ncy, self.ncz = x.size, 1, z.size
                self.X, self.Z = np.meshgrid(x, z)
                self.xyz = np.c_[
                    self.X.flatten(),
                    y*np.ones(self.ncx*self.ncz),
                    self.Z.flatten()
                ]

            elif normal.upper() == "Z":
                self.x, self.y, self.z = x, y, z
                self.ncx, self.ncy, self.ncz = x.size, y.size, 1
                self.X, self.Y = np.meshgrid(x, y)
                self.xyz = np.c_[
                    self.X.flatten(),
                    self.Y.flatten(),
                    z*np.ones(self.ncx*self.ncy)
                ]

        elif geometry.upper() == "PROFILE":
            if normal.upper() == "X":
                self.x, self.y, self.z = x, y, z
                self.ncx, self.ncy, self.ncz = 1, y.size, 1
                self.Y, self.Z = self.y, self.z
                self.xyz = np.c_[x*np.ones_like(self.y), self.Y, self.Z]

            elif normal.upper() == "Y":
                self.x, self.y, self.z = x, y, z
                self.ncx, self.ncy, self.ncz = x.size, 1, 1
                self.Y, self.Z = self.y, self.z
                self.xyz = np.c_[self.x, y*np.ones_like(self.x), self.Z]

            elif normal.upper() == "Z":
                self.x, self.y, self.z = x, y, z
                self.ncx, self.ncy, self.ncz = x.size, 1, 1
                self.Y, self.Z = self.y, self.z
                self.xyz = np.c_[self.x, self.y, z*np.ones_like(self.x)]

    def eval_loc(
        self, srcLoc, obsLoc, log_sigvec, log_fvec, orientation, normal, func
    ):
        self.srcLoc = srcLoc
        self.obsLoc = obsLoc
        self.log_sigvec = log_sigvec
        self.log_fvec = log_fvec
        self.sigvec = 10.**log_sigvec
        self.fvec = 10.**log_fvec
        self.orientation = orientation
        self.normal = normal
        self.func1D = func
        self.val_xfs = np.zeros(
            (len(log_sigvec), len(log_fvec)), dtype=complex
        )
        self.val_yfs = np.zeros(
            (len(log_sigvec), len(log_fvec)), dtype=complex
        )
        self.val_zfs = np.zeros(
            (len(log_sigvec), len(log_fvec)), dtype=complex
        )

        for n in range(len(log_sigvec)):
            self.val_xfs[n], self.val_yfs[n], self.val_zfs[n] = func(
                self.obsLoc, srcLoc, 10.**log_sigvec[n], 10.**log_fvec,
                orientation=self.orientation
            )

    def eval(self, xyz, srcLoc, sig, f, orientation, func, normal="Z", t=0.):
        val_x, val_y, val_z = func(
            xyz, srcLoc, sig, f, orientation=orientation, t=t
        )
        return val_x, val_y, val_z

    def eval_TD(self, xyz, srcLoc, sig, t, orientation, func, normal="Z"):
        val_x, val_y, val_z = func(
            xyz, srcLoc, sig, t, orientation=orientation
        )
        return val_x, val_y, val_z

    def eval_2D(self, srcLoc, sig, f, orientation, func, t=0.):
        self.func2D = func
        self.srcLoc = srcLoc
        self.sig = sig
        self.t = f
        self.orientation = orientation
        self.val_x, self.val_y, self.val_z = func(
            self.xyz, srcLoc, sig, f, orientation=orientation, t=t
        )

        if self.normal.upper() == "X":
            def Freshape(v):
                return v.reshape(self.ncy, self.ncz)
        elif self.normal.upper() == "Y":
            def Freshape(v):
                return v.reshape(self.ncx, self.ncz)
        elif self.normal == "Z":
            def Freshape(v):
                return v.reshape(self.ncx, self.ncy)

        self.VAL_X = Freshape(self.val_x)
        self.VAL_Y = Freshape(self.val_y)
        self.VAL_Z = Freshape(self.val_z)
        self.VEC_R_amp = np.sqrt(
            self.VAL_X.real**2 + self.VAL_Y.real**2 + self.VAL_Z.real**2
        )
        self.VEC_I_amp = np.sqrt(
            self.VAL_X.imag**2 + self.VAL_Y.imag**2 + self.VAL_Z.imag**2
        )
        self.VEC_A_amp = np.sqrt(
            np.abs(self.VAL_X)**2 +
            np.abs(self.VAL_Y)**2 +
            np.abs(self.VAL_Z)**2
        )
        self.VEC_P_amp = np.sqrt(
            phase(self.VAL_X)**2 +
            phase(self.VAL_Y)**2 +
            phase(self.VAL_Z)**2
        )

    def eval_2D_TD(self, srcLoc, sig, t, orientation, func):
        self.func2D = func
        self.srcLoc = srcLoc
        self.sig = sig
        self.t = t
        self.orientation = orientation
        self.val_x, self.val_y, self.val_z = func(
            self.xyz, srcLoc, sig, t, orientation=orientation
        )

        if self.normal.upper() == "X":
            def Freshape(v):
                return v.reshape(self.ncy, self.ncz)
        elif self.normal.upper() == "Y":
            def Freshape(v):
                return v.reshape(self.ncx, self.ncz)
        elif self.normal.upper() == "Z":
            def Freshape(v):
                return v.reshape(self.ncx, self.ncy)

        self.VAL_X = Freshape(self.val_x)
        self.VAL_Y = Freshape(self.val_y)
        self.VAL_Z = Freshape(self.val_z)
        self.VEC_amp = np.sqrt(
            self.VAL_X.real**2 + self.VAL_Y.real**2 + self.VAL_Z.real**2
        )

    def plot2D_FD(
        self, component="real", view="vec", ncontour=20, logamp=True,
        clim=None, showcontour=False, levels=None, ax=None, colorbar=True,
        cmap="viridis"
    ):
        """
            2D visualization of dipole fields
        """
        if ax is None:
            fig = plt.figure(figsize=(6.5, 5))
            ax = plt.subplot(111)

        if component == "real":
            VAL_X = self.VAL_X.real
            VAL_Y = self.VAL_Y.real
            VAL_Z = self.VAL_Z.real
            VEC_amp = self.VEC_R_amp
        elif component == "imag":
            VAL_X = self.VAL_X.imag
            VAL_Y = self.VAL_Y.imag
            VAL_Z = self.VAL_Z.imag
            VEC_amp = self.VEC_I_amp
        elif component == "amplitude":
            VAL_X = abs(self.VAL_X)
            VAL_Y = abs(self.VAL_Y)
            VAL_Z = abs(self.VAL_Z)
            VEC_amp = self.VEC_A_amp
        elif component == "phase":
            VAL_X = phase(self.VAL_X)
            VAL_Y = phase(self.VAL_Y)
            VAL_Z = phase(self.VAL_Z)
            VEC_amp = self.VEC_P_amp
        else:
            raise Exception(
                "component should be in real, imag, amplitude, or phase!"
            )

        if view == "amp" or view == "vec":
            val = VEC_amp

        elif view.upper() == "X":
            val = VAL_X
        elif view.upper() == "Y":
            val = VAL_Y
        elif view.upper() == "Z":
            val = VAL_Z

        if logamp is True:
            zeroind = val == 0
            val = np.log10(abs(val))
            val[zeroind] = val[~zeroind].min()

        if self.normal.upper() == "X":
            a, b = self.y, self.z
            vec_a, vec_b = self.VAL_Y, self.VAL_Z
            xlabel = "Y (m)"
            ylabel = "Z (m)"

        elif self.normal.upper() == "Y":
            a, b = self.x, self.z
            vec_a, vec_b = self.VAL_X, self.VAL_Z
            xlabel = "X (m)"
            ylabel = "Z (m)"

        elif self.normal.upper() == "Z":
            a, b = self.x, self.y
            vec_a, vec_b = self.VAL_X, self.VAL_Y
            xlabel = "X (m)"
            ylabel = "Y (m)"

        if clim is None:
            vmin, vmax = val.min(), val.max()
        else:
            vmin, vmax = clim[0], clim[1]

        dat = ax.contourf(
            a, b, val, ncontour, clim=(vmin, vmax), vmin=vmin, vmax=vmax,
            cmap=cmap
        )

        if showcontour:
            ax.contour(a, b, val, levels, colors="k", linestyles="-")

        if colorbar:
            if logamp is True:
                cb = plt.colorbar(
                    dat, ax=ax, format="$10^{%.1f}$",
                    ticks=np.linspace(vmin, vmax, 3)
                )
            else:
                cb = plt.colorbar(
                    dat, ax=ax, format="%.1e", ticks=np.linspace(vmin, vmax, 3)
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if view == "vec":
            nx = self.x.size
            nskip = int(nx/15)
            if component == "real":
                # ax.quiver(a[::nskip], b[::nskip], (vec_a.real/VEC_amp)[::nskip,::nskip],  (vec_b.real/VEC_amp)[::nskip,::nskip], color="w", linewidth=0.5)
                ax.streamplot(
                    a, b, vec_a.real,  vec_b.real, color="w", linewidth=0.5
                )
            elif component == "imag":
                # ax.quiver(a, b, vec_a.imag/VEC_amp,  vec_b.imag/VEC_amp, color="w", linewidth=0.5)
                ax.streamplot(
                    a, b, vec_a.imag,  vec_b.imag, color="w", linewidth=0.5
                )
            if component == "amplitude":
                # ax.quiver(a, b, abs(vec_a)/VEC_amp,  abs(vec_b)/VEC_amp, color="w", linewidth=0.5)
                ax.streamplot(
                    a, b, abs(vec_a),  abs(vec_b), color="w", linewidth=0.5
                )
            elif component == "phase":
                # ax.quiver(a, b, phase(vec_a)/VEC_amp,  phase(vec_b)/VEC_amp, color="w", linewidth=0.5)
                ax.streamplot(
                    a, b, phase(vec_a),  phase(vec_b), color="w", linewidth=0.5
                )

        return ax, dat

    def plot2D_TD(
        self, view="vec", ncontour=20, logamp=True, clim=None,
        showcontour=False, levels=None, ax=None, colorbar=True, cmap="viridis"
    ):
        """
            2D visualization of dipole fields
        """
        if ax is None:
            fig = plt.figure(figsize=(6.5, 5))
            ax = plt.subplot(111)

        if view == "amp" or view == "vec":
            val = self.VEC_amp

        elif view.upper() == "X":
            val = self.VAL_X
        elif view.upper() == "Y":
            val = self.VAL_Y
        elif view.upper() == "Z":
            val = self.VAL_Z

        if logamp is True:
            zeroind = val == 0
            val = np.log10(abs(val))
            val[zeroind] = val[~zeroind].min()
        if self.normal.upper() == "X":
            a, b = self.y, self.z
            vec_a, vec_b = self.VAL_Y, self.VAL_Z
            xlabel = "Y (m)"
            ylabel = "Z (m)"
        elif self.normal.upper() == "Y":
            a, b = self.x, self.z
            vec_a, vec_b = self.VAL_X, self.VAL_Z
            xlabel = "X (m)"
            ylabel = "Z (m)"
        elif self.normal.upper() == "Z":
            a, b = self.x, self.y
            vec_a, vec_b = self.VAL_X, self.VAL_Y
            xlabel = "X (m)"
            ylabel = "Y (m)"

        if clim is None:
            vmin, vmax = val.min(), val.max()
        else:
            vmin, vmax = clim[0], clim[1]

        dat = ax.contourf(
            a, b, val, ncontour, clim=(vmin, vmax), vmin=vmin, vmax=vmax,
            cmap=cmap
        )

        if showcontour:
            ax.contour(a, b, val, levels, colors="k", linestyles="-")

        if colorbar:
            if logamp is True:
                cb = plt.colorbar(
                    dat, ax=ax, format="$10^{%.1f}$",
                    ticks=np.linspace(vmin, vmax, 3)
                )
            else:
                cb = plt.colorbar(
                    dat, ax=ax, format="%.1e",
                    ticks=np.linspace(vmin, vmax, 3)
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if view == "vec":
            nx = self.x.size
            nskip = int(nx/15)
            # ax.quiver(a[::nskip], b[::nskip], (vec_a.real/VEC_amp)[::nskip,::nskip],  (vec_b.real/VEC_amp)[::nskip,::nskip], color="w", linewidth=0.5)
            ax.streamplot(a, b, vec_a,  vec_b, color="w", linewidth=0.5)

        return ax, dat

    def plot_profile_FD(
        self, start, end, nbmp, component="real", view="x", logamp=True,
        ax=None, color="black"
    ):

        if ax is None:
            fig = plt.figure(figsize=(6.5, 5))
            ax = plt.subplot(111)

        if self.geometry.upper() == "PROFILE":
            start = self.xyz[0]
            end = self.xyz[-1]
            self1D = copy.deepcopy(self)
            # Pr for Profile
            Pr = self.xyz
        elif self.geometry.upper() == "GRID":
            self1D = DataView()
            Pr = np.zeros(shape=(nbmp, 3))
            Pr[:, 0] = np.linspace(start[0], end[0], nbmp)
            Pr[:, 1] = np.linspace(start[1], end[1], nbmp)
            Pr[:, 2] = np.linspace(start[2], end[2], nbmp)
            self1D.set_xyz(
                Pr[:, 0], Pr[:, 1], Pr[:, 2], normal=self.normal,
                geometry="profile"
            )
            self1D.eval_2D(
                self.srcLoc, self.sig, self.f, self.orientation, self.func2D
            )

        # Distance from starting point
        D = np.sqrt(
            (Pr[0, 0]-Pr[:, 0])**2 +
            (Pr[:, 1]-Pr[0, 1])**2 +
            (Pr[:, 2]-Pr[0, 2])**2
        )

        #if self.normal.upper() == "Z":
        #    self1D.set_xyz(Pr[:,0],Pr[:,1],self.z,normal=self.normal,geometry="profile")
        #elif self.normal.upper() == "Y":
        #    self1D.set_xyz(Pr[:,0],self.y,Pr[:,1],normal=self.normal,geometry="profile")
        #elif self.normal.upper() == "X":
        #    self1D.set_xyz(self.x,Pr[:,0],Pr[:,1],normal=self.normal,geometry="profile")

        pltvalue = []

        if view.upper() == "X":
            pltvalue = self1D.val_x
        elif view.upper() == "Y":
            pltvalue = self1D.val_y
        elif view.upper() == "Z":
            pltvalue = self1D.val_z

        if component.upper() == "REAL":
            ax.plot(D, pltvalue.real, color=color)
            ax.set_ylabel("E field, Real part (V/m)")
        elif component.upper() == "IMAG":
            ax.plot(D, pltvalue.imag, color=color)
            ax.set_ylabel("E field, Imag part (V/m)")
        elif component.upper() == "AMPLITUDE":
            if logamp is True:
                ax.set_yscale('log')
            ax.plot(D, np.absolute(pltvalue), color=color)
            ax.set_ylabel("E field, Amplitude (V/m)")
        elif component.upper() == "PHASE":
            ax.plot(D, phase(pltvalue), color=color)
            ax.set_ylabel("E field, Phase")

        ax.set_xlabel("Distance from startinng point (m)")

        return ax

    def plot_1D_RI_section(self, start, end, nbmp, view, ax0, ax1):

        self1D = DataView()

        # Pr for Profile
        Pr = np.zeros(shape=(nbmp, 2))
        Pr[:, 0] = np.linspace(start[0], end[0], nbmp)
        Pr[:, 1] = np.linspace(start[1], end[1], nbmp)

        # Distance from starting point
        D = np.sqrt((Pr[0, 0]-Pr[:, 0])**2+(Pr[:, 1]-Pr[0, 1])**2)

        if self.normal.upper() == "Z":
            self1D.set_xyz(
                Pr[:, 0], Pr[:, 1], self.z, normal=self.normal,
                geometry="profile"
            )
        elif self.normal.upper() == "Y":
            self1D.set_xyz(
                Pr[:, 0], self.y, Pr[:, 1], normal=self.normal,
                geometry="profile"
            )
        elif self.normal.upper() == "X":
            self1D.set_xyz(
                self.x, Pr[:, 0], Pr[:, 1], normal=self.normal,
                geometry="profile"
            )

        self1D.eval_2D(
            self.srcLoc, self.sig, self.f, self.orientation, self.func2D
        )

        if view.upper() == "X":
            ax0.plot(D, self1D.val_x.real, color="blue")
            ax1.plot(D, self1D.val_x.imag, color="red")
        elif view.upper() == "Y":
            ax0.plot(D, self1D.val_y.real, color="blue")
            ax1.plot(D, self1D.val_y.imag, color="red")
        elif view.upper() == "Z":
            ax0.plot(D, self1D.val_z.real, color="blue")
            ax1.plot(D, self1D.val_z.imag, color="red")

        ax0.set_xlabel("Distance from startinng point (m)")
        ax1.set_xlabel("Distance from startinng point (Hz)")
        ax0.set_ylabel("E field, Real part (V/m)")
        ax1.set_ylabel("E field, Imag part (V/m)")

        return ax0, ax1

    def plot_1D_AP_section(self, start, end, nbmp, view, ax0, ax1):

        self1D = copy.deepcopy(self)

        # Pr for Profile
        Pr = np.zeros(shape=(nbmp, 2))
        Pr[:, 0] = np.linspace(start[0], end[0], nbmp)
        Pr[:, 1] = np.linspace(start[1], end[1], nbmp)

        # Distance from starting point
        D = np.sqrt((Pr[0, 0]-Pr[:, 0])**2+(Pr[:, 1]-Pr[0, 1])**2)

        if self.normal.upper() == "Z":
            self1D.set_xyz(
                Pr[:, 0], Pr[:, 1], self.z, normal=self.normal,
                geometry="profile"
            )
        elif self.normal.upper() == "Y":
            self1D.set_xyz(
                Pr[:, 0], self.y, Pr[:, 1], normal=self.normal,
                geometry="profile"
            )
        elif self.normal.upper() == "X":
            self1D.set_xyz(
                self.x, Pr[:, 0], Pr[:, 1], normal=self.normal,
                geometry="profile"
            )

        self1D.eval_2D(
            self.srcLoc, self.sig, self.f, self.orientation, self.func2D
        )

        if view.upper() == "X":
            ax0.plot(D, np.absolute(self1D.val_x), color="blue")
            ax1.plot(D, phase(self1D.val_x), color="red")
        elif view.upper() == "Y":
            ax0.plot(D, np.absolute(self1D.val_y), color="blue")
            ax1.plot(D, phase(self1D.val_y), color="red")
        elif view.upper() == "Z":
            ax0.plot(D, np.absolute(self1D.val_z), color="blue")
            ax1.plot(D, phase(self1D.val_z), color="red")

        ax0.set_xlabel("Distance from startinng point (m)")
        ax1.set_xlabel("Distance from startinng point (Hz)")
        ax0.set_ylabel("E field, Amplitude (V/m)")
        ax1.set_ylabel("E field, Phase (deg)")

        return ax0, ax1

    def plot1D_FD(
        self, component="real", view="x", abscisse="Conductivity", slic=None,
        logamp=True, ax=None,legend=True, color = 'black'
    ):

        if ax is None:
            fig = plt.figure(figsize=(6.5, 5))
            ax = plt.subplot(111)

        slice_ind = 0
        if slic is None:
            slice_ind = np.minimum(len(self.sigvec), len(self.fvec))/2
            if abscisse.upper() == "CONDUCTIVITY":
                slic = self.log_fvec[slice_ind]

            elif abscisse.upper() == "FREQUENCY":
                slic = self.log_sigvec[slice_ind]

        pltvalue = []

        if view.upper() == "X":
            pltvalue = self.val_xfs
        elif view.upper() == "Y":
            pltvalue = self.val_yfs
        elif view.upper() == "Z":
            pltvalue = self.val_zfs

        if component.upper() == "REAL":
            pltvalue = pltvalue.real
            ax.set_ylabel("E field, Real part (V/m)")

        elif component.upper() == "IMAG":
            pltvalue = pltvalue.imag
            ax.set_ylabel("E field, Imag part (V/m)")
        elif component.upper() == "AMPLITUDE":
            pltvalue = np.absolute(pltvalue)
            ax.set_ylabel("E field, Amplitude (V/m)")
            if logamp == True:
                ax.set_yscale('log')
        elif component.upper() == "PHASE":
            pltvalue = phase(pltvalue)
            ax.set_ylabel("E field, Phase")

        if component.upper() == "PHASOR":
            if abscisse.upper() == "CONDUCTIVITY":
                slice_ind = np.where( slic == self.log_fvec)[0][0]
                ax.plot(
                    pltvalue.real[:, slice_ind], pltvalue.imag[:, slice_ind],
                    color=color
                )
                ax.set_xlabel("E field, Real part (V/m)")
                ax.set_ylabel("E field, Imag part(V/m)")

                axymin = pltvalue.imag[:, slice_ind].min()
                axymax = pltvalue.imag[:, slice_ind].max()

                if legend:
                    ax.annotate(
                        ("f =%0.5f Hz") % (self.fvec[slice_ind]),
                        xy=(
                            (
                                pltvalue.real[:, slice_ind].min() +
                                pltvalue.real[:, slice_ind].max()
                            )/2.,
                            axymin+(axymax-axymin)/4.
                            ),
                        xycoords='data',
                        xytext=(
                            (
                                pltvalue.real[:, slice_ind].min() +
                                pltvalue.real[:, slice_ind].max()
                            )/2.,
                            axymin+(axymax-axymin)/4.),
                        textcoords='data',
                        fontsize=14.
                    )

            elif abscisse.upper() == "FREQUENCY":
                slice_ind = np.where(slic == self.log_sigvec)[0][0]
                ax.plot(
                    pltvalue.real[slice_ind, :], pltvalue.imag[slice_ind, :],
                    color=color
                )
                ax.set_xlabel("E field, Real part (V/m)")
                ax.set_ylabel("E field, Imag part(V/m)")

                axymin = pltvalue.imag[slice_ind, :].min()
                axymax = pltvalue.imag[slice_ind, :].max()

                if legend:
                    ax.annotate(
                        ("$\sigma$ =%0.5f S/m") % (self.sigvec[slice_ind]),
                        xy=(
                            (
                                pltvalue.real[slice_ind, :].min() +
                                pltvalue.real[slice_ind, :].max())/2.,
                            axymin+(axymax-axymin)/4.
                        ),
                        xycoords='data',
                        xytext=(
                            (
                                pltvalue.real[slice_ind, :].min() +
                                pltvalue.real[slice_ind, :].max()
                            )/2.,
                            axymin+(axymax-axymin)/4.),
                        textcoords='data',
                        fontsize=14.
                    )

        else:
            if abscisse.upper() == "CONDUCTIVITY":
                ax.set_xlabel("Conductivity (S/m)")
                ax.set_xscale('log')
                slice_ind = np.where( slic == self.log_fvec)[0][0]
                ax.plot(self.sigvec, pltvalue[:, slice_ind], color=color)

                axymin = pltvalue[:, slice_ind].min()
                axymax = pltvalue[:, slice_ind].max()
                if legend:
                    ax.annotate(
                        ("f =%0.5f Hz") % (self.fvec[slice_ind]),
                        xy=(
                            10.**(
                                (
                                    np.log10(self.sigvec.min()) +
                                    np.log10(self.sigvec.max())

                            )/2
                            ),
                            axymin+(axymax-axymin)/4.),
                        xycoords='data',
                        xytext=(
                            10.**(
                                (
                                    np.log10(self.sigvec.min()) +
                                    np.log10(self.sigvec.max())

                            )/2
                            ),
                            axymin+(axymax-axymin)/4.),
                        textcoords='data',
                        fontsize=14.
                    )

            elif abscisse.upper() == "FREQUENCY":
                ax.set_xlabel("Frequency (Hz)")
                ax.set_xscale('log')
                slice_ind = np.where( slic == self.log_sigvec)[0][0]
                ax.plot(self.fvec, pltvalue[slice_ind, :], color=color)

                axymin = pltvalue[slice_ind, :].min()
                axymax = pltvalue[slice_ind, :].max()
                if legend:
                    ax.annotate(
                        ("$\sigma$ =%0.5f S/m") % (self.sigvec[slice_ind]),
                        xy=(
                            10.**(
                                (
                                    np.log10(self.fvec.min()) +
                                    np.log10(self.fvec.max())
                                )/2
                            ),
                            axymin+(axymax-axymin)/4.
                        ),
                        xycoords='data',
                        xytext=(
                            10.**(
                                (
                                    np.log10(self.fvec.min()) +
                                    np.log10(self.fvec.max())

                            )/2
                            ),
                            axymin+(axymax-axymin)/4.
                        ),
                        textcoords='data',
                        fontsize=14.)

        return ax

    def plot_1D_RI_f_x(self, absloc, coordloc, ax0, ax1, sigind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax0.set_xlabel("Frequency (Hz)")
        ax1.set_xlabel("Frequency (Hz)")
        ax0.set_ylabel("E field, Real part (V/m)")
        ax1.set_ylabel("E field, Imag part (V/m)")

        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_xfs.real[:, freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_xfs.real[:, freqind])),linestyle="dashed", color="black",linewidth=3.0)

        ax0.plot(self.fvec, self.val_xfs.real[sigind, :], color="blue")
        ax1.plot(self.fvec, self.val_xfs.imag[sigind, :], color="red")

        ax0ymin = self.val_xfs.real[sigind, :].min()
        ax0ymax = self.val_xfs.real[sigind, :].max()

        ax1ymin = self.val_xfs.imag[sigind, :].min()
        ax1ymax = self.val_xfs.imag[sigind, :].max()

        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(
            ("$\sigma$ =%0.5f S/m") % (self.sigvec[sigind]),
            xy=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())

                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        ax1.annotate(
            ("$\sigma$ =%0.5f S/m") % (self.sigvec[sigind]),
            xy=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max()))/2
                    ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max()))/2
                    ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
        ),
            textcoords='data',
            fontsize=14.)

        return ax0, ax1

    def plot_1D_AP_f_x(self, absloc, coordloc, ax0, ax1, sigind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax0.set_xlabel("Frequency (Hz)")
        ax1.set_xlabel("Frequency (Hz)")
        ax0.set_ylabel("E field, Amplitude (V/m)")
        ax1.set_ylabel("E field, Phase (deg)")

        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_xfs.real[:, freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_xfs.real[:, freqind])),linestyle="dashed", color="black",linewidth=3.0)

        ax0.plot(self.fvec, np.absolute(self.val_xfs[sigind, :]), color="blue")
        ax1.plot(self.fvec, phase(self.val_xfs[sigind, :]), color="red")

        ax0ymin = np.absolute(self.val_xfs[sigind, :]).min()
        ax0ymax = np.absolute(self.val_xfs[sigind, :]).max()
        ax1ymin = phase(self.val_xfs[sigind, :]).min()
        ax1ymax = phase(self.val_xfs[sigind, :]).max()

        #ax2.plot(self.fvec[freqind]*np.ones_like(self.val_xfs[sigind, :]),
        #            np.linspace(ax2ymin,ax2ymax,len(self.val_xfs[sigind, :])),linestyle="dashed", color="black",linewidth=3.0)

        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(
            ("$\sigma$ =%0.5f S/m") % (self.sigvec[sigind]),
            xy=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max()))/2
                    ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max()))/2
                    ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
                ),
            textcoords='data',
            fontsize=14.
        )

        ax1.annotate(
            ("$\sigma$ =%0.5f S/m") % (self.sigvec[sigind]),
            xy=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max()))/2
                    ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            textcoords='data',
            fontsize=14.)

        return ax0, ax1

    def plot_1D_RI_sig_x(self, absloc, coordloc, ax0, ax1, freqind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax0.set_xlabel("Conductivity (S/m)")
        ax1.set_xlabel("Conductivity (S/m)")
        ax0.set_ylabel("E field, Real part (V/m)")
        ax1.set_ylabel("E field, Imag part (V/m)")

        ax0.plot(self.sigvec, self.val_xfs.real[:, freqind], color="blue")
        ax1.plot(self.sigvec, self.val_xfs.imag[:, freqind], color="red")

        ax0ymin = self.val_xfs.real[:, freqind].min()
        ax0ymax = self.val_xfs.real[:, freqind].max()
        ax1ymin = self.val_xfs.imag[:, freqind].min()
        ax1ymax = self.val_xfs.imag[:, freqind].max()

        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_xfs.real[:, freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_xfs.real[:, freqind])),linestyle="dashed", color="black",linewidth=3.0)

        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(
            ("f =%0.5f Hz") % (self.fvec[freqind]),
            xy=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        ax1.annotate(
            ("f =%0.5f Hz") % (self.fvec[freqind]),
            xy=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max()))/2
                    ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        return ax0, ax1

    def plot_1D_AP_sig_x(self, absloc, coordloc, ax0, ax1, freqind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax0.set_xlabel("Conductivity (S/m)")
        ax1.set_xlabel("Conductivity (S/m)")
        ax0.set_ylabel("E field, Amplitude (V/m)")
        ax1.set_ylabel("E field, Phase (deg)")

        ax0.plot(
            self.sigvec, np.absolute(self.val_xfs[:, freqind]), color="blue"
        )
        ax1.plot(self.sigvec, phase(self.val_xfs[:, freqind]), color="red")

        ax0ymin = np.absolute(self.val_xfs[:, freqind]).min()
        ax0ymax = np.absolute(self.val_xfs[:, freqind]).max()
        ax1ymin = phase(self.val_xfs[:, freqind]).min()
        ax1ymax = phase(self.val_xfs[:, freqind]).max()

        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_xfs[:, freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_xfs[:, freqind])),linestyle="dashed", color="black",linewidth=3.0)

        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(
            ("f =%0.5f Hz") % (self.fvec[freqind]),
            xy=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        ax1.annotate(
            ("f =%0.5f Hz") % (self.fvec[freqind]),
            xy=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        return ax0, ax1

    def plot_1D_phasor_f_x(self, absloc, coordloc, ax, sigind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax.plot(self.val_xfs.real[sigind, :], self.val_xfs.imag[sigind, :])

    def plot_1D_phasor_sig_x(self, absloc, coordloc, ax, freqind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax.plot(self.val_xfs.real[:, freqind], self.val_xfs.imag[:, freqind])

    def plot_1D_RI_f_y(self, absloc, coordloc, ax0, ax1, sigind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax0.set_xlabel("Frequency (Hz)")
        ax1.set_xlabel("Frequency (Hz)")
        ax0.set_ylabel("E field, Real part (V/m)")
        ax1.set_ylabel("E field, Imag part (V/m)")

        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_xfs.real[:, freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_xfs.real[:, freqind])),linestyle="dashed", color="black",linewidth=3.0)

        ax0.plot(self.fvec, self.val_yfs.real[sigind, :], color="blue")
        ax1.plot(self.fvec, self.val_yfs.imag[sigind, :], color="red")

        ax0ymin = self.val_yfs.real[sigind, :].min()
        ax0ymax = self.val_yfs.real[sigind, :].max()

        ax1ymin = self.val_yfs.imag[sigind, :].min()
        ax1ymax = self.val_yfs.imag[sigind, :].max()

        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(
            ("$\sigma$ =%0.5f S/m") % (self.sigvec[sigind]),
            xy=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        ax1.annotate(
            ("$\sigma$ =%0.5f S/m") % (self.sigvec[sigind]),
            xy=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        return ax0, ax1

    def plot_1D_AP_f_y(self, absloc, coordloc, ax0, ax1, sigind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax0.set_xlabel("Frequency (Hz)")
        ax1.set_xlabel("Frequency (Hz)")
        ax0.set_ylabel("E field, Amplitude (V/m)")
        ax1.set_ylabel("E field, Phase (deg)")

        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_yfs.real[:, freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_yfs.real[:, freqind])),linestyle="dashed", color="black",linewidth=3.0)

        ax0.plot(self.fvec, np.absolute(self.val_yfs[sigind, :]), color="blue")
        ax1.plot(self.fvec, phase(self.val_yfs[sigind, :]), color="red")

        ax0ymin, ax0ymax = (
            np.absolute(self.val_yfs[sigind, :]).min(),
            np.absolute(self.val_yfs[sigind, :]).max()
        )
        ax1ymin, ax1ymax = (
            phase(self.val_yfs[sigind, :]).min(),
            phase(self.val_yfs[sigind, :]).max()
        )

        #ax2.plot(self.fvec[freqind]*np.ones_like(self.val_yfs[sigind, :]),
        #            np.linspace(ax2ymin,ax2ymax,len(self.val_yfs[sigind, :])),linestyle="dashed", color="black",linewidth=3.0)

        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(
            ("$\sigma$ =%0.5f S/m") % (self.sigvec[sigind]),
            xy=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            textcoords='data',
            fontsize=14.)

        ax1.annotate(
            ("$\sigma$ =%0.5f S/m") % (self.sigvec[sigind]),
            xy=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            textcoords='data',
            fontsize=14.)

        return ax0, ax1

    def plot_1D_RI_sig_y(self,absloc,coordloc,ax0, ax1,freqind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax0.set_xlabel("Conductivity (S/m)")
        ax1.set_xlabel("Conductivity (S/m)")
        ax0.set_ylabel("E field, Real part (V/m)")
        ax1.set_ylabel("E field, Imag part (V/m)")

        ax0.plot(self.sigvec,self.val_yfs.real[:, freqind], color="blue")
        ax1.plot(self.sigvec,self.val_yfs.imag[:, freqind], color="red")

        ax0ymin, ax0ymax = self.val_yfs.real[:, freqind].min(),self.val_yfs.real[:, freqind].max()
        ax1ymin, ax1ymax = self.val_yfs.imag[:, freqind].min(),self.val_yfs.imag[:, freqind].max()

        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_yfs.real[:, freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_yfs.real[:, freqind])),linestyle="dashed", color="black",linewidth=3.0)

        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(
            ("f =%0.5f Hz") % (self.fvec[freqind]),
            xy=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            textcoords='data',
           fontsize=14.)

        ax1.annotate(
            ("f =%0.5f Hz") % (self.fvec[freqind]),
            xy=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            textcoords='data',
           fontsize=14.)

        return ax0, ax1


    def plot_1D_AP_sig_y(self,absloc,coordloc,ax0, ax1,freqind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax0.set_xlabel("Conductivity (S/m)")
        ax1.set_xlabel("Conductivity (S/m)")
        ax0.set_ylabel("E field, Amplitude (V/m)")
        ax1.set_ylabel("E field, Phase (deg)")

        ax0.plot(self.sigvec,np.absolute(self.val_yfs[:, freqind]), color="blue")
        ax1.plot(self.sigvec,phase(self.val_yfs[:, freqind]), color="red")

        ax0ymin, ax0ymax = np.absolute(self.val_yfs[:, freqind]).min(),np.absolute(self.val_yfs[:, freqind]).max()
        ax1ymin, ax1ymax = phase(self.val_yfs[:, freqind]).min(),phase(self.val_yfs[:, freqind]).max()

        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_yfs[:, freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_yfs[:, freqind])),linestyle="dashed", color="black",linewidth=3.0)

        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(
            ("f =%0.5f Hz") % (self.fvec[freqind]),
            xy=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        ax1.annotate(
            ("f =%0.5f Hz") % (self.fvec[freqind]),
            xy=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        return ax0, ax1

    def plot_1D_phasor_f_y(self, absloc, coordloc, ax, sigind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax.plot(self.val_yfs.real[sigind, :], self.val_yfs.imag[sigind, :])

    def plot_1D_phasor_sig_y(self, absloc, coordloc, ax, freqind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax.plot(self.val_yfs.real[:, freqind], self.val_yfs.imag[:, freqind])

    def plot_1D_RI_f_z(self, absloc, coordloc, ax0, ax1, sigind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax0.set_xlabel("Frequency (Hz)")
        ax1.set_xlabel("Frequency (Hz)")
        ax0.set_ylabel("E field, Real part (V/m)")
        ax1.set_ylabel("E field, Imag part (V/m)")

        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_xfs.real[:, freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_xfs.real[:, freqind])),linestyle="dashed", color="black",linewidth=3.0)

        ax0.plot(self.fvec, self.val_zfs.real[sigind, :], color="blue")
        ax1.plot(self.fvec, self.val_zfs.imag[sigind, :], color="red")

        ax0ymin, ax0ymax = (
            self.val_zfs.real[sigind, :].min(),
            self.val_zfs.real[sigind, :].max()
        )

        ax1ymin, ax1ymax = (
            self.val_zfs.imag[sigind, :].min(),
            self.val_zfs.imag[sigind, :].max()
        )

        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(
            ("$\sigma$ =%0.5f S/m") % (self.sigvec[sigind]),
            xy=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        ax1.annotate(
            ("$\sigma$ =%0.5f S/m") % (self.sigvec[sigind]),
            xy=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max()))/2
                    ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        return ax0, ax1

    def plot_1D_AP_f_z(self, absloc, coordloc, ax0, ax1, sigind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax0.set_xlabel("Frequency (Hz)")
        ax1.set_xlabel("Frequency (Hz)")
        ax0.set_ylabel("E field, Amplitude (V/m)")
        ax1.set_ylabel("E field, Phase (deg)")

        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_zfs.real[:, freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_zfs.real[:, freqind])),linestyle="dashed", color="black",linewidth=3.0)

        ax0.plot(self.fvec, np.absolute(self.val_zfs[sigind, :]), color="blue")
        ax1.plot(self.fvec, phase(self.val_zfs[sigind, :]), color="red")

        ax0ymin, ax0ymax = (
            np.absolute(self.val_zfs[sigind, :]).min(),
            np.absolute(self.val_zfs[sigind, :]).max()
        )
        ax1ymin, ax1ymax = (
            phase(self.val_zfs[sigind, :]).min(),
            phase(self.val_zfs[sigind, :]).max()
        )

        #ax2.plot(self.fvec[freqind]*np.ones_like(self.val_zfs[sigind, :]),
        #            np.linspace(ax2ymin,ax2ymax,len(self.val_zfs[sigind, :])),linestyle="dashed", color="black",linewidth=3.0)

        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(
            ("$\sigma$ =%0.5f S/m") % (self.sigvec[sigind]),
            xy=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        ax1.annotate(
            ("$\sigma$ =%0.5f S/m") % (self.sigvec[sigind]),
            xy=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        return ax0, ax1

    def plot_1D_RI_sig_z(self, absloc, coordloc, ax0, ax1, freqind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax0.set_xlabel("Conductivity (S/m)")
        ax1.set_xlabel("Conductivity (S/m)")
        ax0.set_ylabel("E field, Real part (V/m)")
        ax1.set_ylabel("E field, Imag part (V/m)")

        ax0.plot(self.sigvec, self.val_zfs.real[:, freqind], color="blue")
        ax1.plot(self.sigvec, self.val_zfs.imag[:, freqind], color="red")

        ax0ymin = self.val_zfs.real[:, freqind].min()
        ax0ymax = self.val_zfs.real[:, freqind].max()
        ax1ymin = self.val_zfs.imag[:, freqind].min()
        ax1ymax = self.val_zfs.imag[:, freqind].max()

        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_zfs.real[:, freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_zfs.real[:, freqind])),linestyle="dashed", color="black",linewidth=3.0)

        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(
            ("f =%0.5f Hz") % (self.fvec[freqind]),
            xy=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max()))/2
                ), ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        ax1.annotate(
            ("f =%0.5f Hz") % (self.fvec[freqind]),
            xy=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max()))/2
                ), ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        return ax0, ax1

    def plot_1D_AP_sig_z(self, absloc, coordloc, ax0, ax1, freqind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax0.set_xlabel("Conductivity (S/m)")
        ax1.set_xlabel("Conductivity (S/m)")
        ax0.set_ylabel("E field, Amplitude (V/m)")
        ax1.set_ylabel("E field, Phase (deg)")

        ax0.plot(
            self.sigvec, np.absolute(self.val_zfs[:, freqind]), color="blue"
        )
        ax1.plot(self.sigvec, phase(self.val_zfs[:, freqind]), color="red")

        ax0ymin, ax0ymax = (
            np.absolute(self.val_zfs[:, freqind]).min(),
            np.absolute(self.val_zfs[:, freqind]).max()
        )
        ax1ymin, ax1ymax = (
            phase(self.val_zfs[:, freqind]).min(),
            phase(self.val_zfs[:, freqind]).max()
        )

        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_zfs[:, freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_zfs[:, freqind])),linestyle="dashed", color="black",linewidth=3.0)

        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(
            ("f =%0.5f Hz") % (self.fvec[freqind]),
            xy=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        ax1.annotate(
            ("f =%0.5f Hz") % (self.fvec[freqind]),
            xy=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax1ymin+(ax1ymax-ax1ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        return ax0, ax1

    def plot_1D_phasor_f_z(self, absloc, coordloc, ax, sigind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax.plot(self.val_zfs.real[sigind, :], self.val_zfs.imag[sigind, :])

    def plot_1D_phasor_sig_z(self, absloc, coordloc, ax, freqind):

        if self.normal.upper() == "Z":
            obsLoc = np.c_[absloc, coordloc, self.z]
        elif self.normal.upper() == "Y":
            obsLoc = np.c_[absloc, self.y, coordloc]
        elif self.normal.upper() == "X":
            obsLoc = np.c_[self.x, absloc, coordloc]

        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        ax.plot(self.val_zfs.real[:, freqind], self.val_zfs.imag[:, freqind])

    def plot_1D_x(self, obslocx, obslocy, obslocz, sigind, freqind, mode):

        #sigind = np.where( sigplt == self.sigvec)[0][0]
        #freqind = np.where( freqplt == self.fvec)[0][0]

        fig = plt.figure(figsize=(14, 5))
        ax0 = plt.subplot(121)
        ax2 = plt.subplot(122)

        ax1 = ax0.twinx()
        ax3 = ax2.twinx()

        obsLoc = np.c_[obslocx, obslocy, obslocz]
        self.eval_loc(
            self.srcLoc, obsLoc, self.sigvec, self.fvec, self.orientation,
            self.func1D
        )

        if mode == "RI":

            ax0.set_xlabel("Conductivity (S/m)")
            ax0.set_ylabel("E field, Real part (V/m)")
            ax1.set_ylabel("E field, Imag part (V/m)")

            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("E field, Real part (V/m)")
            ax3.set_ylabel("E field, Imag part (V/m)")

            ax0.plot(self.sigvec, self.val_xfs.real[:, freqind], color="blue")
            ax1.plot(self.sigvec, self.val_xfs.imag[:, freqind], color="red")

            ax0ymin = self.val_xfs.real[:, freqind].min()
            ax0ymax = self.val_xfs.real[:, freqind].max()

            ax0.plot(
                self.sigvec[sigind]*np.ones_like(
                    self.val_xfs.real[:, freqind]
                ),
                np.linspace(
                    ax0ymin, ax0ymax, len(self.val_xfs.real[:, freqind])
                ),
                linestyle="dashed", color="black", linewidth=3.0
            )

            ax0.set_ylim(ax0ymin, ax0ymax)

            ax2.plot(self.fvec, self.val_xfs.real[sigind, :], color="blue")
            ax3.plot(self.fvec, self.val_xfs.imag[sigind, :], color="red")

            ax2ymin = self.val_xfs.real[sigind, :].min()
            ax2ymax = self.val_xfs.real[sigind, :].max()

            ax2.plot(
                self.fvec[freqind]*np.ones_like(self.val_xfs.imag[sigind, :]),
                np.linspace(
                    ax2ymin, ax2ymax, len(self.val_xfs.imag[sigind, :])
                ),
                linestyle="dashed", color="black", linewidth=3.0
            )

            ax2.set_ylim(ax2ymin, ax2ymax)

        elif mode == "AP":

            ax0.set_xlabel("Conductivity (S/m)")
            ax0.set_ylabel("E field, Amplitude (V/m)")
            ax1.set_ylabel("E field, Phase (deg)")

            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("E field, Amplitude (V/m)")
            ax3.set_ylabel("E field, Phase (deg)")

            ax0.plot(
                self.sigvec, np.absolute(self.val_xfs[:, freqind]),
                color="blue"
            )
            ax1.plot(
                self.sigvec, phase(self.val_xfs[:, freqind]), color="red"
            )

            ax0ymin = np.absolute(self.val_xfs[:, freqind]).min()
            ax0ymax = np.absolute(self.val_xfs[:, freqind]).max()

            ax0.plot(
                self.sigvec[sigind]*np.ones_like(self.val_xfs[:, freqind]),
                np.linspace(ax0ymin, ax0ymax, len(self.val_xfs[:, freqind])),
                linestyle="dashed", color="black", linewidth=3.0
            )

            ax0.set_ylim(ax0ymin, ax0ymax)

            ax2.plot(
                self.fvec, np.absolute(self.val_xfs[sigind, :]), color="blue"
            )
            ax3.plot(self.fvec, phase(self.val_xfs[sigind, :]), color="red")

            ax2ymin = np.absolute(self.val_xfs[sigind, :]).min()
            ax2ymax = np.absolute(self.val_xfs[sigind, :]).max()

            ax2.plot(
                self.fvec[freqind]*np.ones_like(self.val_xfs[sigind, :]),
                np.linspace(ax2ymin, ax2ymax, len(self.val_xfs[sigind, :])),
                linestyle="dashed", color="black", linewidth=3.0
            )

            ax2.set_ylim(ax2ymin, ax2ymax)

        #ax0.set_yscale('log')
        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax2.set_xscale('log')
        ax3.set_xscale('log')

        ax0.annotate(
            ("f =%0.5f Hz") % (self.fvec[freqind]),
            xy=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.sigvec.min()) +
                        np.log10(self.sigvec.max())
                    )/2
                ),
                ax0ymin+(ax0ymax-ax0ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        ax2.annotate(
            ("$\sigma$ =%0.5f S/m") % (self.sigvec[sigind]),
            xy=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax2ymin+(ax2ymax-ax2ymin)/4.
            ),
            xycoords='data',
            xytext=(
                10.**(
                    (
                        np.log10(self.fvec.min()) +
                        np.log10(self.fvec.max())
                    )/2
                ),
                ax2ymin+(ax2ymax-ax2ymin)/4.
            ),
            textcoords='data',
            fontsize=14.
        )

        plt.tight_layout()

    def plot_1D_y(self, sigind, freqind, mode):

        #sigind = np.where( sigplt == self.sigvec)[0][0]
        #freqind = np.where( freqplt == self.fvec)[0][0]

        fig = plt.figure(figsize=(14, 5))
        ax0 = plt.subplot(121)
        ax2 = plt.subplot(122)

        ax1 = ax0.twinx()
        ax3 = ax2.twinx()

        if mode == "RI":

            ax0.set_xlabel("Conductivity (S/m)")
            ax0.set_ylabel("E field, Real part (V/m)")
            ax1.set_ylabel("E field, Imag part (V/m)")

            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("E field, Real part (V/m)")
            ax3.set_ylabel("E field, Imag part (V/m)")

            ax0.plot(self.sigvec, self.val_yfs.real[:, freqind], color="blue")
            ax1.plot(self.sigvec, self.val_yfs.imag[:, freqind], color="red")

            ax0ymin = self.val_yfs.real[:, freqind].min()
            ax0ymax = self.val_yfs.real[:, freqind].max()

            ax0.plot(
                self.sigvec[sigind]*np.ones_like(
                    self.val_yfs.real[:, freqind]
                ),
                np.linspace(
                    ax0ymin, ax0ymax, len(self.val_yfs.real[:, freqind])
                ),
                linestyle="dashed", color="black", linewidth=3.0
            )

            ax0.set_ylim(ax0ymin, ax0ymax)

            ax2.plot(self.fvec, self.val_yfs.real[sigind, :], color="blue")
            ax3.plot(self.fvec, self.val_yfs.imag[sigind, :], color="red")

            ax2ymin = self.val_yfs.real[sigind, :].min()
            ax2ymax = self.val_yfs.real[sigind, :].max()

            ax2.plot(
                self.fvec[freqind]*np.ones_like(self.val_yfs.imag[sigind, :]),
                np.linspace(
                    ax2ymin, ax2ymax, len(self.val_yfs.imag[sigind, :])
                ), linestyle="dashed", color="black", linewidth=3.0
            )

            ax2.set_ylim(ax2ymin, ax2ymax)

        elif mode == "AP":

            ax0.set_xlabel("Conductivity (S/m)")
            ax0.set_ylabel("E field, Amplitude (V/m)")
            ax1.set_ylabel("E field, Phase (deg)")

            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("E field, Amplitude (V/m)")
            ax3.set_ylabel("E field, Phase (deg)")

            ax0.plot(
                self.sigvec, np.absolute(self.val_yfs[:, freqind]), color="blue"
            )
            ax1.plot(
                self.sigvec, phase(self.val_yfs[:, freqind]), color="red"
            )

            ax0ymin, ax0ymax = (
                np.absolute(self.val_yfs[:, freqind]).min(),
                np.absolute(self.val_yfs[:, freqind]).max()
            )

            ax0.plot(
                self.sigvec[sigind]*np.ones_like(self.val_yfs[:, freqind]),
                np.linspace(ax0ymin, ax0ymax, len(self.val_yfs[:, freqind])),
                linestyle="dashed", color="black", linewidth=3.0
            )

            ax0.set_ylim(ax0ymin, ax0ymax)

            ax2.plot(
                self.fvec, np.absolute(self.val_yfs[sigind, :]), color="blue"
            )
            ax3.plot(
                self.fvec, phase(self.val_yfs[sigind, :]), color="red"
            )

            ax2ymin = np.absolute(self.val_yfs[sigind, :]).min()
            ax2ymax = np.absolute(self.val_yfs[sigind, :]).max()

            ax2.plot(
                self.fvec[freqind]*np.ones_like(self.val_yfs[sigind, :]),
                np.linspace(ax2ymin, ax2ymax, len(self.val_yfs[sigind, :])),
                linestyle="dashed", color="black", linewidth=3.0
            )

            ax2.set_ylim(ax2ymin, ax2ymax)


        #ax0.set_yscale('log')
        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax2.set_xscale('log')
        ax3.set_xscale('log')

        #ax0.annotate(
        # ("$\f$ =%5.5f Hz") % (self.fvec[freqind]*10**(3)),
        #    xy=(self.sigvec.max()/100., ax0ymin+(ax0ymax-ax0ymin)/4.), xycoords='data',
        #    xytext=(self.sigvec.max()/100.,
        # ax0ymin+(ax0ymax-ax0ymin)/4.
        # ),
        # textcoords='data',
        #   fontsize=14.)

        #ax2.annotate(
        # ("$\sigma$ =%3.3f mS/m") % (self.sigvec[sigind]*10**(3)),
        #    xy=(self.fvec.max()/100., ax2ymin+(ax2ymax-ax2ymin)/4.), xycoords='data',
        #    xytext=(self.fvec.max()/100.,
        # ax2ymin+(ax2ymax-ax2ymin)/4.
        # ),
        # textcoords='data',
        #    fontsize=14.)
        plt.tight_layout()

    def plot_1D_z(self, sigind, freqind, mode):

        #sigind = np.where( sigplt == self.sigvec)[0][0]
        #freqind = np.where( freqplt == self.fvec)[0][0]

        fig = plt.figure(figsize=(14, 5))
        ax0 = plt.subplot(121)
        ax2 = plt.subplot(122)

        ax1 = ax0.twinx()
        ax3 = ax2.twinx()

        if mode == "RI":

            ax0.set_xlabel("Conductivity (S/m)")
            ax0.set_ylabel("E field, Real part (V/m)")
            ax1.set_ylabel("E field, Imag part (V/m)")

            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("E field, Real part (V/m)")
            ax3.set_ylabel("E field, Imag part (V/m)")

            ax0.plot(self.sigvec, self.val_zfs.real[:, freqind], color="blue")
            ax1.plot(self.sigvec, self.val_zfs.imag[:, freqind], color="red")

            ax0ymin = self.val_zfs.real[:, freqind].min()
            ax0ymax = self.val_zfs.real[:, freqind].max()

            ax0.plot(
                self.sigvec[sigind]*np.ones_like(
                    self.val_zfs.real[:, freqind]
                ),
                np.linspace(
                    ax0ymin, ax0ymax, len(self.val_zfs.real[:, freqind])
                ),
                linestyle="dashed",
                color="black",
                linewidth=3.0
            )

            ax0.set_ylim(ax0ymin, ax0ymax)

            ax2.plot(self.fvec, self.val_zfs.real[sigind, :], color="blue")
            ax3.plot(self.fvec, self.val_zfs.imag[sigind, :], color="red")

            ax2ymin = self.val_zfs.real[sigind, :].min()
            ax2ymax = self.val_zfs.real[sigind, :].max()

            ax2.plot(
                self.fvec[freqind]*np.ones_like(self.val_zfs.imag[sigind, :]),
                np.linspace(
                    ax2ymin, ax2ymax, len(self.val_zfs.imag[sigind, :])
                ),
                linestyle="dashed", color="black", linewidth=3.0
            )

            ax2.set_ylim(ax2ymin, ax2ymax)

        elif mode == "AP":

            ax0.set_xlabel("Conductivity (S/m)")
            ax0.set_ylabel("E field, Amplitude (V/m)")
            ax1.set_ylabel("E field, Phase (deg)")

            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("E field, Amplitude (V/m)")
            ax3.set_ylabel("E field, Phase (deg)")

            ax0.plot(
                self.sigvec, np.absolute(self.val_zfs[:, freqind]),
                color="blue"
            )
            ax1.plot(
                self.sigvec, phase(self.val_zfs[:, freqind]), color="red"
            )

            ax0ymin = np.absolute(self.val_zfs[:, freqind]).min()
            ax0ymax = np.absolute(self.val_zfs[:, freqind]).max()

            ax0.plot(
                self.sigvec[sigind]*np.ones_like(self.val_zfs[:, freqind]),
                np.linspace(ax0ymin, ax0ymax, len(self.val_zfs[:, freqind])),
                linestyle="dashed",
                color="black",
                linewidth=3.0
            )

            ax0.set_ylim(ax0ymin, ax0ymax)

            ax2.plot(
                self.fvec, np.absolute(self.val_zfs[sigind, :]), color="blue"
            )
            ax3.plot(
                self.fvec, phase(self.val_zfs[sigind, :]), color="red"
            )

            ax2ymin = np.absolute(self.val_zfs[sigind, :]).min()
            ax2ymax = np.absolute(self.val_zfs[sigind, :]).max()

            ax2.plot(
                self.fvec[freqind]*np.ones_like(self.val_zfs[sigind, :]),
                np.linspace(ax2ymin, ax2ymax, len(self.val_zfs[sigind, :])),
                linestyle="dashed", color="black", linewidth=3.0
            )

            ax2.set_ylim(ax2ymin, ax2ymax)

        #ax0.set_yscale('log')
        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax2.set_xscale('log')
        ax3.set_xscale('log')

        #ax0.annotate(
        # ("$\f$ =%5.5f Hz") % (self.fvec[freqind]*10**(3)),
        #    xy=(self.sigvec.max()/100., ax0ymin+(ax0ymax-ax0ymin)/4.), xycoords='data',
        #    xytext=(self.sigvec.max()/100.,
        # ax0ymin+(ax0ymax-ax0ymin)/4.
        # ),
        # textcoords='data',
        #   fontsize=14.)

        #ax2.annotate(
        # ("$\sigma$ =%3.3f mS/m") % (self.sigvec[sigind]*10**(3)),
        #    xy=(self.fvec.max()/100., ax2ymin+(ax2ymax-ax2ymin)/4.), xycoords='data',
        #    xytext=(self.fvec.max()/100.,
        # ax2ymin+(ax2ymax-ax2ymin)/4.
        # ),
        # textcoords='data',
        #    fontsize=14.)

        plt.tight_layout()


