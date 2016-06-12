import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.size"] = 13

def phase(z):
#     val = np.arctan2(z.real, z.imag)
    val = np.angle(z)
    return val

class DataView(object):
    """
        Provides viewing functions for Data
        This can be inherited by XXX
    """
    def __init__(self):
        pass

    def set_xyz(self, x, y, z, normal="Z"):
        self.normal = normal
        if normal =="X" or normal=="x":
            self.x, self.y, self.z = x, y, z
            self.ncx, self.ncy, self.ncz = 1, y.size, z.size
            self.Y, self.Z = np.meshgrid(y, z)
            self.xyz = np.c_[x*np.ones(self.ncy*self.ncz), self.Y.flatten(), self.Z.flatten()]

        elif normal =="Y" or normal =="y":
            self.x, self.y, self.z = x, y, z
            self.ncx, self.ncy, self.ncz = x.size, 1, z.size
            self.X, self.Z = np.meshgrid(x, z)
            self.xyz = np.c_[self.X.flatten(), y*np.ones(self.ncx*self.ncz), self.Z.flatten()]

        elif normal =="Z" or normal =="z":
            self.x, self.y, self.z = x, y, z
            self.ncx, self.ncy, self.ncz = x.size, y.size, 1
            self.X, self.Y = np.meshgrid(x, y)
            self.xyz = np.c_[self.X.flatten(), self.Y.flatten(), z*np.ones(self.ncx*self.ncy)]

    def eval_2D(self, srcLoc, sig, f, orientation, func):
        self.val_x, self.val_y, self.val_z = func(self.xyz, srcLoc, sig, f, orientation=orientation)
        if self.normal =="X" or self.normal=="x":
            Freshape = lambda v: v.reshape(self.ncy, self.ncz)
            self.VAL_X, self.VAL_Y, self.VAL_Z = Freshape(self.val_x), Freshape(self.val_y), Freshape(self.val_z)
            self.VEC_R_amp = np.sqrt(self.VAL_Z.real**2+self.VAL_Y.real**2)
            self.VEC_I_amp = np.sqrt(self.VAL_Z.imag**2+self.VAL_Y.imag**2)
            self.VEC_A_amp = np.sqrt(np.abs(self.VAL_Z)**2+np.abs(self.VAL_Y)**2)
            self.VEC_P_amp = np.sqrt(phase(self.VAL_Z)**2+phase(self.VAL_Y)**2)

        elif self.normal =="Y" or self.normal =="y":
            Freshape = lambda v: v.reshape(self.ncx, self.ncz)
            self.VAL_X, self.VAL_Y, self.VAL_Z = Freshape(self.val_x), Freshape(self.val_y), Freshape(self.val_z)
            self.VEC_R_amp = np.sqrt(self.VAL_X.real**2+self.VAL_Z.real**2)
            self.VEC_I_amp = np.sqrt(self.VAL_X.imag**2+self.VAL_Z.imag**2)
            self.VEC_A_amp = np.sqrt(np.abs(self.VAL_X)**2+np.abs(self.VAL_Z)**2)
            self.VEC_P_amp = np.sqrt(phase(self.VAL_X)**2+phase(self.VAL_Z)**2)

        elif self.normal =="Z" or self.normal =="z":
            Freshape = lambda v: v.reshape(self.ncx, self.ncy)
            self.VAL_X, self.VAL_Y, self.VAL_Z = Freshape(self.val_x), Freshape(self.val_y), Freshape(self.val_z)
            self.VEC_R_amp = np.sqrt(self.VAL_X.real**2+self.VAL_Y.real**2)
            self.VEC_I_amp = np.sqrt(self.VAL_X.imag**2+self.VAL_Y.imag**2)
            self.VEC_A_amp = np.sqrt(np.abs(self.VAL_X)**2+np.abs(self.VAL_Y)**2)
            self.VEC_P_amp = np.sqrt(phase(self.VAL_X)**2+phase(self.VAL_Y)**2)


    def plot2D_FD_RI(self, view="vec", ncontour=20, logamp=True, clim=None,showcontour=True, cont_perc=0.75):
        """

        """
        fig = plt.figure(figsize=(14,5))
        ax0 = plt.subplot(121)
        ax1 = plt.subplot(122)

        if view == "amp" or view == "vec":
            if logamp == True:
                val_a, val_b = np.log10(self.VEC_R_amp), np.log10(self.VEC_I_amp)
            else:
                val_a, val_b = self.VEC_R_amp, self.VEC_I_amp

        elif view =="X" or view=="x":
            val_a, val_b = self.VAL_X.real, self.VAL_X.imag
        elif view =="Y" or view=="y":
            val_a, val_b = self.VAL_Y.real, self.VAL_Y.imag
        elif view =="Z" or view=="z":
            val_a, val_b = self.VAL_Z.real, self.VAL_Z.imag

        if self.normal =="X" or self.normal=="x":
            a, b = self.y, self.z
            vec_a, vec_b = self.VAL_Y, self.VAL_Z
            xlabel = "Y (m)"
            ylabel = "Z (m)"
        elif self.normal =="Y" or self.normal =="y":
            a, b = self.x, self.z
            vec_a, vec_b = self.VAL_X, self.VAL_Z
            xlabel = "X (m)"
            ylabel = "Z (m)"
        elif self.normal =="Z" or self.normal =="z":
            a, b = self.x, self.y
            vec_a, vec_b = self.VAL_X, self.VAL_Y
            xlabel = "X (m)"
            ylabel = "Y (m)"

        if clim == None:
            vamin, vamax = val_a.min(), val_a.max()
            vbmin, vbmax = val_b.min(), val_b.max()
        else:
            vamin, vamax = clim[0][0], clim[0][1]
            vbmin, vbmax = clim[1][0], clim[1][1]

        dat0 = ax0.contourf(a, b, val_a, ncontour, clim=(vamin,vamax), vmin=vamin, vmax=vamax)
        dat1 = ax1.contourf(a, b, val_b, ncontour, clim=(vbmin,vbmax), vmin=vbmin, vmax=vbmax)

        if showcontour:
            ncontours = 1
            levels_a = vamax*cont_perc
            levels_b = vbmax*cont_perc
            ax0.contour(a, b, val_a, ncontours, level=levels_a, colors="k")
            ax1.contour(a, b, val_b, ncontours, level=levels_b, colors="k")

        cb0 = plt.colorbar(dat0,ax=ax0, format="%.1e", ticks=np.linspace(vamin, vamax, 3))
        cb1 = plt.colorbar(dat1,ax=ax1, format="%.1e", ticks=np.linspace(vbmin, vbmax, 3))
        ax0.set_xlabel(xlabel)
        ax0.set_ylabel(ylabel)
        ax1.set_xlabel(xlabel)
        # ax1.set_ylabel(ylabel)

        if view == "vec":
            ax0.streamplot(a, b, vec_a.real,  vec_b.real, color="w", linewidth=0.5)
            ax1.streamplot(a, b, vec_a.imag,  vec_b.imag, color="w", linewidth=0.5)

        plt.show()

    def plot2D_FD_AP(self, view="vec", ncontour=20, logamp=True, clim=None,showcontour=True, cont_perc=0.75):
        """

        """
        fig = plt.figure(figsize=(14,5))
        ax0 = plt.subplot(121)
        ax1 = plt.subplot(122)

        if view == "amp" or view == "vec":
            if logamp == True:
                val_a, val_b = np.log10(self.VEC_A_amp), np.log10(self.VEC_P_amp)
            else:
                val_a, val_b = self.VEC_A_amp, self.VEC_P_amp

        if logamp == True:
            if view =="X" or view=="x":
                val_a, val_b = np.log10(np.abs(self.VAL_X)), phase(self.VAL_X)
            elif view =="Y" or view=="y":
                val_a, val_b = np.log10(np.abs(self.VAL_Y)), phase(self.VAL_Y)
            elif view =="Z" or view=="z":
                val_a, val_b = np.log10(np.abs(self.VAL_Z)), phase(self.VAL_Z)
        else:
            if view =="X" or view=="x":
                val_a, val_b = np.abs(self.VAL_X), phase(self.VAL_X)
            elif view =="Y" or view=="y":
                val_a, val_b = np.abs(self.VAL_Y), phase(self.VAL_Y)
            elif view =="Z" or view=="z":
                val_a, val_b = np.abs(self.VAL_Z), phase(self.VAL_Z)

        if self.normal =="X" or self.normal=="x":
            a, b = self.y, self.z
            vec_a, vec_b = self.VAL_Y, self.VAL_Z
            xlabel = "Y (m)"
            ylabel = "Z (m)"
        elif self.normal =="Y" or self.normal =="y":
            a, b = self.x, self.z
            vec_a, vec_b = self.VAL_X, self.VAL_Z
            xlabel = "X (m)"
            ylabel = "Z (m)"
        elif self.normal =="Z" or self.normal =="z":
            a, b = self.x, self.y
            vec_a, vec_b = self.VAL_X, self.VAL_Y
            xlabel = "X (m)"
            ylabel = "Y (m)"

        if clim == None:
            vamin, vamax = val_a.min(), val_a.max()
            vbmin, vbmax = val_b.min(), val_b.max()
        else:
            vamin, vamax = clim[0][0], clim[0][1]
            vbmin, vbmax = clim[1][0], clim[1][1]

        dat0 = ax0.contourf(a, b, val_a, ncontour, clim=(vamin,vamax), vmin=vamin, vmax=vamax)
        dat1 = ax1.contourf(a, b, val_b, ncontour, clim=(vbmin,vbmax), vmin=vbmin, vmax=vbmax)

        if showcontour:
            ncontours = 1
            levels_a = vamax*cont_perc
            levels_b = vbmax*cont_perc
            ax0.contour(a, b, val_a, ncontours, level=levels_a, colors="k")
            ax1.contour(a, b, val_b, ncontours, level=levels_b, colors="k")

        cb0 = plt.colorbar(dat0,ax=ax0, format="%.1e", ticks=np.linspace(vamin, vamax, 3))
        cb1 = plt.colorbar(dat1,ax=ax1, format="%.1e", ticks=np.linspace(vbmin, vbmax, 3))
        ax0.set_xlabel(xlabel)
        ax0.set_ylabel(ylabel)
        ax1.set_xlabel(xlabel)
        # ax1.set_ylabel(ylabel)

        if view == "vec":
            ax0.streamplot(a, b, np.abs(vec_a),  np.abs(vec_b), color="w", linewidth=0.5)
            ax1.streamplot(a, b, phase(vec_a),  phase(vec_b), color="w", linewidth=0.5)

        plt.show()
