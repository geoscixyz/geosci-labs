import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.size"] = 13

def phase(z):
#     val = np.arctan2(z.real, z.imag)
    val = np.rad2deg(np.unwrap(np.angle((z))))
    return val

class DataView(object):
    """
        Provides viewingtions for Data
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

    def eval_loc(self, srcLoc,obsLoc, sigvec, fvec, orientation, func):
        self.srcLoc=srcLoc
        self.obsLoc = obsLoc
        self.sigvec = sigvec
        self.fvec = fvec
        self.orientation = orientation
        self.func = func
        self.val_xfs=np.zeros((len(sigvec),len(fvec)),dtype=complex)
        self.val_yfs=np.zeros((len(sigvec),len(fvec)),dtype=complex)
        self.val_zfs=np.zeros((len(sigvec),len(fvec)),dtype=complex)
   
        for n in range(len(sigvec)):
            #for k in range(len(fvec)):
                self.val_xfs[n],self.val_yfs[n],self.val_zfs[n]=func(self.obsLoc, srcLoc, sigvec[n], fvec, orientation=self.orientation)
    

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

    
    def plot_1D_RI_f_x(self,absloc,coordloc,ax0,ax1,sigind):

        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]

        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)
        
        ax0.set_xlabel("Frequency (Hz)")
        ax1.set_xlabel("Frequency (Hz)")
        ax0.set_ylabel("E field, Real part (V/m)")
        ax1.set_ylabel("E field, Imag part (V/m)")
            
        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_xfs.real[:,freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_xfs.real[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)
        
        
        ax0.plot(self.fvec,self.val_xfs.real[sigind,:],color="blue")
        ax1.plot(self.fvec,self.val_xfs.imag[sigind,:],color="red")
        
        ax0ymin, ax0ymax = self.val_xfs.real[sigind,:].min(),self.val_xfs.real[sigind,:].max()
        
        ax1ymin, ax1ymax = self.val_xfs.imag[sigind,:].min(),self.val_xfs.imag[sigind,:].max()

        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(("$\sigma$ =%0.5f S/m")%(self.sigvec[sigind]),
            xy=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), textcoords='data',
            fontsize=14.)

        ax1.annotate(("$\sigma$ =%0.5f S/m")%(self.sigvec[sigind]),
            xy=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), textcoords='data',
            fontsize=14.)

        return ax0,ax1

    def plot_1D_AP_f_x(self,absloc,coordloc,ax0,ax1,sigind):

        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]

        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)
        
        ax0.set_xlabel("Frequency (Hz)")
        ax1.set_xlabel("Frequency (Hz)")
        ax0.set_ylabel("E field, Amplitude (V/m)")
        ax1.set_ylabel("E field, Phase")
            
        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_xfs.real[:,freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_xfs.real[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)

        
        ax0.plot(self.fvec,np.absolute(self.val_xfs[sigind,:]),color="blue")
        ax1.plot(self.fvec,phase(self.val_xfs[sigind,:]),color="red")
        
        ax0ymin, ax0ymax = np.absolute(self.val_xfs[sigind,:]).min(),np.absolute(self.val_xfs[sigind,:]).max()
        ax1ymin, ax1ymax = phase(self.val_xfs[sigind,:]).min(),phase(self.val_xfs[sigind,:]).max()

        #ax2.plot(self.fvec[freqind]*np.ones_like(self.val_xfs[sigind,:]),
        #            np.linspace(ax2ymin,ax2ymax,len(self.val_xfs[sigind,:])),linestyle="dashed",color="black",linewidth=3.0)
        
        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(("$\sigma$ =%0.5f S/m")%(self.sigvec[sigind]),
            xy=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), textcoords='data',
            fontsize=14.)

        ax1.annotate(("$\sigma$ =%0.5f S/m")%(self.sigvec[sigind]),
            xy=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), textcoords='data',
            fontsize=14.)

        return ax0,ax1

    def plot_1D_RI_sig_x(self,absloc,coordloc,ax0,ax1,freqind):
        
        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]

        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)

        ax0.set_xlabel("Conductivity (S/m)")
        ax1.set_xlabel("Conductivity (S/m)")
        ax0.set_ylabel("E field, Real part (V/m)")
        ax1.set_ylabel("E field, Imag part (V/m)")
        
        ax0.plot(self.sigvec,self.val_xfs.real[:,freqind],color="blue")
        ax1.plot(self.sigvec,self.val_xfs.imag[:,freqind],color="red")
        
        ax0ymin, ax0ymax = self.val_xfs.real[:,freqind].min(),self.val_xfs.real[:,freqind].max()
        ax1ymin, ax1ymax = self.val_xfs.imag[:,freqind].min(),self.val_xfs.imag[:,freqind].max()

        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_xfs.real[:,freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_xfs.real[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)
        
        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(("f =%0.5f Hz")%(self.fvec[freqind]),
            xy=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), textcoords='data',
           fontsize=14.)

        ax1.annotate(("f =%0.5f Hz")%(self.fvec[freqind]),
            xy=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), textcoords='data',
           fontsize=14.)

        return ax0,ax1
        

    def plot_1D_AP_sig_x(self,absloc,coordloc,ax0,ax1,freqind):

        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]

        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)

        ax0.set_xlabel("Conductivity (S/m)")
        ax1.set_xlabel("Conductivity (S/m)")
        ax0.set_ylabel("E field, Amplitude (V/m)")
        ax1.set_ylabel("E field, Phase")

        ax0.plot(self.sigvec,np.absolute(self.val_xfs[:,freqind]),color="blue")
        ax1.plot(self.sigvec,phase(self.val_xfs[:,freqind]),color="red")
        
        ax0ymin, ax0ymax = np.absolute(self.val_xfs[:,freqind]).min(),np.absolute(self.val_xfs[:,freqind]).max()
        ax1ymin, ax1ymax = phase(self.val_xfs[:,freqind]).min(),phase(self.val_xfs[:,freqind]).max()
        
        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_xfs[:,freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_xfs[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)
        
        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(("f =%0.5f Hz")%(self.fvec[freqind]),
            xy=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), textcoords='data',
           fontsize=14.)

        ax1.annotate(("f =%0.5f Hz")%(self.fvec[freqind]),
            xy=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), textcoords='data',
           fontsize=14.)

        return ax0,ax1

    def plot_1D_phasor_f_x(self,absloc,coordloc,ax,sigind):

        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]
        
        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)

        ax.plot(self.val_xfs.real[sigind,:],self.val_xfs.imag[sigind,:])

    def plot_1D_phasor_sig_x(self,absloc,coordloc,ax,freqind):

        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]
        
        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)

        ax.plot(self.val_xfs.real[:,freqind],self.val_xfs.imag[:,freqind])



    def plot_1D_RI_f_y(self,absloc,coordloc,ax0,ax1,sigind):

        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]

        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)
        
        ax0.set_xlabel("Frequency (Hz)")
        ax1.set_xlabel("Frequency (Hz)")
        ax0.set_ylabel("E field, Real part (V/m)")
        ax1.set_ylabel("E field, Imag part (V/m)")
            
        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_xfs.real[:,freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_xfs.real[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)
        
        
        ax0.plot(self.fvec,self.val_yfs.real[sigind,:],color="blue")
        ax1.plot(self.fvec,self.val_yfs.imag[sigind,:],color="red")
        
        ax0ymin, ax0ymax = self.val_yfs.real[sigind,:].min(),self.val_yfs.real[sigind,:].max()
        
        ax1ymin, ax1ymax = self.val_yfs.imag[sigind,:].min(),self.val_yfs.imag[sigind,:].max()

        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(("$\sigma$ =%0.5f S/m")%(self.sigvec[sigind]),
            xy=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), textcoords='data',
            fontsize=14.)

        ax1.annotate(("$\sigma$ =%0.5f S/m")%(self.sigvec[sigind]),
            xy=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), textcoords='data',
            fontsize=14.)

        return ax0,ax1

    def plot_1D_AP_f_y(self,absloc,coordloc,ax0,ax1,sigind):

        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]

        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)
        
        ax0.set_xlabel("Frequency (Hz)")
        ax1.set_xlabel("Frequency (Hz)")
        ax0.set_ylabel("E field, Amplitude (V/m)")
        ax1.set_ylabel("E field, Phase")
            
        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_yfs.real[:,freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_yfs.real[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)

        
        ax0.plot(self.fvec,np.absolute(self.val_yfs[sigind,:]),color="blue")
        ax1.plot(self.fvec,phase(self.val_yfs[sigind,:]),color="red")
        
        ax0ymin, ax0ymax = np.absolute(self.val_yfs[sigind,:]).min(),np.absolute(self.val_yfs[sigind,:]).max()
        ax1ymin, ax1ymax = phase(self.val_yfs[sigind,:]).min(),phase(self.val_yfs[sigind,:]).max()

        #ax2.plot(self.fvec[freqind]*np.ones_like(self.val_yfs[sigind,:]),
        #            np.linspace(ax2ymin,ax2ymax,len(self.val_yfs[sigind,:])),linestyle="dashed",color="black",linewidth=3.0)
        
        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(("$\sigma$ =%0.5f S/m")%(self.sigvec[sigind]),
            xy=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), textcoords='data',
            fontsize=14.)

        ax1.annotate(("$\sigma$ =%0.5f S/m")%(self.sigvec[sigind]),
            xy=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), textcoords='data',
            fontsize=14.)

        return ax0,ax1

    def plot_1D_RI_sig_y(self,absloc,coordloc,ax0,ax1,freqind):
        
        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]

        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)

        ax0.set_xlabel("Conductivity (S/m)")
        ax1.set_xlabel("Conductivity (S/m)")
        ax0.set_ylabel("E field, Real part (V/m)")
        ax1.set_ylabel("E field, Imag part (V/m)")
        
        ax0.plot(self.sigvec,self.val_yfs.real[:,freqind],color="blue")
        ax1.plot(self.sigvec,self.val_yfs.imag[:,freqind],color="red")
        
        ax0ymin, ax0ymax = self.val_yfs.real[:,freqind].min(),self.val_yfs.real[:,freqind].max()
        ax1ymin, ax1ymax = self.val_yfs.imag[:,freqind].min(),self.val_yfs.imag[:,freqind].max()

        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_yfs.real[:,freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_yfs.real[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)
        
        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(("f =%0.5f Hz")%(self.fvec[freqind]),
            xy=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), textcoords='data',
           fontsize=14.)

        ax1.annotate(("f =%0.5f Hz")%(self.fvec[freqind]),
            xy=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), textcoords='data',
           fontsize=14.)

        return ax0,ax1
        

    def plot_1D_AP_sig_y(self,absloc,coordloc,ax0,ax1,freqind):

        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]

        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)

        ax0.set_xlabel("Conductivity (S/m)")
        ax1.set_xlabel("Conductivity (S/m)")
        ax0.set_ylabel("E field, Amplitude (V/m)")
        ax1.set_ylabel("E field, Phase")

        ax0.plot(self.sigvec,np.absolute(self.val_yfs[:,freqind]),color="blue")
        ax1.plot(self.sigvec,phase(self.val_yfs[:,freqind]),color="red")
        
        ax0ymin, ax0ymax = np.absolute(self.val_yfs[:,freqind]).min(),np.absolute(self.val_yfs[:,freqind]).max()
        ax1ymin, ax1ymax = phase(self.val_yfs[:,freqind]).min(),phase(self.val_yfs[:,freqind]).max()
        
        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_yfs[:,freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_yfs[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)
        
        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(("f =%0.5f Hz")%(self.fvec[freqind]),
            xy=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), textcoords='data',
           fontsize=14.)

        ax1.annotate(("f =%0.5f Hz")%(self.fvec[freqind]),
            xy=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), textcoords='data',
           fontsize=14.)

        return ax0,ax1

    def plot_1D_phasor_f_y(self,absloc,coordloc,ax,sigind):

        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]
        
        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)

        ax.plot(self.val_yfs.real[sigind,:],self.val_yfs.imag[sigind,:])

    def plot_1D_phasor_sig_y(self,absloc,coordloc,ax,freqind):

        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]
        
        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)

        ax.plot(self.val_yfs.real[:,freqind],self.val_yfs.imag[:,freqind])


    def plot_1D_RI_f_z(self,absloc,coordloc,ax0,ax1,sigind):

        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]

        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)
        
        ax0.set_xlabel("Frequency (Hz)")
        ax1.set_xlabel("Frequency (Hz)")
        ax0.set_ylabel("E field, Real part (V/m)")
        ax1.set_ylabel("E field, Imag part (V/m)")
            
        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_xfs.real[:,freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_xfs.real[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)
        
        
        ax0.plot(self.fvec,self.val_zfs.real[sigind,:],color="blue")
        ax1.plot(self.fvec,self.val_zfs.imag[sigind,:],color="red")
        
        ax0ymin, ax0ymax = self.val_zfs.real[sigind,:].min(),self.val_zfs.real[sigind,:].max()
        
        ax1ymin, ax1ymax = self.val_zfs.imag[sigind,:].min(),self.val_zfs.imag[sigind,:].max()

        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(("$\sigma$ =%0.5f S/m")%(self.sigvec[sigind]),
            xy=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), textcoords='data',
            fontsize=14.)

        ax1.annotate(("$\sigma$ =%0.5f S/m")%(self.sigvec[sigind]),
            xy=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), textcoords='data',
            fontsize=14.)

        return ax0,ax1

    def plot_1D_AP_f_z(self,absloc,coordloc,ax0,ax1,sigind):

        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]

        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)
        
        ax0.set_xlabel("Frequency (Hz)")
        ax1.set_xlabel("Frequency (Hz)")
        ax0.set_ylabel("E field, Amplitude (V/m)")
        ax1.set_ylabel("E field, Phase")
            
        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_zfs.real[:,freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_zfs.real[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)

        
        ax0.plot(self.fvec,np.absolute(self.val_zfs[sigind,:]),color="blue")
        ax1.plot(self.fvec,phase(self.val_zfs[sigind,:]),color="red")
        
        ax0ymin, ax0ymax = np.absolute(self.val_zfs[sigind,:]).min(),np.absolute(self.val_zfs[sigind,:]).max()
        ax1ymin, ax1ymax = phase(self.val_zfs[sigind,:]).min(),phase(self.val_zfs[sigind,:]).max()

        #ax2.plot(self.fvec[freqind]*np.ones_like(self.val_zfs[sigind,:]),
        #            np.linspace(ax2ymin,ax2ymax,len(self.val_zfs[sigind,:])),linestyle="dashed",color="black",linewidth=3.0)
        
        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(("$\sigma$ =%0.5f S/m")%(self.sigvec[sigind]),
            xy=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), textcoords='data',
            fontsize=14.)

        ax1.annotate(("$\sigma$ =%0.5f S/m")%(self.sigvec[sigind]),
            xy=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), textcoords='data',
            fontsize=14.)

        return ax0,ax1

    def plot_1D_RI_sig_z(self,absloc,coordloc,ax0,ax1,freqind):
        
        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]

        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)

        ax0.set_xlabel("Conductivity (S/m)")
        ax1.set_xlabel("Conductivity (S/m)")
        ax0.set_ylabel("E field, Real part (V/m)")
        ax1.set_ylabel("E field, Imag part (V/m)")
        
        ax0.plot(self.sigvec,self.val_zfs.real[:,freqind],color="blue")
        ax1.plot(self.sigvec,self.val_zfs.imag[:,freqind],color="red")
        
        ax0ymin, ax0ymax = self.val_zfs.real[:,freqind].min(),self.val_zfs.real[:,freqind].max()
        ax1ymin, ax1ymax = self.val_zfs.imag[:,freqind].min(),self.val_zfs.imag[:,freqind].max()

        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_zfs.real[:,freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_zfs.real[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)
        
        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(("f =%0.5f Hz")%(self.fvec[freqind]),
            xy=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), textcoords='data',
           fontsize=14.)

        ax1.annotate(("f =%0.5f Hz")%(self.fvec[freqind]),
            xy=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), textcoords='data',
           fontsize=14.)

        return ax0,ax1
        

    def plot_1D_AP_sig_z(self,absloc,coordloc,ax0,ax1,freqind):

        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]

        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)

        ax0.set_xlabel("Conductivity (S/m)")
        ax1.set_xlabel("Conductivity (S/m)")
        ax0.set_ylabel("E field, Amplitude (V/m)")
        ax1.set_ylabel("E field, Phase")

        ax0.plot(self.sigvec,np.absolute(self.val_zfs[:,freqind]),color="blue")
        ax1.plot(self.sigvec,phase(self.val_zfs[:,freqind]),color="red")
        
        ax0ymin, ax0ymax = np.absolute(self.val_zfs[:,freqind]).min(),np.absolute(self.val_zfs[:,freqind]).max()
        ax1ymin, ax1ymax = phase(self.val_zfs[:,freqind]).min(),phase(self.val_zfs[:,freqind]).max()
        
        #ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_zfs[:,freqind]),
        #         np.linspace(ax0ymin,ax0ymax,len(self.val_zfs[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)
        
        ax0.set_ylim(ax0ymin, ax0ymax)
        ax1.set_ylim(ax1ymin, ax1ymax)

        ax0.set_xscale('log')
        ax1.set_xscale('log')

        ax0.annotate(("f =%0.5f Hz")%(self.fvec[freqind]),
            xy=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), textcoords='data',
           fontsize=14.)

        ax1.annotate(("f =%0.5f Hz")%(self.fvec[freqind]),
            xy=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax1ymin+(ax1ymax-ax1ymin)/4.), textcoords='data',
           fontsize=14.)

        return ax0,ax1

    def plot_1D_phasor_f_z(self,absloc,coordloc,ax,sigind):

        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]
        
        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)

        ax.plot(self.val_zfs.real[sigind,:],self.val_zfs.imag[sigind,:])

    def plot_1D_phasor_sig_z(self,absloc,coordloc,ax,freqind):

        if self.normal.upper() == "Z":
            obsLoc=np.c_[absloc,coordloc,self.z]
        elif self.normal.upper() == "Y":
            obsLoc=np.c_[absloc,self.y,coordloc]
        elif self.normal.upper() == "X":
            obsLoc=np.c_[self.x,absloc,coordloc]
        
        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation,self.func)

        ax.plot(self.val_zfs.real[:,freqind],self.val_zfs.imag[:,freqind])

        



    def plot_1D_x(self,obslocx,obslocy,obslocz,sigind,freqind,mode):
        
        #sigind = np.where( sigplt == self.sigvec)[0][0]
        #freqind = np.where( freqplt == self.fvec)[0][0]
            
        fig = plt.figure(figsize=(14,5))
        ax0 = plt.subplot(121)
        ax2 = plt.subplot(122)
        
        ax1 = ax0.twinx()
        ax3 = ax2.twinx()

        obsLoc=np.c_[obslocx,obslocy,obslocz]
        self.eval_loc(self.srcLoc,obsLoc, self.sigvec, self.fvec, self.orientation, self.func)
        
        if mode =="RI":        
        
            ax0.set_xlabel("Conductivity (S/m)")
            ax0.set_ylabel("E field, Real part (V/m)")
            ax1.set_ylabel("E field, Imag part (V/m)")
            
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("E field, Real part (V/m)")
            ax3.set_ylabel("E field, Imag part (V/m)")
            
            ax0.plot(self.sigvec,self.val_xfs.real[:,freqind],color="blue")
            ax1.plot(self.sigvec,self.val_xfs.imag[:,freqind],color="red")
            
            ax0ymin, ax0ymax = self.val_xfs.real[:,freqind].min(),self.val_xfs.real[:,freqind].max()
            
            ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_xfs.real[:,freqind]),
                     np.linspace(ax0ymin,ax0ymax,len(self.val_xfs.real[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)
            
            ax0.set_ylim(ax0ymin, ax0ymax)
            
            ax2.plot(self.fvec,self.val_xfs.real[sigind,:],color="blue")
            ax3.plot(self.fvec,self.val_xfs.imag[sigind,:],color="red")
            
            ax2ymin, ax2ymax = self.val_xfs.real[sigind,:].min(),self.val_xfs.real[sigind,:].max()
            
            ax2.plot(self.fvec[freqind]*np.ones_like(self.val_xfs.imag[sigind,:]),
                        np.linspace(ax2ymin,ax2ymax,len(self.val_xfs.imag[sigind,:])),linestyle="dashed",color="black",linewidth=3.0)
            
            ax2.set_ylim(ax2ymin, ax2ymax)
        
        elif mode=="AP":
            
            ax0.set_xlabel("Conductivity (S/m)")
            ax0.set_ylabel("E field, Amplitude (V/m)")
            ax1.set_ylabel("E field, Phase")
            
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("E field, Amplitude (V/m)")
            ax3.set_ylabel("E field, Phase")
            
            ax0.plot(self.sigvec,np.absolute(self.val_xfs[:,freqind]),color="blue")
            ax1.plot(self.sigvec,phase(self.val_xfs[:,freqind]),color="red")
            
            ax0ymin, ax0ymax = np.absolute(self.val_xfs[:,freqind]).min(),np.absolute(self.val_xfs[:,freqind]).max()
            
            ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_xfs[:,freqind]),
                     np.linspace(ax0ymin,ax0ymax,len(self.val_xfs[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)
            
            ax0.set_ylim(ax0ymin, ax0ymax)
            
            ax2.plot(self.fvec,np.absolute(self.val_xfs[sigind,:]),color="blue")
            ax3.plot(self.fvec,phase(self.val_xfs[sigind,:]),color="red")
            
            ax2ymin, ax2ymax = np.absolute(self.val_xfs[sigind,:]).min(),np.absolute(self.val_xfs[sigind,:]).max()
            
            ax2.plot(self.fvec[freqind]*np.ones_like(self.val_xfs[sigind,:]),
                        np.linspace(ax2ymin,ax2ymax,len(self.val_xfs[sigind,:])),linestyle="dashed",color="black",linewidth=3.0)
            
            ax2.set_ylim(ax2ymin, ax2ymax)
      
        #ax0.set_yscale('log')
        ax0.set_xscale('log')
        ax1.set_xscale('log')
        
        ax2.set_xscale('log')
        ax3.set_xscale('log')

        ax0.annotate(("f =%0.5f Hz")%(self.fvec[freqind]),
            xy=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.sigvec.min())+np.log10(self.sigvec.max()))/2), ax0ymin+(ax0ymax-ax0ymin)/4.), textcoords='data',
           fontsize=14.)

        ax2.annotate(("$\sigma$ =%0.5f S/m")%(self.sigvec[sigind]),
            xy=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax2ymin+(ax2ymax-ax2ymin)/4.), xycoords='data',
            xytext=(10.**((np.log10(self.fvec.min())+np.log10(self.fvec.max()))/2), ax2ymin+(ax2ymax-ax2ymin)/4.), textcoords='data',
            fontsize=14.)

        
        plt.tight_layout()
    
    def plot_1D_y(self,sigind,freqind,mode):
        
        #sigind = np.where( sigplt == self.sigvec)[0][0]
        #freqind = np.where( freqplt == self.fvec)[0][0]
            
        fig = plt.figure(figsize=(14,5))
        ax0 = plt.subplot(121)
        ax2 = plt.subplot(122)
        
        ax1 = ax0.twinx()
        ax3 = ax2.twinx()
        
        if mode =="RI":        
        
            ax0.set_xlabel("Conductivity (S/m)")
            ax0.set_ylabel("E field, Real part (V/m)")
            ax1.set_ylabel("E field, Imag part (V/m)")
            
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("E field, Real part (V/m)")
            ax3.set_ylabel("E field, Imag part (V/m)")
            
            ax0.plot(self.sigvec,self.val_yfs.real[:,freqind],color="blue")
            ax1.plot(self.sigvec,self.val_yfs.imag[:,freqind],color="red")
            
            ax0ymin, ax0ymax = self.val_yfs.real[:,freqind].min(),self.val_yfs.real[:,freqind].max()
            
            ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_yfs.real[:,freqind]),
                     np.linspace(ax0ymin,ax0ymax,len(self.val_yfs.real[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)
            
            ax0.set_ylim(ax0ymin, ax0ymax)
            
            ax2.plot(self.fvec,self.val_yfs.real[sigind,:],color="blue")
            ax3.plot(self.fvec,self.val_yfs.imag[sigind,:],color="red")
            
            ax2ymin, ax2ymax = self.val_yfs.real[sigind,:].min(),self.val_yfs.real[sigind,:].max()
            
            ax2.plot(self.fvec[freqind]*np.ones_like(self.val_yfs.imag[sigind,:]),
                        np.linspace(ax2ymin,ax2ymax,len(self.val_yfs.imag[sigind,:])),linestyle="dashed",color="black",linewidth=3.0)
            
            ax2.set_ylim(ax2ymin, ax2ymax)
        
        elif mode=="AP":
            
            ax0.set_xlabel("Conductivity (S/m)")
            ax0.set_ylabel("E field, Amplitude (V/m)")
            ax1.set_ylabel("E field, Phase")
            
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("E field, Amplitude (V/m)")
            ax3.set_ylabel("E field, Phase")
            
            ax0.plot(self.sigvec,np.absolute(self.val_yfs[:,freqind]),color="blue")
            ax1.plot(self.sigvec,phase(self.val_yfs[:,freqind]),color="red")
            
            ax0ymin, ax0ymax = np.absolute(self.val_yfs[:,freqind]).min(),np.absolute(self.val_yfs[:,freqind]).max()
            
            ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_yfs[:,freqind]),
                     np.linspace(ax0ymin,ax0ymax,len(self.val_yfs[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)
            
            ax0.set_ylim(ax0ymin, ax0ymax)
            
            ax2.plot(self.fvec,np.absolute(self.val_yfs[sigind,:]),color="blue")
            ax3.plot(self.fvec,phase(self.val_yfs[sigind,:]),color="red")
            
            ax2ymin, ax2ymax = np.absolute(self.val_yfs[sigind,:]).min(),np.absolute(self.val_yfs[sigind,:]).max()
            
            ax2.plot(self.fvec[freqind]*np.ones_like(self.val_yfs[sigind,:]),
                        np.linspace(ax2ymin,ax2ymax,len(self.val_yfs[sigind,:])),linestyle="dashed",color="black",linewidth=3.0)
            
            ax2.set_ylim(ax2ymin, ax2ymax)
        
      
        #ax0.set_yscale('log')
        ax0.set_xscale('log')
        ax1.set_xscale('log')
        
        ax2.set_xscale('log')
        ax3.set_xscale('log')

        #ax0.annotate(("$\f$ =%5.5f Hz")%(self.fvec[freqind]*10**(3)),
        #    xy=(self.sigvec.max()/100., ax0ymin+(ax0ymax-ax0ymin)/4.), xycoords='data',
        #    xytext=(self.sigvec.max()/100., ax0ymin+(ax0ymax-ax0ymin)/4.), textcoords='data',
         #   fontsize=14.)

        #ax2.annotate(("$\sigma$ =%3.3f mS/m")%(self.sigvec[sigind]*10**(3)),
        #    xy=(self.fvec.max()/100., ax2ymin+(ax2ymax-ax2ymin)/4.), xycoords='data',
        #    xytext=(self.fvec.max()/100., ax2ymin+(ax2ymax-ax2ymin)/4.), textcoords='data',
        #    fontsize=14.)

        
        plt.tight_layout()

    def plot_1D_z(self,sigind,freqind,mode):
        
        #sigind = np.where( sigplt == self.sigvec)[0][0]
        #freqind = np.where( freqplt == self.fvec)[0][0]
            
        fig = plt.figure(figsize=(14,5))
        ax0 = plt.subplot(121)
        ax2 = plt.subplot(122)
        
        ax1 = ax0.twinx()
        ax3 = ax2.twinx()
        
        if mode =="RI":        
        
            ax0.set_xlabel("Conductivity (S/m)")
            ax0.set_ylabel("E field, Real part (V/m)")
            ax1.set_ylabel("E field, Imag part (V/m)")
            
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("E field, Real part (V/m)")
            ax3.set_ylabel("E field, Imag part (V/m)")
            
            ax0.plot(self.sigvec,self.val_zfs.real[:,freqind],color="blue")
            ax1.plot(self.sigvec,self.val_zfs.imag[:,freqind],color="red")
            
            ax0ymin, ax0ymax = self.val_zfs.real[:,freqind].min(),self.val_zfs.real[:,freqind].max()
            
            ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_zfs.real[:,freqind]),
                     np.linspace(ax0ymin,ax0ymax,len(self.val_zfs.real[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)
            
            ax0.set_ylim(ax0ymin, ax0ymax)
            
            ax2.plot(self.fvec,self.val_zfs.real[sigind,:],color="blue")
            ax3.plot(self.fvec,self.val_zfs.imag[sigind,:],color="red")
            
            ax2ymin, ax2ymax = self.val_zfs.real[sigind,:].min(),self.val_zfs.real[sigind,:].max()
            
            ax2.plot(self.fvec[freqind]*np.ones_like(self.val_zfs.imag[sigind,:]),
                        np.linspace(ax2ymin,ax2ymax,len(self.val_zfs.imag[sigind,:])),linestyle="dashed",color="black",linewidth=3.0)
            
            ax2.set_ylim(ax2ymin, ax2ymax)
        
        elif mode=="AP":
            
            ax0.set_xlabel("Conductivity (S/m)")
            ax0.set_ylabel("E field, Amplitude (V/m)")
            ax1.set_ylabel("E field, Phase")
            
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("E field, Amplitude (V/m)")
            ax3.set_ylabel("E field, Phase")
            
            ax0.plot(self.sigvec,np.absolute(self.val_zfs[:,freqind]),color="blue")
            ax1.plot(self.sigvec,phase(self.val_zfs[:,freqind]),color="red")
            
            ax0ymin, ax0ymax = np.absolute(self.val_zfs[:,freqind]).min(),np.absolute(self.val_zfs[:,freqind]).max()
            
            ax0.plot(self.sigvec[sigind]*np.ones_like(self.val_zfs[:,freqind]),
                     np.linspace(ax0ymin,ax0ymax,len(self.val_zfs[:,freqind])),linestyle="dashed",color="black",linewidth=3.0)
            
            ax0.set_ylim(ax0ymin, ax0ymax)
            
            ax2.plot(self.fvec,np.absolute(self.val_zfs[sigind,:]),color="blue")
            ax3.plot(self.fvec,phase(self.val_zfs[sigind,:]),color="red")
            
            ax2ymin, ax2ymax = np.absolute(self.val_zfs[sigind,:]).min(),np.absolute(self.val_zfs[sigind,:]).max()
            
            ax2.plot(self.fvec[freqind]*np.ones_like(self.val_zfs[sigind,:]),
                        np.linspace(ax2ymin,ax2ymax,len(self.val_zfs[sigind,:])),linestyle="dashed",color="black",linewidth=3.0)
            
            ax2.set_ylim(ax2ymin, ax2ymax)
      
        #ax0.set_yscale('log')
        ax0.set_xscale('log')
        ax1.set_xscale('log')
        
        ax2.set_xscale('log')
        ax3.set_xscale('log')

        #ax0.annotate(("$\f$ =%5.5f Hz")%(self.fvec[freqind]*10**(3)),
        #    xy=(self.sigvec.max()/100., ax0ymin+(ax0ymax-ax0ymin)/4.), xycoords='data',
        #    xytext=(self.sigvec.max()/100., ax0ymin+(ax0ymax-ax0ymin)/4.), textcoords='data',
         #   fontsize=14.)

        #ax2.annotate(("$\sigma$ =%3.3f mS/m")%(self.sigvec[sigind]*10**(3)),
        #    xy=(self.fvec.max()/100., ax2ymin+(ax2ymax-ax2ymin)/4.), xycoords='data',
        #    xytext=(self.fvec.max()/100., ax2ymin+(ax2ymax-ax2ymin)/4.), textcoords='data',
        #    fontsize=14.)

        
        plt.tight_layout()
    
    