import numpy as np
from pymatsolver import Pardiso
from SimPEG import DC, Utils, Maps
from SimPEG import (
    DataMisfit, Regularization, Optimization,
    InvProblem, Directives, Inversion
)
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings("ignore")
matplotlib.rcParams['font.size'] = 14

from ipywidgets import GridspecLayout, widgets
import os
from scipy.interpolate import interp1d

class DCRSimulationApp(object):
    """docstring for DCRSimulationApp"""

    # Parameters for sensitivity matrix, G
    # Parameters for Model
    seed = None
    percentage = None
    floor = None
    uncertainty = None
    mesh = None
    actind = None
    IO = None
    survey = None
    _rho = None
    a = None
    n_spacing = None
    xmax = None
    survey_type = None

    def __init__(self):
        super(DCRSimulationApp, self).__init__()
        self.IO = DC.IO()

    @property
    def rho(self):
        return self._rho

    def get_survey(
        self,
        a=10,
        n_spacing=8,
        xmax = 200,
        survey_type='dipole-dipole'
    ):
        # Generate survey
        xmin = 0
        ymin, ymax = 0., 0.
        zmin, zmax = 0, 0
        endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
        self.survey = DC.Utils.gen_DCIPsurvey(endl, survey_type=survey_type, dim=2, a=a, b=a, n=n_spacing)
        self.survey.getABMN_locations()
        self.survey = self.IO.from_ambn_locations_to_survey(
            self.survey.a_locations, self.survey.b_locations,
            self.survey.m_locations, self.survey.n_locations,
            survey_type, data_dc_type='volt'
        )

    def plot_src_rx(self, i_src, i_rx):
        src = self.survey.srcList[i_src]
        rx = src.rxList[0]
        if i_rx > rx.nD-1:
            print("Maximum rx number is {0}!".format(rx.nD-1))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.set_xlabel("x (m)")
            ax.set_ylabel("n-spacing")

            if self.IO.survey_type == 'pole-dipole':
                a_location = src.loc
                m_locations = rx.locs[0]
                n_locations = rx.locs[1]
                grid = (
                    0.5*(a_location[0] + 0.5*(m_locations[i_rx,0] + n_locations[i_rx,0])),
                    1./3*(a_location[0] - 0.5*(m_locations[i_rx,0] + n_locations[i_rx,0])),
                )
                plt.plot(a_location[0], -a_location[1], 'rv')
                plt.plot(m_locations[i_rx,0], -m_locations[i_rx,1], 'yv')
                plt.plot(n_locations[i_rx,0], -n_locations[i_rx,1], 'gv')
                plt.plot(grid[0], grid[1], 'ro')
            elif self.IO.survey_type == 'dipole-dipole':
                a_location = src.loc[0]
                b_location = src.loc[1]
                m_locations = rx.locs[0]
                n_locations = rx.locs[1]
                grid = (
                    0.5*(0.5*(a_location[0]+b_location[0]) + 0.5*(m_locations[i_rx,0] + n_locations[i_rx,0])),
                    1./3*(0.5*(a_location[0]+b_location[0]) - 0.5*(m_locations[i_rx,0] + n_locations[i_rx,0])),
                )
                plt.plot(a_location[0], -a_location[1], 'rv')
                plt.plot(b_location[0], -b_location[1], 'bv')
                plt.plot(m_locations[i_rx,0], -m_locations[i_rx,1], 'yv')
                plt.plot(n_locations[i_rx,0], -n_locations[i_rx,1], 'gv')
                plt.plot(grid[0], grid[1], 'ro')
            elif self.IO.survey_type == 'dipole-pole':
                a_location = src.loc[0]
                b_location = src.loc[1]
                m_locations = rx.locs
                grid = (
                    0.5*(0.5*(a_location[0]+b_location[0]) + (m_locations[i_rx,0])),
                    1./3*(0.5*(a_location[0]+b_location[0]) - (m_locations[i_rx,0])),
                )
                plt.plot(a_location[0], -a_location[1], 'rv')
                plt.plot(b_location[0], -b_location[1], 'bv')
                plt.plot(m_locations[i_rx,0], -m_locations[i_rx,1], 'yv')
                plt.plot(grid[0], grid[1], 'ro')
            elif self.IO.survey_type == 'pole-pole':
                a_location = src.loc
                m_locations = rx.locs
                grid = (
                    0.5*((a_location[0]) + (m_locations[i_rx,0])),
                    1./3*((a_location[0]) - (m_locations[i_rx,0])),
                )
                plt.plot(a_location[0], -a_location[1], 'rv')
                plt.plot(m_locations[i_rx,0], -m_locations[i_rx,1], 'yv')
                plt.plot(grid[0], grid[1], 'ro')

            plt.plot(self.IO.grids[:,0], -self.IO.grids[:,1], 'k.')
            ax.set_aspect(1)
            xmin, xmax = self.IO.grids[:,0].min(), self.IO.grids[:,0].max()
            dx = xmax-xmin
            ax.set_xlim(xmin-dx/10., xmax+dx/10.)
            dummy = ax.set_yticks([])

    def plot_survey(
        self,
        line_length = 200.,
        a=10,
        survey_type='dipole-dipole',
        n_spacing=8,
        i_src=0,
        i_rx=0,
    ):
        self.a = a
        self.n_spacing = n_spacing
        self.survey_type = survey_type
        self.line_length = line_length
        self.get_survey(
            a=a,
            n_spacing=n_spacing,
            survey_type=survey_type,
            xmax=line_length
        )
        self.plot_src_rx(i_src, i_rx)

    def get_mesh(self, add_topography=False, seed=1):

        self.get_survey(
            a=self.a,
            n_spacing=self.n_spacing,
            survey_type=self.survey_type,
            xmax=self.line_length
        )

        self.mesh, self.actind = self.IO.set_mesh()

        if add_topography:
            topo, mesh1D = DC.Utils.genTopography(self.mesh, -10, 0, its=100, seed=seed)
            self.actind = Utils.surface2ind_topo(self.mesh, np.c_[mesh1D.vectorCCx, topo])

        self.survey.drapeTopo(self.mesh, self.actind, option="top")
        self.survey.getABMN_locations()
        self.survey = self.IO.from_ambn_locations_to_survey(
            self.survey.a_locations, self.survey.b_locations,
            self.survey.m_locations, self.survey.n_locations,
            self.survey_type, data_dc_type='volt'
        )



    def get_block_index(self, xc=50, yc=50, dx=20, dy=20):
        p0 = np.array([xc - dx / 2.0, yc + dy / 2])
        p1 = np.array([xc + dx / 2.0, yc - dy / 2])
        index = Utils.ModelBuilder.getIndicesBlock(p0, p1, self.mesh.gridCC)
        return index

    def get_block_points(self, xc=50, yc=50, dx=20, dy=20):
        x = np.array(
            [xc - dx / 2.0, xc + dx / 2.0, xc + dx / 2.0, xc - dx / 2.0, xc - dx / 2.0]
        )
        y = np.array(
            [yc - dy / 2.0, yc - dy / 2.0, yc + dy / 2.0, yc + dy / 2.0, yc - dy / 2.0]
        )
        return x, y

    def plot_model(
        self,
        rho0,
        rho1,
        xc,
        yc,
        dx,
        dy,
        std,
        show_grid=False,
        show_core=False,
        add_topography=False,
        simulate=False,
        update=False,
        write_obs_file=False,
        obs_name=None,
    ):

        self.std = std

        if simulate:
            fig, ax = plt.subplots(1, 1, figsize = (10, 7))
            self.get_mesh(add_topography=add_topography)
            self._rho = np.ones(self.mesh.nC) * rho0
            index = self.get_block_index(xc=xc, yc=yc, dx=dx, dy=dy)
            self._rho[index] = rho1
            self._rho[~self.actind] = np.nan

            self.plot_data(ax=ax)
            ax.set_aspect(1)
            if write_obs_file:
                self.write_to_csv(obs_name)
                print ('{0} is written'.format(obs_name))
        else:
            if write_obs_file:
                print ('>> write_obs_file is only activated when simiulate is checked!')
            else:
                fig, ax = plt.subplots(1, 1, figsize = (10, 7))
                self.get_mesh(add_topography=add_topography)
                self._rho = np.ones(self.mesh.nC) * rho0
                index = self.get_block_index(xc=xc, yc=yc, dx=dx, dy=dy)
                self._rho[index] = rho1
                self._rho[~self.actind] = np.nan
                vmin = np.log10(self.rho[self.actind].min())
                vmax = np.log10(self.rho[self.actind].max())
                out = self.mesh.plotImage(
                    np.log10(self._rho),
                    ax=ax,
                    pcolorOpts={"cmap":"jet"},
                    grid=show_grid,
                    gridOpts={"color": "white", "alpha": 0.5},
                    clim=(vmin, vmax)
                )
                cb = plt.colorbar(out[0], ax=ax, fraction=0.02, orientation='horizontal', ticks=np.linspace(vmin, vmax, 3))
                cb.set_ticklabels([("%.1f")%(10**value) for value in np.linspace(vmin, vmax, 3)])
                ax.plot(self.IO.electrode_locations[:, 0], self.IO.electrode_locations[:, 1], "wo", markeredgecolor='k')
                ax.set_aspect(1)
                ax.set_xlabel("x (m)")
                ax.set_ylabel("z (m)")
                cb.set_label("Resistivity ($\Omega$m)")
                if show_core:
                    ax.set_xlim(self.IO.xyzlim[0,:])
                    ax.set_ylim(self.IO.xyzlim[1,:])
                else:
                    ax.set_ylim(self.mesh.vectorNy.min(), 5)

    def interact_plot_survey(self):
        a=widgets.FloatText(value=10, description='spacing')
        survey_type = widgets.RadioButtons(
            options=["dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"],
            value="dipole-dipole",
            description="array type",
            disabled=False,
        )
        line_length=widgets.FloatText(value=200, description='line length')
        n_spacing = widgets.IntSlider(min=5, max=13, step=1, value=8, description='n-spacing')
        i_src = widgets.IntSlider(min=0, max=10, step=1, value=0, description='src #')
        i_rx = widgets.IntSlider(min=0, max=10, step=1, value=0, description='rx #')
        out = widgets.interactive_output(
            self.plot_survey,
            {
                "a": a,
                "survey_type": survey_type,
                "line_length": line_length,
                "n_spacing": n_spacing,
                "i_src": i_src,
                "i_rx": i_rx,
            },
        )
        grid = GridspecLayout(7, 3, height='300px')
        grid[:4, 1:] = out
        grid[0, 0] = a
        grid[1, 0] = survey_type
        grid[2, 0] = line_length
        grid[3, 0] = n_spacing
        grid[4, 0] = i_src
        grid[5, 0] = i_rx

        return grid

    def interact_plot_model(self):
        std=widgets.FloatText(value=0., description='noise (%)')
        dx = widgets.FloatSlider(
            description="dx", continuous_update=False, min=0, max=500, step=10, value=20
        )
        dy = widgets.FloatSlider(
            description="dz", continuous_update=False, min=0, max=50, step=1, value=10
        )
        xc = widgets.FloatSlider(
            description="xc", continuous_update=False, min=0, max=200, step=1, value=100
        )
        yc = widgets.FloatSlider(
            description="zc", continuous_update=False, min=-50, max=0, step=1, value=-10
        )
        rho0 = widgets.FloatSlider(
            description="$\\rho_0$",
            continuous_update=False,
            min=1,
            max=1000,
            step=1,
            value=1000,
        )
        rho1 = widgets.FloatSlider(
            description="$\\rho_1$",
            continuous_update=False,
            min=1,
            max=1000,
            step=50,
            value=100,
        )

        add_block = widgets.RadioButtons(
            options=["active", "inactive"],
            value="active",
            description="add block",
            disabled=False,
        )
        model_type = widgets.RadioButtons(
            options=["background", "block"],
            value="block",
            description="model type",
            disabled=False,
        )

        show_grid = widgets.Checkbox(
            value=False, description="show grid?", disabled=False
        )
        show_core = widgets.Checkbox(
            value=True, description="show core?", disabled=False
        )
        add_topography = widgets.Checkbox(
            value=False, description="topography?", disabled=False
        )
        simulate = widgets.Checkbox(
            value=False, description="simulate?", disabled=False
        )
        update = widgets.ToggleButton(
            value=False,
            description='Update',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            icon='check'
        )
        obs_name = widgets.Text(
            value='dc.csv',
            placeholder='Type something',
            description='filename:',
            disabled=False,
            continuous_update=False
        )

        write_obs_file = widgets.Checkbox(
            value=False, description="write obs file", disabled=False
        )

        out = widgets.interactive_output(
            self.plot_model,
            {
                "dx": dx,
                "dy": dy,
                "xc": xc,
                "yc": yc,
                "rho0": rho0,
                "rho1": rho1,
                "show_grid": show_grid,
                "show_core": show_core,
                "add_topography": add_topography,
                "simulate": simulate,
                "update": update,
                "std": std,
                "obs_name": obs_name,
                "write_obs_file": write_obs_file,
            },
        )

        grid = GridspecLayout(8, 3, height='400px')
        grid[:5, 1:] = out
        grid[0, 0] = dx
        grid[1, 0] = dy
        grid[2, 0] = xc
        grid[3, 0] = yc
        grid[4, 0] = rho0
        grid[5, 0] = rho1
        grid[6, 0] = std
        grid[7, 0] = obs_name
        grid[5, 1] = show_grid
        grid[6, 1] = show_core
        grid[7, 1] = write_obs_file
        grid[5, 2] = simulate
        grid[6, 2] = add_topography
        grid[7, 2] = update

        return grid

    def simulate(self):
        sigma_air = 1e-10
        actmap = Maps.InjectActiveCells(
            self.mesh,
            indActive=self.actind,
            valInactive=np.log(sigma_air)
        )
        mapping = Maps.ExpMap(self.mesh) * actmap

        # Generate mtrue
        m = np.log(1./self.rho[self.actind])

        if self.survey.ispaired:
            self.survey.unpair()

        if self.survey_type == 'pole-pole':
            problem = DC.Problem2D_CC(
                self.mesh,
                sigmaMap=mapping,
                storeJ=True,
                Solver=Pardiso
            )
        else:
            problem = DC.Problem2D_N(
                self.mesh,
                sigmaMap=mapping,
                storeJ=True,
                Solver=Pardiso
            )
        problem.pair(self.survey)
        data = self.survey.makeSyntheticData(m, std=self.std / 100., force=True)
        self.survey.std = abs(data) * self.std / 100.
        return data

    def plot_data(self, ax=None):
        data = self.simulate()
        self.IO.plotPseudoSection(data=data/self.IO.G, scale='log', ncontour=10, cmap='jet', ax=ax)

    def write_to_csv(self, fname):
        self.IO.write_to_csv(fname, self.survey.dobs, self.survey.std)


class DCRInversionApp(object):
    """docstring for DCRInversionApp"""

    uncertainty = None
    mesh = None
    actind = None
    IO = None
    survey = None
    phi_d = None
    phi_m = None
    dpred = None
    m = None
    sigma_air = 1e-8
    topo = None

    def __init__(self):
        super(DCRInversionApp, self).__init__()
        self.IO = DC.IO()

    def set_mesh(self):

        if self.topo is None:
            self.topo = self.IO.electrode_locations
        tmp_x = np.r_[-1e10, self.topo[:,0], 1e10]
        tmp_z = np.r_[self.topo[0,1], self.topo[:,1], self.topo[-1,1]]
        self.topo = np.c_[tmp_x, tmp_z]
        self.mesh, self.actind = self.IO.set_mesh(topo=self.topo, method='linear')

    def load_obs(self, fname, load, input_type):
        if load:
            try:
                if input_type == 'csv':
                    self.survey = self.IO.read_dc_data_csv(fname)
                elif input_type == 'ubc_dc2d':
                    self.survey = self.IO.read_ubc_dc2d_obs_file(fname)
                print (">> {} is loaded".format(fname))
                print (">> survey type: {}".format(self.IO.survey_type))
                print ("   # of data: {0}".format(self.survey.nD))
                rho_0 = self.get_initial_resistivity()
                print ((">> suggested initial resistivity: %1.f ohm-m")%(rho_0))
                self.set_mesh()
                print (">> 2D tensor mesh is set.")
                print ("   # of cells: {0}".format(self.mesh.nC))
                print ("   # of active cells: {0}".format(self.actind.sum()))
            except:
                print (">> Reading input file is failed!")
                # print (">> {} does not exist!".format(fname))

    def get_problem(self):
        actmap = Maps.InjectActiveCells(
            self.mesh, indActive=self.actind, valInactive=np.log(self.sigma_air)
        )
        mapping = Maps.ExpMap(self.mesh) * actmap
        problem = DC.Problem2D_N(
            self.mesh,
            sigmaMap=mapping,
            storeJ=True,
            Solver=Pardiso
        )
        if self.survey.ispaired:
            self.survey.unpair()
        problem.pair(self.survey)
        return problem

    def get_initial_resistivity(self):
        out = np.histogram(np.log10(abs(self.IO.voltages/self.IO.G)))
        return 10**out[1][np.argmax(out[0])]

    def set_uncertainty(self, percentage, floor, set_value=True):
        self.percentage = percentage
        self.floor = floor

        if set_value:
            self.uncertainty = abs(self.survey.dobs) * percentage / 100.+ floor
            print ((">> percent error: %.1f and floor error: %1.e are set") % (percentage, floor))
        else:
            self.uncertainty = self.survey.std.copy()
            print (">> uncertainty in the observation file is used")
        if np.any(self.uncertainty==0.):
            print ("warning: uncertainty includse zero values!")


    def run_inversion(
        self,
        rho_0,
        rho_ref=None,
        alpha_s=1e-3,
        alpha_x=1,
        alpha_z=1,
        maxIter=20,
        chifact=1.,
        beta0_ratio=1.,
        coolingFactor=5,
        coolingRate=2,
        rho_upper=np.Inf,
        rho_lower=-np.Inf,
        run=True,
    ):
        if run:
            maxIterCG=20
            problem = self.get_problem()
            m0 = np.ones(self.actind.sum()) * np.log(1./rho_0)
            if rho_ref is None:
                rho_ref = rho_0
            mref = np.ones(self.actind.sum()) * np.log(1./rho_ref)
            dmis = DataMisfit.l2_DataMisfit(self.survey)
            dmis.W = 1./self.uncertainty
            reg = Regularization.Tikhonov(
                self.mesh,
                indActive=self.actind,
                alpha_s=alpha_s,
                alpha_x=alpha_x,
                alpha_y=alpha_z,
                mapping=Maps.IdentityMap(nP=self.actind.sum()),
                mref=mref
            )
            # Personal preference for this solver with a Jacobi preconditioner
            opt = Optimization.ProjectedGNCG(maxIter=maxIter, maxIterCG=maxIterCG)
            opt.remember('xc')
            invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
            beta = Directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
            target = Directives.TargetMisfit(chifact=chifact)
            beta_schedule = Directives.BetaSchedule(
                coolingFactor=coolingFactor,
                coolingRate=coolingRate
            )
            save = Directives.SaveOutputEveryIteration()
            save_outputs = Directives.SaveOutputDictEveryIteration()
            sense_weight = Directives.UpdateSensitivityWeights()
            inv = Inversion.BaseInversion(
                invProb,
                directiveList=[beta, target, beta_schedule, save_outputs]
            )

            minv = inv.run(m0)

            self.phi_d = []
            self.phi_m = []
            self.m = []
            self.dpred = []

            for key in save_outputs.outDict.keys():
                self.phi_d.append(save_outputs.outDict[key]['phi_d'].copy()*2.)
                self.phi_m.append(save_outputs.outDict[key]['phi_m'].copy()*2.)
                self.m.append(save_outputs.outDict[key]['m'].copy())
                self.dpred.append(save_outputs.outDict[key]['dpred'].copy())
            os.system("rm -f *.txt")
        else:
            pass

    def interact_load_obs(self):
        obs_name = widgets.Text(
            value='./ubc_dc_data/obs_dc.dat',
            placeholder='Type something',
            description='filename:',
            disabled=False
        )
        load = widgets.Checkbox(
            value=True, description="load", disabled=False
        )
        input_type = widgets.ToggleButtons(
            options=["csv", "ubc_dc2d"],
            value="ubc_dc2d",
            description="input type"
        )
        widgets.interact(self.load_obs, fname=obs_name, load=load, input_type=input_type)

    def plot_obs_data(self, data_type, plot_type):
        fig, ax = plt.subplots(1,1, figsize=(10, 5))
        if plot_type == "pseudo-section":
            self.IO.plotPseudoSection(aspect_ratio=1, cmap='jet', data_type=data_type, ax=ax)
        elif plot_type == "histogram":
            if data_type == "apparent_resistivity":
                out = ax.hist(np.log10(self.IO.apparent_resistivity), edgecolor='k')
                xlabel = 'App. Res ($\Omega$m)'
                xticks = ax.get_xticks()
                ax.set_xticklabels([ ("%.1f")%(10**xtick)for xtick in xticks])

            elif data_type == "volt":
                out = ax.hist(np.log10(abs(self.IO.voltages)), edgecolor='k')
                xlabel = 'Voltage (V)'
                xticks = ax.get_xticks()
                ax.set_xticklabels([ ("%.1e")%(10**xtick)for xtick in xticks])
            ax.set_ylabel('Count')
            ax.set_xlabel(xlabel)

    def plot_misfit_curve(self, iteration, scale='linear', curve_type='misfit'):
        fig, ax = plt.subplots(1,1, figsize=(10, 5))
        if curve_type == "misfit":
            ax_1 = ax.twinx()
            ax.plot(np.arange(len(self.phi_m))+1, self.phi_d, 'k.-')
            ax_1.plot(np.arange(len(self.phi_d))+1, self.phi_m, 'r.-')
            ax.plot(iteration, self.phi_d[iteration-1], 'ko', ms=10)
            ax_1.plot(iteration, self.phi_m[iteration-1], 'ro', ms=10)

            xlim = plt.xlim()
            ax.plot(xlim, np.ones(2)*self.survey.nD, 'k--')
            ax.set_xlim(xlim)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("$\phi_d$", fontsize=16)
            ax_1.set_ylabel("$\phi_m$", fontsize=16)
            ax.set_yscale(scale)
            ax_1.set_yscale(scale)
            ax.set_title(("Misfit / Target misfit: %1.f / %.1f")%(self.phi_d[iteration-1], self.survey.nD))
        elif curve_type == "tikhonov":
            ax.plot(self.phi_m, self.phi_d, 'k.-')
            ax.plot(self.phi_m[iteration-1], self.phi_d[iteration-1], 'ko', ms=10)
            ax.set_ylabel("$\phi_d$", fontsize=16)
            ax.set_xlabel("$\phi_m$", fontsize=16)
            ax.set_xscale(scale)
            ax.set_yscale(scale)

    def plot_data_misfit(self, iteration):
        dobs = self.survey.dobs
        appres = dobs/self.IO.G
        vmin, vmax = appres.min(), appres.max()
        dpred = self.dpred[iteration-1]
        fig, axs = plt.subplots(3,1, figsize = (10, 9))
        self.IO.plotPseudoSection(data=appres, clim=(vmin, vmax), aspect_ratio=1, ax=axs[0], cmap='jet', scale='log')
        self.IO.plotPseudoSection(data=dpred/self.IO.G, clim=(vmin, vmax), aspect_ratio=1, ax=axs[1], cmap='jet', scale='log')
        misfit = (dpred-dobs) / self.uncertainty
        self.IO.plotPseudoSection(
            data=misfit, data_type='volt', scale='linear', aspect_ratio=1, ax=axs[2], clim=(-3, 3),
            label='Normalized Misfit', cmap='jet'
        )
        titles = ["Observed", "Predicted", "Normalized misfit"]
        for i_ax, ax in enumerate(axs):
            ax.set_title(titles[i_ax])

    def plot_model(self, iteration, vmin=None, vmax=None, show_core=True, show_grid=False):
        clim = (vmin, vmax)
        # inds_core, self. = Utils.ExtractCoreMesh(self.IO.xyzlim, self.mesh)
        fig, ax = plt.subplots(1,1, figsize=(10, 5))
        tmp = np.log10(1./(self.survey.prob.sigmaMap*self.m[iteration-1]))
        tmp[~self.actind] = np.nan
        if clim is None:
            vmin, vmax = tmp[self.actind].min(), tmp[self.actind].max()
        else:
            vmin, vmax = np.log10(clim[0]), np.log10(clim[1])
        out = self.mesh.plotImage(
            tmp, grid=show_grid, clim=(vmin, vmax), pcolorOpts={'cmap':'jet'}, ax=ax,
            gridOpts={"color": "white", "alpha": 0.5}
        )
        ticks = np.linspace(vmin, vmax, 3)
        cb = plt.colorbar(out[0], orientation='horizontal', fraction=0.03, ticks=ticks, ax=ax)
        cb.set_ticklabels([("%.1f")%(10**tick)for tick in ticks])
        ax.plot(self.IO.electrode_locations[:,0], self.IO.electrode_locations[:,1], 'wo', markeredgecolor='k')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_aspect(1)
        if show_core:
            ymin, ymax = self.IO.xyzlim[1,:]
            xmin, xmax = self.IO.xyzlim[0,:]
            dy = (ymax-ymin)/10.
            ax.set_ylim(ymin, ymax+dy)
            ax.set_xlim(xmin, xmax)
        else:
            ymin, ymax = self.mesh.vectorNy.min(), self.mesh.vectorNy.max()
            xmin, xmax = self.mesh.vectorNx.min(), self.mesh.vectorNx.max()
            dy = (ymax-ymin)/10.
            ax.set_ylim(ymin, ymax+dy)
            ax.set_xlim(xmin, xmax+dy)

        plt.tight_layout()

    def plot_inversion_results(
        self,
        iteration=1,
        curve_type='misfit',
        scale='log',
        plot_type='misfit_curve',
        rho_min=100,
        rho_max=1000,
        show_grid=False,
        show_core=True,
    ):
        if plot_type == "misfit_curve":
            self.plot_misfit_curve(iteration, curve_type=curve_type, scale=scale)
        elif plot_type == "model":
            self.plot_model(iteration, vmin=rho_min, vmax=rho_max, show_core=show_core, show_grid=show_grid)
        elif plot_type == "data_misfit":
            self.plot_data_misfit(iteration)
        else:
            raise NotImplementedError()

    def interact_plot_obs_data(self):
        data_type = widgets.ToggleButtons(
            options=["apparent_resistivity", "volt"],
            value="apparent_resistivity",
            description="data type"
        )
        plot_type = widgets.ToggleButtons(
            options=["pseudo-section", "histogram"],
            value="pseudo-section",
            description="plot type"
        )
        widgets.interact(
            self.plot_obs_data,
            data_type=data_type,
            plot_type=plot_type,
        )

    def interact_set_uncertainty(self):
        percentage = widgets.FloatText(value=5.)
        floor = widgets.FloatText(value=0.)
        widgets.interact(
            self.set_uncertainty,
            percentage=percentage,
            floor=floor
        )

    def interact_run_inversion(self):
        run = widgets.Checkbox(
            value=True, description="run", disabled=False
        )

        rho_initial = np.ceil(self.get_initial_resistivity()                )
        maxIter=widgets.IntText(value=30, continuous_update=False)
        rho_0=widgets.FloatText(
            value=rho_initial, continuous_update=False,
            description="$\\rho_0$"
        )
        rho_ref=widgets.FloatText(
            value=rho_initial, continuous_update=False,
            description="$\\rho_{ref}$"
            )
        percentage=widgets.FloatText(value=self.percentage, continuous_update=False)
        floor=widgets.FloatText(value=self.floor, continuous_update=False)
        chifact=widgets.FloatText(value=1.0, continuous_update=False)
        beta0_ratio=widgets.FloatText(value=10., continuous_update=False)
        coolingFactor=widgets.FloatSlider(
            min=0.1, max=10, step=1, value=2, continuous_update=False
        )
        coolingRate=widgets.IntSlider(
            min=1, max=10, step=1, value=1, continuous_update=False,
            description='n_iter / beta'
        )
        alpha_s=widgets.FloatText(
            value=1e-10, continuous_update=False,
            description="$\\alpha_{s}$"
        )
        alpha_x=widgets.FloatText(
            value=1, continuous_update=False,
            description="$\\alpha_{x}$"
        )
        alpha_z=widgets.FloatText(
            value=1, continuous_update=False,
            description="$\\alpha_{z}$"
        )

        widgets.interact(
            self.run_inversion,
            run=run,
            rho_initial=rho_initial,
            maxIter=maxIter,
            rho_0=rho_0,
            rho_ref=rho_ref,
            percentage=percentage,
            floor=floor,
            chifact=chifact,
            beta0_ratio=beta0_ratio,
            coolingFactor=coolingFactor,
            coolingRate=coolingRate,
            alpha_s=alpha_s,
            alpha_x=alpha_x,
            alpha_z=alpha_z
        )

    def interact_plot_inversion_results(self):
        iteration = widgets.IntSlider(
            min=1, max=len(self.m), step=1, value=1, continuous_update=False
        )
        curve_type = widgets.ToggleButtons(
            options=["misfit", "tikhonov"],
            value="misfit",
            description="curve type"
        )
        scale=widgets.ToggleButtons(
            options=["linear", "log"],
            value="linear",
            description="scale"
        )
        plot_type = widgets.ToggleButtons(
            options=["misfit_curve", "model", "data_misfit"],
            value="misfit_curve",
            description="plot type"
        )
        rho = 1./np.exp(self.m[-1])
        rho_min=widgets.FloatText(
            value=np.ceil(rho.min()), continuous_update=False,
            description="$\\rho_{min}$"
        )
        rho_max=widgets.FloatText(
            value=np.ceil(rho.max()), continuous_update=False,
            description="$\\rho_{max}$"
        )

        show_grid = widgets.Checkbox(
            value=False, description="show grid?", disabled=False
        )
        show_core = widgets.Checkbox(
            value=True, description="show core?", disabled=False
        )

        widgets.interact(
            self.plot_inversion_results,
            iteration=iteration,
            curve_type=curve_type,
            scale=scale,
            plot_type=plot_type,
            rho_min=rho_min,
            rho_max=rho_max,
            show_grid=show_grid,
            show_core=show_core
        )
