import numpy as np
import warnings
from pymatsolver import Pardiso
from SimPEG import DC, Utils, Maps
import matplotlib.pyplot as plt
import matplotlib

warnings.filterwarnings("ignore")
matplotlib.rcParams['font.size'] = 14

from ipywidgets import GridspecLayout, widgets

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
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
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
            self.plot_data()
            if write_obs_file:
                self.write_ubc_obs_file(obs_name)
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
                cb = plt.colorbar(out[0], ax=ax, fraction=0.03, orientation='horizontal', ticks=np.linspace(vmin, vmax, 3))
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
        n_spacing = widgets.IntSlider(min=5, max=18, step=1, value=8, description='n-spacing')
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

#         return widgets.HBox(
#             [
#                 widgets.VBox([a, survey_type, line_length, n_spacing, i_src, i_rx]),
#                 out,
#             ]
#         )

    def interact_plot_model(self):
        std=widgets.FloatText(value=0., description='noise (%)')
        dx = widgets.FloatSlider(
            description="dx", continuous_update=False, min=0, max=500, step=10, value=20
        )
        dy = widgets.FloatSlider(
            description="dz", continuous_update=False, min=0, max=50, step=10, value=10
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
            value=False, description="show core?", disabled=False
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
            value='dc.obs',
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
        from SimPEG import Maps
        # Use Exponential Map: m = log(rho)
        sigma_air = 1e-10
        actmap = Maps.InjectActiveCells(
            self.mesh, indActive=self.actind, valInactive=np.log(sigma_air)
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

    def plot_data(self):
        data = self.simulate()
        self.IO.plotPseudoSection(data=data/self.IO.G, scale='log', ncontour=10, cmap='jet')

    def write_ubc_obs_file(self, fname):
        DC.Utils.writeUBC_DCobs(
            'test.obs', self.survey, 2,
            'GENERAL',
            survey_type=self.survey_type
        )
