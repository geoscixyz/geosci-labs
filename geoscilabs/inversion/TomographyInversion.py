import numpy as np
from SimPEG import (
    Mesh, Maps, Utils, Problem, Survey, DataMisfit,
    Directives, Optimization, Regularization,
    InvProblem, Inversion,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from pymatsolver import Pardiso
import matplotlib
from ipywidgets import (
    interact, FloatSlider, ToggleButtons, IntSlider, FloatText, IntText, SelectMultiple
)
import ipywidgets as widgets
from SimPEG.SEIS import StraightRay
from pylab import hist


class TomographyInversionApp(object):
    """docstring for TomographyInversionApp"""

    # Parameters for sensitivity matrix, G
    # Parameters for Model
    seed = None
    percentage = None
    floor = None
    uncertainty = None

    def __init__(self):
        super(TomographyInversionApp, self).__init__()
        self.get_problem_survey()

    @property
    def problem(self):
        return self._problem

    @property
    def survey(self):
        return self._survey

    @property
    def src_locations(self):
        return self._src_locations

    @property
    def rx_locations(self):
        return self._rx_locations

    @property
    def slowness(self):
        return self._slowness

    @property
    def mesh(self):
        return self._mesh

    def get_problem_survey(self, nx=20, ny=20, dx=10, dy=20):
        hx = np.ones(nx) * dx
        hy = np.ones(ny) * dy
        self._mesh = Mesh.TensorMesh([hx, hy])
        y = np.linspace(self._mesh.x0[1], dy*ny, ny/2)
        self._src_locations = np.c_[y*0+self._mesh.vectorCCx[0], y]
        self._rx_locations = np.c_[y*0+self._mesh.vectorCCx[-1], y]
        rx = StraightRay.Rx(self._rx_locations, None)
        srcList = [
            StraightRay.Src(loc=self._src_locations[i, :], rxList=[rx])
            for i in range(y.size)
        ]
        self._survey = StraightRay.Survey(srcList)
        self._problem = StraightRay.Problem(self._mesh, slownessMap=Maps.IdentityMap(self._mesh))
        self._problem.pair(self._survey)

    @property
    def pred(self):
        self._pred = self.survey.dpred(self.slowness)
        return self._pred

    @property
    def dobs(self):
        return self._dobs

    @property
    def percentage(self):
        return self._percentage

    @property
    def floor(self):
        return self._floor

    @property
    def seed(self):
        return self._seed

    def add_noise(self):
        np.random.seed(self.seed)

        noise = (
            np.random.randn(self.survey.nD) * self.pred * self.percentage * 0.01 +
            np.random.randn(self.survey.nD) * self.floor
        )
        return self.pred + noise

    def get_block_index(
        self,
        xc=50,
        yc=50,
        dx=20,
        dy=20
    ):
        p0 = np.array([xc-dx/2., yc+dy/2])
        p1 = np.array([xc+dx/2., yc-dy/2])
        index = Utils.ModelBuilder.getIndicesBlock(p0, p1, self.mesh.gridCC)
        return index

    def get_block_points(
        self,
        xc=50,
        yc=50,
        dx=20,
        dy=20
    ):
        x = np.array([xc-dx/2., xc+dx/2., xc+dx/2., xc-dx/2., xc-dx/2.])
        y = np.array([yc-dy/2., yc-dy/2., yc+dy/2., yc+dy/2., yc-dy/2.])
        return x, y

    def plot_model(self, v0, v1, xc, yc, dx, dy, model_type="background", add_block="inactive"):
        fig, ax = plt.subplots(1,1)
        self._slowness = np.ones(self.mesh.nC) * 1./v0
        if model_type == "background":
            pass
        elif model_type == "block":
            x, y = self.get_block_points(xc, yc, dx, dy)
            ax.plot(x, y, 'w-')
            if add_block == "active":
                index = self.get_block_index(xc=xc, yc=yc, dx=dx, dy=dy)
                self._slowness[index] = 1./v1
        out = self.mesh.plotImage(1./self._slowness, ax=ax)
        cb = plt.colorbar(out[0], ax=ax, fraction=0.02)
        ax.plot(self.rx_locations[:,0], self.rx_locations[:,1], 'wv')
        ax.plot(self.src_locations[:,0], self.src_locations[:,1], 'w*')
        ax.set_aspect(1)

    def plot_survey_data(self, percentage, floor, seed, add_noise, plot_type):
        self._percentage = percentage
        self._floor = floor
        self._seed = seed
        self._dobs = self.add_noise()
        self.survey.dobs = self._dobs.copy()

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        out = self.mesh.plotImage(1./self.slowness, ax=axs[0])
        cb = plt.colorbar(out[0], ax=axs[0], fraction=0.02)
        cb.set_label("Velocity (m/s)")
        self.survey.plot(ax=axs[0])
        axs[0].set_title('Survey')
        x = np.arange(10) + 1
        y = np.arange(10) + 1
        xy = Utils.ndgrid(x, y)
        if plot_type == 'tx_rx_plane':
            if add_noise:
                out = Utils.plot2Ddata(xy, self.dobs, ax=axs[1])
            else:
                out = Utils.plot2Ddata(xy, self.pred, ax=axs[1])
            axs[1].set_xlabel("Rx")
            axs[1].set_ylabel("Tx")
            axs[1].set_xticks(x)
            axs[1].set_yticks(y)
            cb = plt.colorbar(out[0], ax=axs[1], fraction=0.02)
            cb.set_label("Traveltime (s)")
            for ax in axs:
                ax.set_aspect(1)
        else:
            if add_noise:
                out = axs[1].hist(self.pred)
            else:
                out = axs[1].hist(self.dobs)
            axs[1].set_ylabel("Count")
            axs[1].set_xlabel("Travel time (s)")
            axs[0].set_aspect(1)
        plt.tight_layout()

    def interact_plot_model(self):
        dx = widgets.FloatSlider(description='dx', continuous_update=False, min=0, max=400, step=10, value=50)
        dy = widgets.FloatSlider(description='dy', continuous_update=False, min=0, max=400, step=10, value=50)
        xc = widgets.FloatSlider(description='xc', continuous_update=False, min=0, max=400, step=1, value=100)
        yc = widgets.FloatSlider(description='yc', continuous_update=False, min=0, max=400, step=1, value=200)
        v0 = widgets.FloatSlider(description='v0', continuous_update=False, min=500, max=3000, step=50, value=1000)
        v1 = widgets.FloatSlider(description='v1', continuous_update=False, min=500, max=3000, step=50, value=2000)
        add_block = widgets.RadioButtons(
            options=["active", "inactive"],
            value='active',
            description='add block',
            disabled=False
        )
        model_type = widgets.RadioButtons(
            options=["background", "block"],
            value='block',
            description='model type',
            disabled=False
        )
        out = widgets.interactive_output(
            self.plot_model,
            {
                'dx': dx,
                'dy': dy,
                'xc': xc,
                'yc': yc,
                'v0': v0,
                'v1': v1,
                'add_block': add_block,
                'model_type': model_type,
            }
        )
        return widgets.HBox([widgets.VBox([v0, v1, xc, yc, dx, dy]), out, widgets.VBox([add_block, model_type])])

    def interact_data(self):
        percentage = widgets.BoundedFloatText(
            value=5,
            min=0,
            max=100.,
            step=1,
            disabled=False,
            description='percent ($\%$):',
        )

        floor = widgets.FloatText(
            value=1e-2,
            min=0.,
            max=10.,
            step=1,
            description='floor (s):',
            disabled=False
        )
        seed = widgets.IntText(
            value=1,
            min=1,
            max=10,
            step=1,
            description='random seed',
            disabled=False
        )

        add_noise = widgets.Checkbox(
            value=False,
            description='add noise?',
            disabled=False
        )

        plot_type = widgets.ToggleButtons(
            options=['tx_rx_plane', 'histogram'],
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['tx-rx plane', 'histogram'],
        )
        out = widgets.interactive_output(
            self.plot_survey_data,
            {
                'percentage': percentage,
                'floor': floor,
                'seed': seed,
                'add_noise': add_noise,
                'plot_type': plot_type,
            }
        )
        form_item_layout = widgets.Layout(
            display='flex',
            flex_flow='row',
            justify_content='space-between'
        )

        form_items = [
            widgets.Box([percentage], layout=form_item_layout),
            widgets.Box([floor], layout=form_item_layout),
            widgets.Box([seed], layout=form_item_layout),
            widgets.Box([add_noise], layout=form_item_layout),
            widgets.Box([plot_type], layout=form_item_layout)
        ]

        form = widgets.Box(form_items, layout=widgets.Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px',
            align_items='stretch',
            width='40%',
            height='30%',
            margin='5%'
        ))
        return widgets.HBox([out, form])

    def run_inversion(
        self,
        maxIter=60,
        m0=0.,
        mref=0.,
        percentage=5,
        floor=0.1,
        chifact=1,
        beta0_ratio=1.,
        coolingFactor=1,
        coolingRate=1,
        alpha_s=1.,
        alpha_x=1.,
        alpha_z=1.,
        use_target=False,
        use_lplq=False,
        p=2,
        qx=2,
        qy=2,
        qz=2,
        beta_start=None
    ):

        self.uncertainty = percentage*abs(self.survey.dobs)*0.01 + floor

        m0 = np.ones(self.mesh.nC) * m0
        mref = np.ones(self.mesh.nC) * mref
        reg = Regularization.Sparse(
            self.mesh,
            alpha_s =alpha_s,
            alpha_x =alpha_x,
            alpha_y =alpha_z,
            mref =mref,
            mapping = Maps.IdentityMap(self.mesh),
            cell_weights = self.mesh.vol
        )
        dmis = DataMisfit.l2_DataMisfit(self.survey)
        dmis.W =1./self.uncertainty

        opt = Optimization.InexactGaussNewton(
            maxIter =maxIter,
            maxIterCG =20
        )
        opt.remember('xc')
        opt.tolG =1e-10
        opt.eps =1e-10
        invProb =InvProblem.BaseInvProblem(dmis, reg, opt)
        save =Directives.SaveOutputEveryIteration()
        beta_schedule =Directives.BetaSchedule(
            coolingFactor =coolingFactor,
            coolingRate =coolingRate
        )
        update_Jacobi = Directives.UpdatePreconditioner()

        if use_lplq:
            IRLS = Directives.Update_IRLS(
                f_min_change=1e-4,
                minGNiter=1,
                silent=False,
                maxIRLSiter=40,
                beta_tol=5e-1,
                coolEpsFact=1.3,
                chifact_start=chifact
            )

            if beta_start is None:
                directives = [
                    Directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio),
                    IRLS,
                    save
                ]
            else:
                directives = [
                    IRLS,
                    save
                ]
                invProb.beta = beta_start
            reg.norms = np.c_[p, qx, qz, 2]
        else:
            target = Directives.TargetMisfit(chifact=chifact)
            directives = [
                Directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio),
                beta_schedule,
                save
            ]
            if use_target:
                directives.append(target)

        inv = Inversion.BaseInversion(invProb, directiveList=directives)
        mopt = inv.run(m0)
        model = opt.recall('xc')
        model.append(mopt)
        pred = []
        for m in model:
            pred.append(self.survey.dpred(m))
        return model, pred, save

    def plot_model_inversion(self, ii, model, fixed=False):
        fig, axs = plt.subplots(1, 2)
        if fixed:
            clim = (1./self.slowness).min(), (1./self.slowness).max()
        else:
            clim = None
        out = self.mesh.plotImage(1./model[ii], ax=axs[0], clim=clim)
        cb = plt.colorbar(out[0], ax=axs[0], fraction=0.02)
        out = self.mesh.plotImage(1./self.slowness, ax=axs[1], clim=clim)
        cb = plt.colorbar(out[0], ax=axs[1], fraction=0.02)
        axs[1].set_aspect(1)
        for ax in axs:
            ax.plot(self.rx_locations[:,0], self.rx_locations[:,1], 'wv')
            ax.plot(self.src_locations[:,0], self.src_locations[:,1], 'w*')
            ax.set_aspect(1)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("z (m)")
        plt.tight_layout()

    def plot_data_inversion(self, ii, pred, fixed=False):
        titles = ["Observed", "Predicted", "Normalized misfit"]
        x = np.arange(10) + 1
        y = np.arange(10) + 1
        xy = Utils.ndgrid(x, y)
        fig, axs = plt.subplots(1,3, figsize=(10, 3))
        if fixed:
            clim = (self.dobs.min(),self.dobs.max())
        else:
            clim = None
        out = Utils.plot2Ddata(xy, self.dobs, ax=axs[0], clim=clim)
        cb = plt.colorbar(out[0], ax=axs[0], fraction=0.02)
        out = Utils.plot2Ddata(xy, pred[ii], ax=axs[1], clim=clim)
        cb = plt.colorbar(out[0], ax=axs[1], fraction=0.02)
        out = Utils.plot2Ddata(xy, (pred[ii]-self.dobs)/self.uncertainty, ax=axs[2], clim=(-1,1))
        cb = plt.colorbar(out[0], ax=axs[2], fraction=0.02, ticks=[-1, 0, 1])
        for ii, ax in enumerate(axs):
            ax.set_aspect(1)
            ax.set_title(titles[ii])
            ax.set_xlabel('Rx')
            ax.set_ylabel('Tx')
        plt.tight_layout()

    def interact_model_inversion(self, model):
        def foo(ii, fixed=False):
            self.plot_model_inversion(ii, model, fixed=fixed)
        interact(
            foo,
            ii=IntSlider(min=0, max=len(model)-1, step=1, value=len(model)-1),
            continuous_update=False
        )

    def interact_data_inversion(self, pred):
        def foo(ii, fixed=False):
            self.plot_data_inversion(ii, pred, fixed=fixed)
        interact(
            foo,
            ii=IntSlider(
                min=0, max=len(pred)-1, step=1, value=len(pred)-1,
                continuous_update=False
            )
        )
