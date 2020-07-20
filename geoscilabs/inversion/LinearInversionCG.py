import numpy as np
import discretize
from SimPEG.simulation import LinearSimulation
from SimPEG.survey import LinearSurvey
from SimPEG.data import Data
from SimPEG import data_misfit
from SimPEG import directives
from SimPEG import optimization
from SimPEG import regularization
from SimPEG import inverse_problem
from SimPEG import inversion
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from pymatsolver import Pardiso
import matplotlib
from ipywidgets import (
    interact,
    FloatSlider,
    ToggleButtons,
    IntSlider,
    FloatText,
    IntText,
    SelectMultiple,
    RadioButtons,
)
import ipywidgets as widgets


class LinearinversionCGApp(object):
    """docstring for LinearinversionApp"""

    # Parameters for sensitivity matrix, G
    N = None
    M = None
    j_start = None
    j_end = None
    p = None
    q = None
    seed = None

    # Parameters for Model
    m_background = None
    m1 = None
    m2 = None
    m1_center = None
    dm1 = None
    m2_center = None
    dm2 = None
    sigma = None
    m_min = None
    m_max = None

    data = None
    save = None

    def __init__(self):
        super(LinearinversionCGApp, self).__init__()

    @property
    def G(self):
        return self._G

    @property
    def jk(self):
        return self._jk

    @property
    def mesh(self):
        return self._mesh

    def set_G(self, N=20, M=100, p=-0.25, q=0.25, j1=1, jn=60):
        """
        Parameters
        ----------
        N: # of data
        M: # of model parameters
        ...

        """
        self.N = N
        self.M = M
        self._mesh = discretize.TensorMesh([M])
        jk = np.linspace(j1, jn, N)
        self._G = np.zeros((N, self.mesh.nC), dtype=float, order="C")

        def g(k):
            return np.exp(p * jk[k] * self.mesh.vectorCCx) * np.cos(
                np.pi * q * jk[k] * self.mesh.vectorCCx
            )

        for i in range(N):
            self._G[i, :] = g(i) * self.mesh.hx
        self._jk = jk

    def plot_G(
        self,
        N=20,
        M=100,
        p=-0.25,
        q=0.25,
        j1=1,
        jn=60,
        scale="log",
        fixed=False,
        ymin=-0.001,
        ymax=0.011,
    ):
        self.set_G(N=N, M=M, p=p, q=q, j1=j1, jn=jn)

        _, s, _ = np.linalg.svd(self.G, full_matrices=False)

        matplotlib.rcParams["font.size"] = 14

        plt.figure(figsize=(10, 4))

        gs1 = gridspec.GridSpec(1, 4)
        ax1 = plt.subplot(gs1[0, :3])
        ax2 = plt.subplot(gs1[0, 3:])

        ax1.plot(self.mesh.vectorCCx, self.G.T)
        if fixed:
            ax1.set_ylim(ymin, ymax)
        ax1.set_xlabel("x")
        ax1.set_ylabel("g(x)")

        ax2.plot(np.arange(self.N) + 1, s, "ro")
        ax2.set_xlabel("")
        ax2.set_title("singular values", fontsize=12)
        ax2.set_xscale(scale)
        ax2.set_yscale(scale)
        ax2.xaxis.set_major_locator(plt.NullLocator())
        ax2.xaxis.set_minor_locator(plt.NullLocator())
        ax2.xaxis.set_major_formatter(plt.NullFormatter())
        ax2.xaxis.set_minor_formatter(plt.NullFormatter())

        plt.tight_layout()
        plt.show()

    def set_model(
        self,
        m_background=0.0,
        m1=1.0,
        m2=-1.0,
        m1_center=0.2,
        dm1=0.2,
        m2_center=0.5,
        sigma_2=1.0,
    ):
        m = np.zeros(self.mesh.nC) + m_background
        m1_inds = np.logical_and(
            self.mesh.vectorCCx > m1_center - dm1 / 2.0,
            self.mesh.vectorCCx < m1_center + dm1 / 2.0,
        )
        m[m1_inds] = m1

        def gaussian(x, x0, sigma):
            return np.exp(-np.power((x - x0) / sigma, 2.0) / 2.0)

        m += gaussian(self.mesh.vectorCCx, m2_center, sigma_2) * m2
        return m

    def plot_model(
        self,
        m_background=0.0,
        m1=1.0,
        m1_center=0.2,
        dm1=0.2,
        m2=-1.0,
        m2_center=0.5,
        sigma_2=1.0,
        option="model",
        add_noise=True,
        percentage=10,
        floor=1e-1,
    ):

        m = self.set_model(
            m_background=m_background,
            m1=m1,
            m2=m2,
            m1_center=m1_center,
            dm1=dm1,
            m2_center=m2_center,
            sigma_2=sigma_2,
        )

        np.random.seed(1)

        if add_noise:
            survey, _ = self.get_simulation()
            data = survey.dpred(m)
            noise = (
                abs(data) * percentage * 0.01 * np.random.randn(self.N)
                + np.random.randn(self.N) * floor
            )
        else:
            survey, _ = self.get_simulation()
            data = survey.dpred(m)
            noise = np.zeros(self.N, float)

        data += noise
        self.data = data.copy()
        self.m = m.copy()
        self.uncertainty = abs(self.data) * percentage * 0.01 + floor
        self.percentage = percentage
        self.floor = floor

        option_bools = [False, False, False]

        for item in option:
            if item == "kernel":
                option_bools[0] = True
            elif item == "model":
                option_bools[1] = True
            elif item == "data":
                option_bools[2] = True

        fig, axes = plt.subplots(1, 3, figsize=(12 * 1.2, 3 * 1.2))
        for i, ax in enumerate(axes):
            if option_bools[i]:
                if i == 0:
                    ax.plot(self.mesh.vectorCCx, self.G.T)
                    ax.set_title("Rows of matrix G")
                    ax.set_xlabel("x")
                    ax.set_ylabel("g(x)")
                elif i == 1:
                    ax.plot(self.mesh.vectorCCx, m)
                    ax.set_ylim([-2.5, 2.5])
                    ax.set_title("Model")
                    ax.set_xlabel("x")
                    ax.set_ylabel("m(x)")
                    ax.set_ylabel("$d_j$")
                elif i == 2:
                    if add_noise:
                        # this is just for visualization of uncertainty
                        ax.errorbar(
                            x=self.jk,
                            y=self.data,
                            yerr=self.uncertainty,
                            color="k",
                            lw=1,
                        )
                        ax.plot(self.jk, self.data, "ko")
                    else:
                        ax.plot(self.jk, self.data, "ko-")

                    ax.set_title("Data")
                    ax.set_xlabel("$k_j$")

        for i, ax in enumerate(axes):
            if not option_bools[i]:
                ax.axis("off")
                # ax.xaxis.set_minor_locator(plt.NullLocator())
                # ax.xaxis.set_major_formatter(plt.NullFormatter())
                # ax.xaxis.set_minor_formatter(plt.NullFormatter())
                # ax.yaxis.set_major_locator(plt.NullLocator())
                # ax.yaxis.set_minor_locator(plt.NullLocator())
                # ax.yaxis.set_major_formatter(plt.NullFormatter())
                # ax.yaxis.set_minor_formatter(plt.NullFormatter())
        plt.tight_layout()

    def get_simulation(self):
        survey = LinearSurvey()
        simulation = LinearSimulation(self.mesh, G=self.G, survey=survey)
        return simulation

    def run_inversion_cg(
        self,
        maxIter=60,
        m0=0.0,
        mref=0.0,
        percentage=5,
        floor=0.1,
        chifact=1,
        beta0_ratio=1.0,
        coolingFactor=1,
        coolingRate=1,
        alpha_s=1.0,
        alpha_x=1.0,
        use_target=False,
    ):
        sim = self.get_simulation()
        data = Data(
            sim.survey, dobs=self.data, relative_error=percentage, noise_floor=floor
        )
        self.uncertainty = data.uncertainty

        m0 = np.ones(self.M) * m0
        mref = np.ones(self.M) * mref
        reg = regularization.Tikhonov(
            self.mesh, alpha_s=alpha_s, alpha_x=alpha_x, mref=mref
        )
        dmis = data_misfit.L2DataMisfit(data=data, simulation=sim)

        opt = optimization.InexactGaussNewton(maxIter=maxIter, maxIterCG=20)
        opt.remember("xc")
        opt.tolG = 1e-10
        opt.eps = 1e-10
        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
        save = directives.SaveOutputEveryIteration()
        beta_schedule = directives.BetaSchedule(
            coolingFactor=coolingFactor, coolingRate=coolingRate
        )
        target = directives.TargetMisfit(chifact=chifact)

        if use_target:
            directs = [
                directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio),
                beta_schedule,
                target,
                save,
            ]
        else:
            directs = [
                directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio),
                beta_schedule,
                save,
            ]
        inv = inversion.BaseInversion(invProb, directiveList=directs)
        mopt = inv.run(m0)
        model = opt.recall("xc")
        model.append(mopt)
        pred = []
        for m in model:
            pred.append(sim.dpred(m))
        return model, pred, save

    def plot_inversion(
        self,
        mode="Run",
        maxIter=60,
        m0=0.0,
        mref=0.0,
        percentage=5,
        floor=0.1,
        chifact=1,
        beta0_ratio=1.0,
        coolingFactor=1,
        coolingRate=1,
        alpha_s=1.0,
        alpha_x=1.0,
        use_target=False,
        option="model",
        i_iteration=1,
    ):

        if mode == "Run":
            self.model, self.pred, self.save = self.run_inversion_cg(
                maxIter=maxIter,
                m0=m0,
                mref=mref,
                percentage=percentage,
                floor=floor,
                chifact=chifact,
                beta0_ratio=beta0_ratio,
                coolingFactor=coolingFactor,
                coolingRate=coolingRate,
                alpha_s=alpha_s,
                alpha_x=alpha_x,
                use_target=use_target,
            )
        if len(self.model) == 2:
            fig, axes = plt.subplots(1, 2, figsize=(14 * 1.2 * 2 / 3, 3 * 1.2))
            i_plot = -1
        else:
            self.save.load_results()
            if self.save.i_target is None:
                i_plot = -1
            else:
                i_plot = self.save.i_target + 1
            fig, axes = plt.subplots(1, 3, figsize=(14 * 1.2, 3 * 1.2))

        axes[0].plot(self.mesh.vectorCCx, self.m)
        if mode == "Run":
            axes[0].plot(self.mesh.vectorCCx, self.model[i_plot])
        axes[0].set_ylim([-2.5, 2.5])
        axes[1].errorbar(x=self.jk, y=self.data, yerr=self.uncertainty, color="k", lw=1)
        axes[1].plot(self.jk, self.data, "ko")
        if mode == "Run":
            axes[1].plot(self.jk, self.pred[i_plot], "bx")
        axes[1].legend(("Observed", "Predicted"))
        axes[0].legend(("True", "Pred"))
        axes[0].set_title("Model")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("m(x)")

        axes[1].set_title("Data")
        axes[1].set_xlabel("$k_j$")
        axes[1].set_ylabel("$d_j$")

        if len(self.model) > 2:
            max_iteration = len(self.model) - 1
            if i_iteration > max_iteration:
                print(
                    (
                        ">> Warning: input iteration (%i) is greater than maximum iteration (%i)"
                    )
                    % (i_iteration, len(self.model) - 1)
                )
                i_iteration = max_iteration

            if option == "misfit":
                if mode == "Explore":
                    axes[0].plot(self.mesh.vectorCCx, self.model[i_iteration])
                    axes[1].plot(self.jk, self.pred[i_iteration], "bx")
                    # axes[0].legend(("True", "Pred", ("%ith")%(i_iteration)))
                    # axes[1].legend(("Observed", "Predicted", ("%ith")%(i_iteration)))
                    axes[1].legend(("Observed", "Predicted"))

                    if i_iteration == 0:
                        i_iteration = 1
                    axes[2].plot(
                        np.arange(len(self.save.phi_d))[i_iteration - 1] + 1,
                        self.save.phi_d[i_iteration - 1] * 2,
                        "go",
                        ms=10,
                    )

                ax_1 = axes[2].twinx()
                axes[2].semilogy(
                    np.arange(len(self.save.phi_d)) + 1, self.save.phi_d * 2, "k-", lw=2
                )
                if self.save.i_target is not None:
                    axes[2].plot(
                        np.arange(len(self.save.phi_d))[self.save.i_target] + 1,
                        self.save.phi_d[self.save.i_target] * 2,
                        "k*",
                        ms=10,
                    )
                    axes[2].plot(
                        np.r_[axes[2].get_xlim()[0], axes[2].get_xlim()[1]],
                        np.ones(2) * self.save.target_misfit * 2,
                        "k:",
                    )

                ax_1.semilogy(
                    np.arange(len(self.save.phi_d)) + 1, self.save.phi_m, "r", lw=2
                )
                axes[2].set_xlabel("Iteration")
                axes[2].set_ylabel("$\phi_d$")
                ax_1.set_ylabel("$\phi_m$", color="r")
                for tl in ax_1.get_yticklabels():
                    tl.set_color("r")
                axes[2].set_title("Misfit curves")

            elif option == "tikhonov":
                if mode == "Explore":
                    axes[0].plot(self.mesh.vectorCCx, self.model[i_iteration])
                    axes[1].plot(self.jk, self.pred[i_iteration], "bx")
                    # axes[0].legend(("True", "Pred", ("%ith")%(i_iteration)))
                    # axes[1].legend(("Observed", "Predicted", ("%ith")%(i_iteration)))
                    axes[0].legend(("True", "Pred"))
                    axes[1].legend(("Observed", "Predicted"))

                    if i_iteration == 0:
                        i_iteration = 1
                    axes[2].plot(
                        self.save.phi_m[i_iteration - 1],
                        self.save.phi_d[i_iteration - 1] * 2,
                        "go",
                        ms=10,
                    )

                axes[2].plot(self.save.phi_m, self.save.phi_d * 2, "k-", lw=2)
                axes[2].set_xlim(
                    np.hstack(self.save.phi_m).min(), np.hstack(self.save.phi_m).max()
                )
                axes[2].set_xlabel("$\phi_m$", fontsize=14)
                axes[2].set_ylabel("$\phi_d$", fontsize=14)
                if self.save.i_target is not None:
                    axes[2].plot(
                        self.save.phi_m[self.save.i_target],
                        self.save.phi_d[self.save.i_target] * 2.0,
                        "k*",
                        ms=10,
                    )
                axes[2].set_title("Tikhonov curve")
        plt.tight_layout()

    def interact_plot_G(self):
        Q = interact(
            self.plot_G,
            N=IntSlider(min=1, max=100, step=1, value=20, continuous_update=False),
            M=IntSlider(min=1, max=100, step=1, value=100, continuous_update=False),
            p=FloatSlider(
                min=-1, max=0, step=0.05, value=-0.15, continuous_update=False
            ),
            q=FloatSlider(min=0, max=1, step=0.05, value=0.25, continuous_update=False),
            j1=FloatText(value=1.0),
            jn=FloatText(value=19.0),
            scale=ToggleButtons(options=["linear", "log"], value="log"),
            fixed=False,
            ymin=FloatText(value=-0.005),
            ymax=FloatText(value=0.011),
        )
        return Q

    def interact_plot_model(self):
        Q = interact(
            self.plot_model,
            m_background=FloatSlider(
                min=-2,
                max=2,
                step=0.05,
                value=0.0,
                continuous_update=False,
                description="m$_{background}$",
            ),
            m1=FloatSlider(
                min=-2,
                max=2,
                step=0.05,
                value=1.0,
                continuous_update=False,
                description="m1",
            ),
            m2=FloatSlider(
                min=-2,
                max=2,
                step=0.05,
                value=2.0,
                continuous_update=False,
                description="m2",
            ),
            m1_center=FloatSlider(
                min=-2,
                max=2,
                step=0.05,
                value=0.2,
                continuous_update=False,
                description="m1$_{center}$",
            ),
            dm1=FloatSlider(
                min=0,
                max=0.5,
                step=0.05,
                value=0.2,
                continuous_update=False,
                description="m1$_{width}$",
            ),
            m2_center=FloatSlider(
                min=-2,
                max=2,
                step=0.05,
                value=0.75,
                continuous_update=False,
                description="m2$_{center}$",
            ),
            sigma_2=FloatSlider(
                min=0.01,
                max=0.1,
                step=0.01,
                value=0.07,
                continuous_update=False,
                description="m2$_{sigma}$",
            ),
            option=SelectMultiple(
                options=["kernel", "model", "data"],
                value=["model"],
                description="option",
            ),
            percentage=FloatText(value=5),
            floor=FloatText(value=0.02),
        )
        return Q

    def interact_plot_inversion(self, maxIter=30):
        interact(
            self.plot_inversion,
            mode=RadioButtons(
                description="mode", options=["Run", "Explore"], value="Run"
            ),
            maxIter=IntText(value=maxIter),
            m0=FloatSlider(
                min=-2, max=2, step=0.05, value=0.0, continuous_update=False
            ),
            mref=FloatSlider(
                min=-2, max=2, step=0.05, value=0.0, continuous_update=False
            ),
            percentage=FloatText(value=self.percentage),
            floor=FloatText(value=self.floor),
            chifact=FloatText(value=1.0),
            beta0_ratio=FloatText(value=100),
            coolingFactor=FloatSlider(
                min=0.1, max=10, step=1, value=2, continuous_update=False
            ),
            coolingRate=IntSlider(
                min=1, max=10, step=1, value=1, continuous_update=False
            ),
            alpha_s=FloatText(value=1e-10),
            alpha_x=FloatText(value=0),
            target=False,
            option=ToggleButtons(options=["misfit", "tikhonov"], value="misfit"),
            i_iteration=IntSlider(
                min=0, max=maxIter, step=1, value=0, continuous_update=False
            ),
        )
