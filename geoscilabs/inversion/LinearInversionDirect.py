import numpy as np
from discretize import TensorMesh
from SimPEG import (
    maps,
    simulation,
    survey,
    data,
    data_misfit,
    directives,
    optimization,
    regularization,
    inverse_problem,
    inversion,
)
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


class LinearInversionDirectApp(object):
    """docstring for LinearInversionApp"""

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

    data_vec = None
    save = None

    def __init__(self):
        super(LinearInversionDirectApp, self).__init__()

    @property
    def G(self):
        return self._G

    @property
    def jk(self):
        return self._jk

    @property
    def mesh_prop(self):
        return self._mesh_prop

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
        self._mesh_prop = TensorMesh([M])
        jk = np.linspace(j1, jn, N)
        self._G = np.zeros((N, self.mesh_prop.nC), dtype=float, order="C")

        def g(k):
            return np.exp(p * jk[k] * self.mesh_prop.vectorCCx) * np.cos(
                np.pi * q * jk[k] * self.mesh_prop.vectorCCx
            )

        for i in range(N):
            self._G[i, :] = g(i) * self.mesh_prop.hx
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

        ax1.plot(self.mesh_prop.vectorCCx, self.G.T)
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
        m = np.zeros(self.mesh_prop.nC) + m_background
        m1_inds = np.logical_and(
            self.mesh_prop.vectorCCx > m1_center - dm1 / 2.0,
            self.mesh_prop.vectorCCx < m1_center + dm1 / 2.0,
        )
        m[m1_inds] = m1

        def gaussian(x, x0, sigma):
            return np.exp(-np.power((x - x0) / sigma, 2.0) / 2.0)

        m += gaussian(self.mesh_prop.vectorCCx, m2_center, sigma_2) * m2
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
            survey_obj, simulation_obj = self.get_problem_survey()
            d = simulation_obj.dpred(m)
            noise = (
                abs(d) * percentage * 0.01 * np.random.randn(self.N)
                + np.random.randn(self.N) * floor
            )
        else:
            survey_obj, simulation_obj = self.get_problem_survey()
            d = simulation_obj.dpred(m)
            noise = np.zeros(self.N, float)

        d += noise
        self.data_vec = d.copy()
        self.m = m.copy()
        self.uncertainty = abs(self.data_vec) * percentage * 0.01 + floor
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
                    ax.plot(self.mesh_prop.vectorCCx, self.G.T)
                    ax.set_title("Rows of matrix G")
                    ax.set_xlabel("x")
                    ax.set_ylabel("g(x)")
                elif i == 1:
                    ax.plot(self.mesh_prop.vectorCCx, m)
                    ax.set_ylim([-2.5, 2.5])
                    ax.set_title("Model")
                    ax.set_xlabel("x")
                    ax.set_ylabel("m(x)")
                elif i == 2:
                    if add_noise:
                        # this is just for visualization of uncertainty
                        ax.errorbar(
                            x=self.jk,
                            y=self.data_vec,
                            yerr=self.uncertainty,
                            color="k",
                            lw=1,
                        )
                        ax.plot(self.jk, self.data, "ko")
                    else:
                        ax.plot(self.jk, self.data, "ko-")
                    ax.set_ylabel("$d_j$")
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

    def get_problem_survey(self):
        survey_obj = survey.LinearSurvey()
        simulation_obj = simulation.LinearSimulation(
            survey=survey_obj,
            mesh=self.mesh_prop,
            model_map=maps.IdentityMap(),
            G=self.G,
        )
        return survey_obj, simulation_obj

    def run_inversion_direct(
        self,
        m0=0.0,
        mref=0.0,
        percentage=5,
        floor=0.1,
        chi_fact=1.0,
        beta_min=1e-4,
        beta_max=1e0,
        n_beta=31,
        alpha_s=1.0,
        alpha_x=1.0,
    ):

        self.uncertainty = percentage * abs(self.data_vec) * 0.01 + floor

        survey_obj, simulation_obj = self.get_problem_survey()
        data_obj = data.Data(
            survey_obj, dobs=self.data_vec, noise_floor=self.uncertainty
        )
        dmis = data_misfit.L2DataMisfit(simulation=simulation_obj, data=data_obj)

        m0 = np.ones(self.M) * m0
        mref = np.ones(self.M) * mref
        reg = regularization.Tikhonov(
            self.mesh_prop, alpha_s=alpha_s, alpha_x=alpha_x, mref=mref
        )

        betas = np.logspace(np.log10(beta_min), np.log10(beta_max), n_beta)[::-1]

        phi_d = np.zeros(n_beta, dtype=float)
        phi_m = np.zeros(n_beta, dtype=float)
        models = []
        preds = []

        G = dmis.W.dot(self.G)

        for ii, beta in enumerate(betas):
            A = G.T.dot(G) + beta * reg.deriv2(m0)
            b = -(dmis.deriv(m0) + beta * reg.deriv(m0))
            m = np.linalg.solve(A, b)
            phi_d[ii] = dmis(m) * 2.0
            phi_m[ii] = reg(m) * 2.0
            models.append(m)
            preds.append(simulation_obj.dpred(m))

        return phi_d, phi_m, models, preds, betas

    def plot_inversion(
        self,
        mode=True,
        mref=0.0,
        percentage=5,
        floor=0.1,
        chifact=1,
        data_option="obs/pred",
        beta_min=1e-4,
        beta_max=1e0,
        n_beta=31,
        alpha_s=1.0,
        alpha_x=1.0,
        option="model",
        i_beta=1,
        scale="log",
    ):
        m0 = 0.0
        if mode == "Run":
            (
                self.phi_d,
                self.phi_m,
                self.model,
                self.pred,
                self.betas,
            ) = self.run_inversion_direct(
                m0=m0,
                mref=mref,
                percentage=percentage,
                floor=floor,
                beta_min=beta_min,
                beta_max=beta_max,
                n_beta=n_beta,
                alpha_s=alpha_s,
                alpha_x=alpha_x,
            )
        nD = self.data_vec.size
        i_target = np.argmin(abs(self.phi_d - nD * chifact))

        if i_beta > n_beta - 1:
            print(
                (">> Warning: input i_beta (%i) is greater than n_beta (%i)")
                % (i_beta, n_beta - 1)
            )
            i_beta = n_beta - 1

        fig, axes = plt.subplots(1, 3, figsize=(14 * 1.2, 3 * 1.2))
        axes[0].plot(self.mesh_prop.vectorCCx, self.m)
        if mode == "Run":
            axes[0].plot(self.mesh_prop.vectorCCx, self.model[i_target])
        axes[0].set_ylim([-2.5, 2.5])
        if data_option == "obs/pred":
            axes[1].plot(self.jk, self.data_vec, "ko")
            if mode == "Run":
                axes[1].plot(self.jk, self.pred[i_target], "bx")
            axes[1].legend(("Observed", "Predicted"))
            axes[1].set_title("Data")
            axes[1].set_xlabel("$k_j$")
            axes[1].set_ylabel("$d_j$")
        else:
            if mode == "Run":
                misfit = (self.pred[i_target] - self.data_vec) / self.uncertainty
            else:
                misfit = (self.pred[i_beta] - self.data_vec) / self.uncertainty

            axes[1].plot(self.jk, misfit, "ko")
            axes[1].set_title("Normalized misfit")
            axes[1].set_xlabel("$k_j$")
            axes[1].set_ylabel("$\epsilon_j$")
            axes[1].set_ylim(-3, 3)
            axes[1].set_yticks([-2, -1, 0, 1, 2])
            xlim = axes[1].get_xlim()
            axes[1].plot(xlim, [-1, -1], "k--", lw=1, alpha=0.5)
            axes[1].plot(xlim, [1, 1], "k--", lw=1, alpha=0.5)
            axes[1].set_xlim(xlim)
        axes[0].legend(("True", "Pred"))
        axes[0].set_title("Model")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("m(x)")

        if option == "misfit":
            if mode == "Explore":
                axes[0].plot(self.mesh_prop.vectorCCx, self.model[i_beta])
                if data_option == "obs/pred":
                    axes[1].plot(self.jk, self.pred[i_beta], "bx")
                    axes[1].legend(("Observed", "Predicted"))
                axes[2].plot(self.betas[i_beta], self.phi_d[i_beta], "go", ms=10)

            ax_1 = axes[2].twinx()
            axes[2].loglog(self.betas, self.phi_d, "k-", lw=2)
            axes[2].plot(self.betas[i_target], self.phi_d[i_target], "k*", ms=10)
            ax_1.plot(self.betas, self.phi_m, "r", lw=2)
            axes[2].set_xlabel("Beta")
            axes[2].set_ylabel("$\phi_d$")
            ax_1.set_ylabel("$\phi_m$", color="r")
            for tl in ax_1.get_yticklabels():
                tl.set_color("r")

            xmin, xmax = beta_max, beta_min
        elif option == "tikhonov":
            if mode == "Explore":
                axes[0].plot(self.mesh_prop.vectorCCx, self.model[i_beta])
                if data_option == "obs/pred":
                    axes[1].plot(self.jk, self.pred[i_beta], "bx")
                    axes[1].legend(("Observed", "Predicted"))
                axes[0].legend(("True", "Pred"))
                axes[2].plot(self.phi_m[i_beta], self.phi_d[i_beta], "go", ms=10)

            axes[2].plot(self.phi_m, self.phi_d, "k-", lw=2)
            axes[2].plot(self.phi_m[i_target], self.phi_d[i_target], "k*", ms=10)
            xmin, xmax = np.hstack(self.phi_m).min(), np.hstack(self.phi_m).max()

            axes[2].set_xlabel("$\phi_m$", fontsize=14)
            axes[2].set_ylabel("$\phi_d$", fontsize=14)
            # axes[2].set_title("Tikhonov curve")
        if scale == "log":
            axes[2].set_yscale("log")
            axes[2].set_xscale("log")
        else:
            axes[2].set_yscale("linear")
            axes[2].set_xscale("linear")
        axes[2].plot((xmin, xmax), (nD * chifact, nD * chifact), "k--")
        axes[2].set_xlim(xmin, xmax)
        if mode == "Run":
            title = ("$\phi_d^{\\ast}$=%.1e, $\phi_m$=%.1e, $\\beta$=%.1e") % (
                self.phi_d[i_target],
                self.phi_m[i_target],
                self.betas[i_target],
            )
        elif mode == "Explore":
            title = ("$\phi_d$=%.1e, $\phi_m$=%.1e, $\\beta$=%.1e") % (
                self.phi_d[i_beta],
                self.phi_m[i_beta],
                self.betas[i_beta],
            )
        axes[2].set_title(title, fontsize=14)
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

    def interact_plot_inversion(self, n_beta=81):
        interact(
            self.plot_inversion,
            mode=RadioButtons(
                description="mode", options=["Run", "Explore"], value="Run"
            ),
            mref=FloatSlider(
                min=-2, max=2, step=0.05, value=0.0, continuous_update=False
            ),
            percentage=FloatText(value=self.percentage),
            floor=FloatText(value=self.floor),
            beta_min=FloatText(value=1e-3),
            beta_max=FloatText(value=1e5),
            n_beta=IntText(value=n_beta, min=10, max=100),
            alpha_s=FloatText(value=1.0),
            alpha_x=FloatText(value=0),
            option=ToggleButtons(options=["misfit", "tikhonov"], value="tikhonov"),
            data_option=ToggleButtons(
                options=["obs/pred", "misfit"], value="obs/pred", description="data"
            ),
            scale=ToggleButtons(options=["linear", "log"], value="log"),
            i_beta=IntSlider(
                min=0, max=n_beta - 1, step=1, value=0, continuous_update=False
            ),
            chifact=FloatText(value=1.0),
        )
