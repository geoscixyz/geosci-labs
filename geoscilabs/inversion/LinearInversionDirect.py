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
import properties

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
    VBox,
    HBox,
    Checkbox,
    interactive_output,
    Button,
    Layout
)
import warnings
warnings.filterwarnings("ignore")

class LinearInversionDirectApp(properties.HasProperties):
    """docstring for LinearInversionApp"""

    # Parameters for Model

    m = None

    m_background = properties.Float(
        "backgroound model", default=0., required=True
    )
    m1 = properties.Float(
        "m1", default=1., required=True
    )
    m1_center = properties.Float(
        "m1_center", default=0.2, required=True
    )
    dm1 = properties.Float(
        "dm1", default=0.2, required=True
    )
    m2 = properties.Float(
        "m2", default=2., required=True
    )
    m2_center = properties.Float(
        "m2_center", default=0.75, required=True
    )
    dm2 = properties.Float(
        "dm2", default=0.07, required=True
    )
    sigma_2 = properties.Float(
        "sigma_2", default=0.07, required=True
    )

    # Parameters for Sensitivity
    N = properties.Integer(
        "N", default=20, required=True
    )
    M = properties.Integer(
        "M", default=100, required=True
    )
    pmin = properties.Float(
        "pmin", default=-0.25, required=True
    )
    pmax = properties.Float(
        "pmax", default=-3, required=True
    )
    qmin = properties.Float(
        "qmin", default=0, required=True
    )
    qmax = properties.Float(
        "qmax", default=5, required=True
    )
    scale = properties.StringChoice(
        "scale",
        default="log",
        choices=["linear", "log"],
    )
    fixed = properties.Bool(
        "add_noise", default=False, required=True
    )
    ymin = properties.Float(
        "ymin", default=-0.005, required=True
    )
    ymax =properties.Float(
        "ymax", default=0.011, required=True
    )
    show_singular = properties.Bool(
        "show_singular", default=False, required=True
    )

    # Parameters for Data
    add_noise = properties.Bool(
        "add_noise", default=False, required=True
    )
    percentage = properties.Float(
        "percentage", default=0., required=True
    )
    floor = properties.Float(
        "floor", default=0.03, required=True
    )
    show_relative_noise = properties.Bool(
        "show_relative_noise", default=False, required=True
    )

    # Parameters for Inversion
    mode = properties.StringChoice(
        "mode",
        default="Run",
        choices=["Run", "Explore"],
        required=True
    )
    mref =properties.Float(
        "mref", default=0.0, required=True
    )
    chifact =properties.Float(
        "chifact", default=1.0, required=True
    )
    data_option = properties.StringChoice(
        "data_option",
        default="obs & pred",
        choices=["obs & pred", "normalized misfit"],
        required=True
    )
    beta_min =properties.Float(
        "beta_min", default=1e-4, required=True
    )
    beta_max =properties.Float(
        "beta_max", default=1e5, required=True
    )
    n_beta = properties.Integer(
        "n_beta", default=81, required=True
    )
    alpha_s = properties.Float(
        "alpha_s", default=1, min=0, required=True
    )
    alpha_x = properties.Float(
        "alpha_x", default=0., min=0, required=True
    )
    tikhonov =properties.StringChoice(
        "tikhonov",
        default="phi_d & phi_m",
        choices=["phi_d & phi_m", "phi_d vs phi_m"],
        required=True
    )
    i_beta = properties.Integer(
        "i_beta", default=0, required=True
    )
    scale_inv = properties.StringChoice(
        "scale_inv",
        default="log",
        choices=["linear", "log"],
    )

    noise_option = properties.StringChoice(
        "noise_option",
        default="error contaminated",
        choices=["error contaminated", "clean data"],
    )

    # Other parameters
    show_model = properties.Bool(
        "show_model", default=True, required=True
    )
    show_data = properties.Bool(
        "show_data", default=True, required=True
    )
    show_kernel = properties.Bool(
        "show_kernel", default=True, required=True
    )
    seed = None
    data_vec = None
    data_clen_vec = None
    save = None
    return_axis = False

    def __init__(self, **kwargs):
        super(LinearInversionDirectApp, self).__init__(**kwargs)

    @property
    def G(self):
        return self._G

    @property
    def p_values(self):
        return self._p_values

    @property
    def q_values(self):
        return self._q_values

    @property
    def mesh_prop(self):
        return self._mesh_prop

    def reset_to_defaults(self, **kwargs):
        for name in kwargs.keys():
            if name not in self._props:
                raise AttributeError(
                    "Input name '{}' is not a known " "property or attribute".format(name)
                )
        for key in self._props:
            if key in kwargs.keys():
                val = kwargs[key]
            else:
                val = self._props[key].default
            setattr(self, key, val)

    def set_G(self, N=20, M=100, pmin=-0.25, pmax=-15, qmin=0.25, qmax=15):
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
        p_values = np.linspace(pmin, pmax, N)
        q_values = np.linspace(qmin, qmax, N)
        self._G = np.zeros((N, self.mesh_prop.nC), dtype=float, order="C")

        def g(k):
            return np.exp(p_values[k] * self.mesh_prop.vectorCCx) * np.cos(
                2*np.pi * q_values[k] * self.mesh_prop.vectorCCx
            )

        for i in range(N):
            self._G[i, :] = g(i) * self.mesh_prop.hx
        self._p_values = p_values
        self._q_values = q_values

    def plot_G(
        self,
        N=None,
        M=None,
        pmin=-0.25,
        pmax=1,
        qmin=0.25,
        qmax=1,
        scale="log",
        fixed=False,
        ymin=-0.001,
        ymax=0.011,
        show_singular=False
    ):

        self.N = N
        self.M = M
        self.pmin = pmin
        self.pmax = pmax
        self.scale = scale
        self.fixed = fixed
        self.ymin = ymin
        self.ymax = ymax
        self.show_singular = show_singular

        self.set_G(N=N, M=M, pmin=pmin, pmax=pmax, qmin=qmin, qmax=qmax)

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

        if show_singular:
            ax2.plot(np.arange(self.N) + 1, s, "ro")
            ax2.set_xlabel("")
            ax2.set_title("singular values", fontsize=12)
            ax2.set_xscale(scale)
            ax2.set_yscale(scale)
            ax2.xaxis.set_major_locator(plt.NullLocator())
            ax2.xaxis.set_minor_locator(plt.NullLocator())
            ax2.xaxis.set_major_formatter(plt.NullFormatter())
            ax2.xaxis.set_minor_formatter(plt.NullFormatter())
        else:
            ax2.axis('off')
        plt.tight_layout()
        plt.show()
        if self.return_axis:
            return [ax1, ax2]

    def set_model(
        self,
        m_background=0.0,
        m1=1.0,
        m1_center=0.2,
        dm1=0.2,
        m2=2.0,
        m2_center=0.75,
        sigma_2=0.07
    ):

        m = np.zeros(self.mesh_prop.nC) + m_background
        m1_inds = np.logical_and(
            self.mesh_prop.vectorCCx > m1_center - dm1 / 2.0,
            self.mesh_prop.vectorCCx < m1_center + dm1 / 2.0,
        )
        m[m1_inds] = m1 + m_background

        def gaussian(x, x0, sigma):
            return np.exp(-np.power((x - x0) / sigma, 2.0) / 2.0)

        m += gaussian(self.mesh_prop.vectorCCx, m2_center, sigma_2) * m2
        self.m = m
        return m

    def plot_model_only(
        self,
        m_background=0.0,
        m1=1.0,
        m1_center=0.2,
        dm1=0.2,
        m2=2.0,
        m2_center=0.75,
        sigma_2=0.07,
        M=100
        ):

        self.M = M
        self.m_background = m_background
        self.m1 = m1
        self.m2 = m2
        self.m1_center = m1_center
        self.dm1 = dm1
        self.m2_center = m2_center
        self.sigma_2 = sigma_2

        self._mesh_prop = TensorMesh([self.M])
        m = self.set_model(
            m_background=m_background,
            m1=m1,
            m2=m2,
            m1_center=m1_center,
            dm1=dm1,
            m2_center=m2_center,
            sigma_2=sigma_2,
        )

        fig = plt.figure()
        ax  = plt.subplot(111)
        ax.plot(self.mesh_prop.vectorCCx, m)
        ax.set_ylim([-2.5, 2.5])
        ax.set_title("Model")
        ax.set_xlabel("x")
        ax.set_ylabel("m(x)")
        plt.show()
        if self.return_axis:
            return ax

    def plot_data_only(
        self,
        add_noise=False,
        percentage=0,
        floor=3e-2,
        show_relative_noise=False
    ):

        np.random.seed(1)
        self.add_noise = add_noise
        self.percentage = percentage
        self.floor = floor
        self.show_relative_noise = show_relative_noise

        self.set_G(N=self.N, M=self.M, pmin=self.pmin, pmax=self.pmax, qmin=self.qmin, qmax=self.qmax)

        m = self.set_model(
            m_background=self.m_background,
            m1=self.m1,
            m2=self.m2,
            m1_center=self.m1_center,
            dm1=self.dm1,
            m2_center=self.m2_center,
            sigma_2=self.sigma_2,
        )

        fig, axes = plt.subplots(1, 3, figsize=(13 * 1.2, 3 * 1.2))
        ax1, ax2, ax3 = axes

        if add_noise:
            # survey_obj, simulation_obj = self.get_problem_survey()
            simulation_obj = self.get_problem_survey()
            d_clean = simulation_obj.dpred(m)
            noise = (
                abs(d_clean) * percentage * 0.01 * np.random.randn(self.N)
                + np.random.randn(self.N) * floor
            )
        else:
            # survey_obj, simulation_obj = self.get_problem_survey()
            simulation_obj = self.get_problem_survey()
            d_clean = simulation_obj.dpred(m)
            noise = np.zeros(self.N, float)

        d = d_clean + noise
        self.data_vec = d.copy()
        self.data_clen_vec = d_clean.copy()
        self.m = m.copy()
        self.uncertainty = abs(self.data_vec) * percentage * 0.01 + floor
        self.percentage = percentage
        self.floor = floor

        if add_noise:
            # this is just for visualization of uncertainty
            ax3.errorbar(
                x=np.arange(self.N),
                y=d,
                yerr=self.uncertainty,
                color="k",
                lw=1,
                capsize=2
            )
            ax3.plot(np.arange(self.N), d, "ko")
            ax1.plot(np.arange(self.N), d_clean, "ko-")
            if self.show_relative_noise:
                eps =  np.percentile(abs(d_clean), 1)
                ax2.plot(np.arange(self.N), noise/(d + eps), "kx")
                ax2.set_title("Noise / data")
            else:
                ax2.plot(np.arange(self.N), noise, "kx")
                ax2.set_title("Noise")
            ylim = ax3.get_ylim()
            for ii, ax in enumerate([ax1, ax2, ax3]):
                if self.show_relative_noise:
                    if ii != 1:
                        ax.set_ylim(ylim)
                else:
                    ax.set_ylim(ylim)
                    if ii !=0:
                        ax.set_yticklabels([])

            ax3.set_title("Noisy data")

        else:
            ax1.plot(np.arange(self.N), d_clean, "ko-")
            ax2.axis('off')
            ax3.axis('off')
        ax1.set_ylabel("$d_j$")
        ax1.set_title("Clean data")
        for ax in axes:
            ax.set_xlabel("$j$")
        plt.show()
        if self.return_axis:
            return ax

    def plot_model(
        self,
        m_background=0.0,
        m1=1.0,
        m1_center=0.2,
        dm1=0.2,
        m2=2.0,
        m2_center=0.75,
        sigma_2=0.07,
        M=100,
        pmin=-0.25,
        pmax=1,
        qmin=0.25,
        qmax=1,
        N=20,
        add_noise=False,
        percentage=0,
        floor=3e-2,
        show_model=True,
        show_kernel=True,
        show_data=True,
    ):

        self.set_G(N=N, M=M, pmin=pmin, pmax=pmax, qmin=qmin, qmax=qmax)
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
            # survey_obj, simulation_obj = self.get_problem_survey()
            simulation_obj = self.get_problem_survey()
            d = simulation_obj.dpred(m)
            noise = (
                abs(d) * percentage * 0.01 * np.random.randn(self.N)
                + np.random.randn(self.N) * floor
            )
        else:
            # survey_obj, simulation_obj = self.get_problem_survey()
            simulation_obj = self.get_problem_survey()
            d = simulation_obj.dpred(m)
            noise = np.zeros(self.N, float)

        d += noise
        self.data_vec = d.copy()
        self.m = m.copy()
        self.uncertainty = abs(self.data_vec) * percentage * 0.01 + floor
        self.percentage = percentage
        self.floor = floor
        option_bools = [show_model, show_model, show_data]

        fig, axes = plt.subplots(1, 3, figsize=(12 * 1.2, 3 * 1.2))
        ax1, ax2, ax3 = axes

        if show_model:
            ax1.plot(self.mesh_prop.vectorCCx, m)
            ax1.set_ylim([-2.5, 2.5])
            ax1.set_title("Model")
            ax1.set_xlabel("x")
            ax1.set_ylabel("m(x)")

        if show_kernel:
            ax2.plot(self.mesh_prop.vectorCCx, self.G.T)
            ax2.set_title("Rows of matrix G")
            ax2.set_xlabel("x")
            ax2.set_ylabel("g(x)")

        if show_data:
            if add_noise:
                # this is just for visualization of uncertainty
                ax3.errorbar(
                    x=np.arange(self.N),
                    y=self.data_vec,
                    yerr=self.uncertainty,
                    color="k",
                    lw=1,
                    capsize=2
                )
                ax3.plot(np.arange(self.N), self.data_vec, "ko")
                ax3.set_title("Noisy data")
            else:
                ax3.plot(np.arange(self.N), self.data_vec, "ko-")
                ax3.set_title("Clean data")
            ax3.set_ylabel("$d_j$")
            ax3.set_xlabel("$k_j$")

        option_bools = [show_model, show_kernel, show_data]
        for i, ax in enumerate(axes):
            if not option_bools[i]:
                ax.axis("off")

        plt.tight_layout()
        plt.show()

        if self.return_axis:
            return axes

    def get_problem_survey(self):
        # survey_obj = survey.BaseSurvey()
        simulation_obj = simulation.LinearSimulation(
            # survey=survey_obj,
            mesh=self.mesh_prop,
            model_map=maps.IdentityMap(),
            G=self.G,
        )
        # return survey_obj, simulation_obj
        return simulation_obj

    def run_inversion_direct(
        self,
        m0=0.0,
        mref=0.0,
        percentage=0,
        floor=3e-2,
        chi_fact=1.0,
        beta_min=1e-4,
        beta_max=1e5,
        n_beta=81,
        alpha_s=1.0,
        alpha_x=0,
    ):

        self.uncertainty = percentage * abs(self.data_vec) * 0.01 + floor

        # survey_obj, simulation_obj = self.get_problem_survey()
        simulation_obj = self.get_problem_survey()
        survey_obj = survey.BaseSurvey([])
        survey_obj._vnD = np.r_[len(self.data_vec)]
        data_obj = data.Data(
            survey_obj,
            dobs=self.data_vec, noise_floor=self.uncertainty
        )
        dmis = data_misfit.L2DataMisfit(simulation=simulation_obj, data=data_obj)

        m0 = np.ones(self.M) * m0
        mref = np.ones(self.M) * mref
        reg = regularization.Tikhonov(
            self.mesh_prop, alpha_s=alpha_s, alpha_x=alpha_x, reference_model=mref
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
        mode="Run",
        noise_option="error contaminated",
        percentage=0,
        floor=0.03,
        chifact=1,
        mref=0.0,
        alpha_s=1.0,
        alpha_x=0.,
        beta_min=1e-4,
        beta_max=1e5,
        n_beta=81,
        i_beta=0,
        data_option="obs & pred",
        tikhonov="phi_d & phi_m",
        scale="log"
    ):
        self.mode = mode
        self.noise_option = noise_option
        self.percentage = percentage
        self.floor = floor
        self.chifact = chifact
        self.mref = mref
        self.alpha_s = alpha_s
        self.alpha_x = alpha_x
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_beta = n_beta
        self.i_beta = i_beta
        self.data_option = data_option
        self.tikhonov = tikhonov
        self.scale = scale

        m0 = 0.
        np.random.seed(1)
        if mode == "Run":
            if noise_option == "error contaminated":
                # survey_obj, simulation_obj = self.get_problem_survey()
                simulation_obj = self.get_problem_survey()
                d = simulation_obj.dpred(self.m)
                noise = (
                    abs(d) * percentage * 0.01 * np.random.randn(self.N)
                    + np.random.randn(self.N) * floor

                )
            elif noise_option == "clean data":
                # survey_obj, simulation_obj = self.get_problem_survey()
                simulation_obj = self.get_problem_survey()
                d = simulation_obj.dpred(self.m)
                noise = np.zeros(self.N, float)

            self.data_vec = d + noise

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
        self.i_beta = i_target

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
        if data_option == "obs & pred":
            if self.noise_option == "error contaminated":
                axes[1].errorbar(np.arange(self.N), self.data_vec, yerr=self.uncertainty, color="k", lw=1, capsize=2, label="Observed")
            else:
                axes[1].plot(np.arange(self.N), self.data_vec, "ko", label="Observed")
            if mode == "Run":
                axes[1].plot(np.arange(self.N), self.pred[i_target], "bx", label="Predicted")
            axes[1].legend()
            axes[1].set_title("Data")
            axes[1].set_xlabel("$k_j$")
            axes[1].set_ylabel("$d_j$")
        else:
            if mode == "Run":
                misfit = (self.pred[i_target] - self.data_vec) / self.uncertainty
            else:
                misfit = (self.pred[i_beta] - self.data_vec) / self.uncertainty

            axes[1].plot(np.arange(self.N), misfit, "ko")
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

        if tikhonov == "phi_d & phi_m":
            if mode == "Explore":
                axes[0].plot(self.mesh_prop.vectorCCx, self.model[i_beta])
                if data_option == "obs & pred":
                    axes[1].plot(np.arange(self.N), self.pred[i_beta], "bx", label="Predicted")
                    axes[1].legend()
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
        elif tikhonov == "phi_d vs phi_m":
            if mode == "Explore":
                axes[0].plot(self.mesh_prop.vectorCCx, self.model[i_beta])
                if data_option == "obs & pred":
                    axes[1].plot(np.arange(self.N), self.pred[i_beta], "bx")
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
        plt.show()
        if self.return_axis:
            return axes


    def interact_plot_model(self):
        Q = interact(
            self.plot_model_only,
            m_background=FloatSlider(
                min=-2,
                max=2,
                step=0.05,
                value=self.m_background,
                continuous_update=False,
                description="m$_{background}$",
            ),
            m1=FloatSlider(
                min=-2,
                max=2,
                step=0.05,
                value=self.m1,
                continuous_update=False,
                description="m1",
            ),
            m2=FloatSlider(
                min=-2,
                max=2,
                step=0.05,
                value=self.m2,
                continuous_update=False,
                description="m2",
            ),
            m1_center=FloatSlider(
                min=-2,
                max=2,
                step=0.05,
                value=self.m1_center,
                continuous_update=False,
                description="m1$_{center}$",
            ),
            dm1=FloatSlider(
                min=0,
                max=0.5,
                step=0.05,
                value=self.dm1,
                continuous_update=False,
                description="m1$_{width}$",
            ),
            m2_center=FloatSlider(
                min=-2,
                max=2,
                step=0.05,
                value=self.m2_center,
                continuous_update=False,
                description="m2$_{center}$",
            ),
            sigma_2=FloatSlider(
                min=0.01,
                max=0.1,
                step=0.01,
                value=self.sigma_2,
                continuous_update=False,
                description="m2$_{sigma}$",
            ),
            M=IntSlider(min=1, max=100, step=1, value=self.M, continuous_update=False),
        )
        return Q

    def interact_plot_G(self):
        Q = interact(
            self.plot_G,
            N=IntSlider(min=1, max=100, step=1, value=self.N, continuous_update=False),
            M=IntSlider(min=1, max=100, step=1, value=self.M, continuous_update=False),
            pmin=FloatText(self.pmin),
            pmax=FloatText(self.pmax),
            qmin=FloatText(self.qmin),
            qmax=FloatText(self.qmax),
            scale=ToggleButtons(options=["linear", "log"], value=self.scale),
            fixed=self.fixed,
            ymin=FloatText(value=self.ymin),
            ymax=FloatText(value=self.ymax),
            show_singular=Checkbox(value=self.show_singular)
        )
        return Q

    def interact_plot_data(self):
        Q = interact(
            self.plot_data_only,
            percentage=FloatText(value=self.percentage, description="percentage"),
            floor=FloatText(value=self.floor, description="floor"),
            add_noise=Checkbox(value=self.add_noise, description="add_noise"),
            show_relative_noise=Checkbox(value=self.show_relative_noise, description="show_relative_noise"),
        )
        return Q

    def interact_plot_all_three_together(self):

        m_background=FloatSlider(
            min=-2,
            max=2,
            step=0.05,
            value=self.m_background,
            continuous_update=False,
            description="m$_{background}$",
        )
        m1=FloatSlider(
            min=-2,
            max=2,
            step=0.05,
            value=self.m1,
            continuous_update=False,
            description="m1",
        )
        m2=FloatSlider(
            min=-2,
            max=2,
            step=0.05,
            value=self.m2,
            continuous_update=False,
            description="m2",
        )
        m1_center=FloatSlider(
            min=-2,
            max=2,
            step=0.05,
            value=self.m1_center,
            continuous_update=False,
            description="m1$_{center}$",
        )
        dm1=FloatSlider(
            min=0,
            max=0.5,
            step=0.05,
            value=self.dm1,
            continuous_update=False,
            description="m1$_{width}$",
        )
        m2_center=FloatSlider(
            min=-2,
            max=2,
            step=0.05,
            value=self.m2_center,
            continuous_update=False,
            description="m2$_{center}$",
        )
        sigma_2=FloatSlider(
            min=0.01,
            max=0.1,
            step=0.01,
            value=self.sigma_2,
            continuous_update=False,
            description="m2$_{sigma}$",
        )
        pmin=FloatText(self.pmin, description="pmin")
        pmax=FloatText(self.pmax, description="pmax")
        qmin=FloatText(self.qmin, description="qmin")
        qmax=FloatText(self.qmax, description="qmin")
        show_model = Checkbox(value=self.show_model, description="show model")
        show_kernel = Checkbox(value=self.show_kernel, description="show sensitivity")
        show_data = Checkbox(value=self.show_data, description="show data")
        percentage=FloatText(value=self.percentage, description="percentage")
        floor=FloatText(value=self.floor, description="floor")
        add_noise=Checkbox(value=self.add_noise, description="add_noise")
        M=IntSlider(min=1, max=100, step=1, value=self.M, continuous_update=False, description="M")
        N=IntSlider(min=1, max=100, step=1, value=self.N, continuous_update=False, description="N")

        out = interactive_output(
            self.plot_model,
            {
                "m_background": m_background,
                "m1": m1,
                "m1_center": m1_center,
                "dm1": dm1,
                "m2": m2,
                "m2_center": m2_center,
                "sigma_2": sigma_2,
                "show_model": show_model,
                "show_data": show_data,
                "show_kernel": show_kernel,
                "add_noise": add_noise,
                "percentage": percentage,
                "floor": floor,
                "pmin": pmin,
                "pmax": pmax,
                "qmin": qmin,
                "qmax": qmax,
                "M":M,
                "N":N,
            },
        )
        a = Button(description='Model',
                   layout=Layout(width='100%', height='30px'))
        b = Button(description='Sensitivity',
                   layout=Layout(width='100%', height='30px'))
        c = Button(description='Noise',
                   layout=Layout(width='100%', height='30px'))

        return VBox(
                [
                    HBox(
                            [
                                VBox([a, M, m_background, m1, m1_center, dm1, m2, m2_center, sigma_2, show_model]),
                                VBox([b, N, pmin, pmax, qmin, qmax, show_kernel]),
                                VBox([c, add_noise, percentage, floor, show_data])
                            ]
                        ),
                    HBox([out])
                ]
            )

    def interact_plot_inversion(self):

        mode=RadioButtons(
            description="mode", options=["Run", "Explore"], value=self.mode
        )
        mref=FloatSlider(
            min=-2, max=2, step=0.05, value=self.mref, continuous_update=False, description="mref"
        )
        percentage=FloatText(value=self.percentage, description="percentage")
        floor=FloatText(value=self.floor, description="floor")
        beta_min=FloatText(value=self.beta_min, description="beta_min")
        beta_max=FloatText(value=self.beta_max, description="beta_min")
        n_beta=IntText(value=self.n_beta, min=10, max=100, description="n_beta")
        alpha_s=FloatText(value=self.alpha_s, description="alpha_s")
        alpha_x=FloatText(value=self.alpha_x, description="alpha_x")
        tikhonov=RadioButtons(
            options=["phi_d & phi_m", "phi_d vs phi_m"], value=self.tikhonov,
            description='tikhonov'
        )
        data_option=RadioButtons(
            options=["obs & pred", "normalized misfit"], value="obs & pred", description="data"
        )
        scale=RadioButtons(options=["linear", "log"], value=self.scale_inv, description='scale')
        i_beta=IntSlider(
            min=0, max=self.n_beta - 1, step=1, value=self.i_beta, continuous_update=False,
            description='i_beta'
        )
        chifact=FloatText(value=self.chifact, description='chifact')
        noise_option=RadioButtons(
            options=["error contaminated", "clean data"], value=self.noise_option, description = 'noise option'
        )
        out = interactive_output(
            self.plot_inversion,
            {
                "mode": mode,
                "mref": mref,
                "percentage": percentage,
                "floor": floor,
                "beta_min": beta_min,
                "beta_max": beta_max,
                "n_beta": n_beta,
                "alpha_s": alpha_s,
                "alpha_x": alpha_x,
                "tikhonov": tikhonov,
                "i_beta": i_beta,
                "chifact": chifact,
                "floor": floor,
                "data_option":data_option,
                "tikhonov":tikhonov,
                "scale":scale,
                "noise_option":noise_option
            },
        )
        a = Button(description='Misfit',
                   layout=Layout(width='100%', height='30px'))
        b = Button(description='Model norm',
                   layout=Layout(width='100%', height='30px'))
        c = Button(description='Beta',
                   layout=Layout(width='100%', height='30px'))
        d = Button(description='Plotting options',
                   layout=Layout(width='92%', height='30px'))

        return VBox(
                [
                    HBox([mode, noise_option]),
                    HBox(
                            [
                                VBox([a, percentage, floor, chifact]),
                                VBox([b, mref, alpha_s, alpha_x]),
                                VBox([c, beta_min, beta_max, n_beta, i_beta])
                            ]
                        ),
                    VBox(
                            [
                                d,
                                HBox([data_option, tikhonov, scale]),
                            ]
                        ),
                    HBox([out])
                ]
            )
