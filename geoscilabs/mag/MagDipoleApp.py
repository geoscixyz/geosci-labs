import numpy as np
from scipy.interpolate import interp1d
from discretize import TensorMesh
from SimPEG import utils
from geoana import em
from ipywidgets import widgets
from ipywidgets import Layout
import matplotlib.pyplot as plt
from .Simulator import definePrism, plotObj3D
from .Mag import Simulation, createMagSurvey


class MagneticDipoleApp(object):
    """docstring for MagneticDipoleApp"""

    mesh = None
    component = None
    z = None
    inclination = None
    declination = None
    length = None
    dx = None
    moment = None
    depth = None
    data = None
    profile = None
    xy_profile = None
    clim = None

    def __init__(self):
        super(MagneticDipoleApp, self).__init__()

    def id_to_cartesian(self, inclination, declination):
        ux = np.cos(inclination / 180.0 * np.pi) * np.sin(declination / 180.0 * np.pi)
        uy = np.cos(inclination / 180.0 * np.pi) * np.cos(declination / 180.0 * np.pi)
        uz = -np.sin(inclination / 180.0 * np.pi)
        return np.r_[ux, uy, uz]

    def dot_product(self, b_vec, orientation):
        b_tmi = np.dot(b_vec, orientation)
        return b_tmi

    def simulate_dipole(
        self,
        component,
        target,
        inclination,
        declination,
        length,
        dx,
        moment,
        depth,
        profile,
        fixed_scale,
        show_halfwidth,
    ):
        self.component = component
        self.target = target
        self.inclination = inclination
        self.declination = declination
        self.length = length
        self.dx = dx
        self.moment = moment
        self.depth = depth
        self.profile = profile
        self.fixed_scale = fixed_scale
        self.show_halfwidth = show_halfwidth

        nT = 1e9
        nx = ny = int(length / dx)
        hx = np.ones(nx) * dx
        hy = np.ones(ny) * dx
        self.mesh = TensorMesh((hx, hy), "CC")
        z = np.r_[1.0]
        orientation = self.id_to_cartesian(inclination, declination)
        if self.target == "Dipole":
            md = em.static.MagneticDipoleWholeSpace(
                location=np.r_[0, 0, -depth], orientation=orientation, moment=moment
            )
            xyz = utils.ndgrid(self.mesh.vectorCCx, self.mesh.vectorCCy, z)
            b_vec = md.magnetic_flux_density(xyz)

        elif self.target == "Monopole (+)":
            md = em.static.MagneticPoleWholeSpace(
                location=np.r_[0, 0, -depth], orientation=orientation, moment=moment
            )
            xyz = utils.ndgrid(self.mesh.vectorCCx, self.mesh.vectorCCy, z)
            b_vec = md.magnetic_flux_density(xyz)

        elif self.target == "Monopole (-)":
            md = em.static.MagneticPoleWholeSpace(
                location=np.r_[0, 0, -depth], orientation=orientation, moment=moment
            )
            xyz = utils.ndgrid(self.mesh.vectorCCx, self.mesh.vectorCCy, z)
            b_vec = -md.magnetic_flux_density(xyz)

        # Project to the direction  of earth field
        if component == "Bt":
            rx_orientation = orientation.copy()
        elif component == "Bg":
            rx_orientation = orientation.copy()
            xyz_up = utils.ndgrid(self.mesh.vectorCCx, self.mesh.vectorCCy, z + 1.0)
            b_vec -= md.magnetic_flux_density(xyz_up)

        elif component == "Bx":
            rx_orientation = self.id_to_cartesian(0, 0)
        elif component == "By":
            rx_orientation = self.id_to_cartesian(0, 90)
        elif component == "Bz":
            rx_orientation = self.id_to_cartesian(90, 0)

        self.data = self.dot_product(b_vec, rx_orientation) * nT

        # Compute profile
        if (profile == "North") or (profile == "None"):
            self.xy_profile = np.c_[np.zeros(self.mesh.nCx), self.mesh.vectorCCx]

        elif profile == "East":
            self.xy_profile = np.c_[self.mesh.vectorCCx, np.zeros(self.mesh.nCx)]
        self.inds_profile = utils.closestPoints(self.mesh, self.xy_profile)
        self.data_profile = self.data[self.inds_profile]

    def simulate_two_monopole(
        self,
        component,
        inclination,
        declination,
        length,
        dx,
        moment,
        depth_n,
        depth_p,
        profile,
        fixed_scale,
        show_halfwidth,
    ):
        self.component = component
        self.inclination = inclination
        self.declination = declination
        self.length = length
        self.dx = dx
        self.moment = moment
        self.depth = depth_n
        self.depth = depth_p
        self.profile = profile
        self.fixed_scale = fixed_scale
        self.show_halfwidth = show_halfwidth

        nT = 1e9
        nx = ny = int(length / dx)
        hx = np.ones(nx) * dx
        hy = np.ones(ny) * dx
        self.mesh = TensorMesh((hx, hy), "CC")
        z = np.r_[1.0]
        orientation = self.id_to_cartesian(inclination, declination)

        md_n = em.static.MagneticPoleWholeSpace(
            location=np.r_[0, 0, -depth_n], orientation=orientation, moment=moment
        )
        md_p = em.static.MagneticPoleWholeSpace(
            location=np.r_[0, 0, -depth_p], orientation=orientation, moment=moment
        )
        xyz = utils.ndgrid(self.mesh.vectorCCx, self.mesh.vectorCCy, z)
        b_vec = -md_n.magnetic_flux_density(xyz) + md_p.magnetic_flux_density(xyz)

        # Project to the direction  of earth field
        if component == "Bt":
            rx_orientation = orientation.copy()
        elif component == "Bx":
            rx_orientation = self.id_to_cartesian(0, 0)
        elif component == "By":
            rx_orientation = self.id_to_cartesian(0, 90)
        elif component == "Bz":
            rx_orientation = self.id_to_cartesian(90, 0)
        elif component == "Bg":
            rx_orientation = orientation.copy()
            xyz_up = utils.ndgrid(self.mesh.vectorCCx, self.mesh.vectorCCy, z + 1.0)
            b_vec -= -md_n.magnetic_flux_density(xyz_up)
            b_vec -= md_p.magnetic_flux_density(xyz_up)

        self.data = self.dot_product(b_vec, rx_orientation) * nT

        # Compute profile
        if (profile == "North") or (profile == "None"):
            self.xy_profile = np.c_[np.zeros(self.mesh.nCx), self.mesh.vectorCCx]

        elif profile == "East":
            self.xy_profile = np.c_[self.mesh.vectorCCx, np.zeros(self.mesh.nCx)]

        self.inds_profile = utils.closestPoints(self.mesh, self.xy_profile)
        self.data_profile = self.data[self.inds_profile]

    def get_prism(self, dx, dy, dz, x0, y0, elev, prism_inc, prism_dec):
        prism = definePrism()
        prism.dx, prism.dy, prism.dz, prism.z0 = dy, dx, dz, elev
        prism.x0, prism.y0 = x0, y0
        prism.pinc, prism.pdec = prism_inc, prism_dec
        return prism

    def plot_prism(self, prism, dip=30, azim=310):
        return plotObj3D(prism, self.survey, dip, azim, self.mesh.vectorCCx.max())

    def simulate_prism(
        self,
        component,
        inclination,
        declination,
        length,
        dx,
        B0,
        kappa,
        depth,
        profile,
        fixed_scale,
        show_halfwidth,
        prism_dx,
        prism_dy,
        prism_dz,
        prism_inclination,
        prism_declination,
    ):
        self.component = component
        self.inclination = -inclination  # -ve accounts for LH modeling in SimPEG
        self.declination = declination
        self.length = length
        self.dx = dx
        self.B0 = B0
        self.kappa = kappa
        self.depth = depth
        self.profile = profile
        self.fixed_scale = fixed_scale
        self.show_halfwidth = show_halfwidth

        # prism parameter
        self.prism = self.get_prism(
            prism_dx,
            prism_dy,
            prism_dz,
            0,
            0,
            -depth,
            prism_inclination,
            prism_declination,
        )

        nx = ny = int(length / dx)
        hx = np.ones(nx) * dx
        hy = np.ones(ny) * dx
        self.mesh = TensorMesh((hx, hy), "CC")

        z = np.r_[1.0]
        B = np.r_[
            B0, -inclination, declination
        ]  # -ve accounts for LH modeling in SimPEG

        # Project to the direction  of earth field
        if component == "Bt":
            uType = "tf"
        elif component == "Bx":
            uType = "bx"
        elif component == "By":
            uType = "by"
        elif component == "Bz":
            uType = "bz"

        xyz = utils.ndgrid(self.mesh.vectorCCx, self.mesh.vectorCCy, z)
        out = createMagSurvey(np.c_[xyz, np.ones(self.mesh.nC)], B)
        self.survey = out[0]
        self.dobs = out[1]
        self.sim = Simulation()

        self.sim.prism = self.prism
        self.sim.survey = self.survey
        self.sim.susc = kappa
        self.sim.uType, self.sim.mType = uType, "induced"

        self.data = self.sim.fields()[0]

        # Compute profile
        if (profile == "North") or (profile == "None"):
            self.xy_profile = np.c_[np.zeros(self.mesh.nCx), self.mesh.vectorCCx]

        elif profile == "East":
            self.xy_profile = np.c_[self.mesh.vectorCCx, np.zeros(self.mesh.nCx)]
        self.inds_profile = utils.closestPoints(self.mesh, self.xy_profile)
        self.data_profile = self.data[self.inds_profile]

    def plot_map(self):
        length = self.length
        data = self.data
        profile = self.profile

        fig = plt.figure(figsize=(5, 6))
        ax1 = plt.subplot2grid((8, 4), (0, 0), rowspan=5, colspan=3)
        ax2 = plt.subplot2grid((8, 4), (6, 0), rowspan=2, colspan=3)

        if (self.clim is None) or (not self.fixed_scale):
            self.clim = data.min(), data.max()

        out = self.mesh.plotImage(
            self.data, pcolorOpts={"cmap": "jet"}, ax=ax1, clim=self.clim
        )
        cax = fig.add_axes([0.75, 0.45, 0.02, 0.5])
        ratio = abs(self.clim[0] / self.clim[1])
        if ratio < 0.05:
            ticks = [self.clim[0], self.clim[1]]
        else:
            ticks = [self.clim[0], 0, self.clim[1]]

        cb = plt.colorbar(out[0], ticks=ticks, format="%.3f", cax=cax)
        cb.set_label("nT", labelpad=-40, y=-0.05, rotation=0)
        ax1.set_aspect(1)
        ax1.set_ylabel("Northing")
        ax1.set_xlabel("Easting")
        if profile == "North":
            # xy_profile = np.c_[np.zeros(self.mesh.nCx), self.mesh.vectorCCx]
            ax1.text(1, length / 2 - length / 2 * 0.1, "B", color="w")
            ax1.text(1, -length / 2, "A", color="w")
        elif profile == "East":
            # xy_profile = np.c_[self.mesh.vectorCCx, np.zeros(self.mesh.nCx)]
            ax1.text(length / 2 - length / 2 * 0.1, 1, "B", color="w")
            ax1.text(-length / 2, 1, "A", color="w")

        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2.yaxis.tick_right()
        ax2.plot(self.mesh.vectorCCx, self.data_profile, "k", lw=2)
        ax2.plot(
            self.mesh.vectorCCx, np.zeros(self.mesh.nCx), "k--", color="grey", lw=1
        )

        ax2.set_xlim(-self.length / 2.0, self.length / 2.0)
        ymin, ymax = ax2.get_ylim()
        ax2.text(-self.length / 2.0, ymax, "A")
        ax2.text(self.length / 2.0 - self.length / 2 * 0.05, ymax, "B")
        ax2.set_yticks([self.clim[0], self.clim[1]])
        if self.show_halfwidth:
            x_half, data_half = self.get_half_width()
            ax2.plot(x_half, data_half, "bo--")
            ax2.set_xlabel(("Halfwidth: %.1fm") % (abs(np.diff(x_half))))
        else:
            ax2.set_xlabel(" ")
        # plt.tight_layout()
        if profile == "None":
            ax2.remove()
        else:
            ax1.plot(self.xy_profile[:, 0], self.xy_profile[:, 1], "w")

    def get_half_width(self, n_points=200):
        ind_max = np.argmax(abs(self.data_profile))
        A_half = self.data_profile[ind_max] / 2.0
        if self.profile == "North":
            x = self.xy_profile[:, 1]
        else:
            x = self.xy_profile[:, 0]
        f = interp1d(x, self.data_profile)
        x_int = np.linspace(x.min(), x.max(), n_points)
        data_profile_int = f(x_int)
        inds = np.argsort(abs(data_profile_int - A_half))
        ind_first = inds[0]
        for ind in inds:
            dx = abs(x_int[ind_first] - x_int[ind])
            if dx > self.dx:
                ind_second = ind
                break
        inds = [ind_first, ind_second]
        return x_int[inds], data_profile_int[inds]

    def magnetic_dipole_applet(
        self,
        component,
        target,
        inclination,
        declination,
        length,
        dx,
        moment,
        depth,
        profile,
        fixed_scale,
        show_halfwidth,
    ):
        self.simulate_dipole(
            component,
            target,
            inclination,
            declination,
            length,
            dx,
            moment,
            depth,
            profile,
            fixed_scale,
            show_halfwidth,
        )
        self.plot_map()

    def magnetic_two_monopole_applet(
        self,
        component,
        inclination,
        declination,
        length,
        dx,
        moment,
        depth_n,
        depth_p,
        profile,
        fixed_scale,
        show_halfwidth,
    ):
        self.simulate_two_monopole(
            component,
            inclination,
            declination,
            length,
            dx,
            moment,
            depth_n,
            depth_p,
            profile,
            fixed_scale,
            show_halfwidth,
        )
        self.plot_map()

    def magnetic_prism_applet(
        self,
        plot,
        component,
        inclination,
        declination,
        length,
        dx,
        B0,
        kappa,
        depth,
        profile,
        fixed_scale,
        show_halfwidth,
        prism_dx,
        prism_dy,
        prism_dz,
        prism_inclination,
        prism_declination,
    ):
        if plot == "field":
            self.simulate_prism(
                component,
                inclination,
                declination,
                length,
                dx,
                B0,
                kappa,
                depth,
                profile,
                fixed_scale,
                show_halfwidth,
                prism_dx,
                prism_dy,
                prism_dz,
                prism_inclination,
                prism_declination,
            )
            self.plot_map()
        elif plot == "model":
            self.prism = self.get_prism(
                prism_dx,
                prism_dy,
                prism_dz,
                0,
                0,
                -depth,
                prism_inclination,
                prism_declination,
            )
            self.plot_prism(self.prism)

    def interact_plot_model_dipole(self):
        component = widgets.RadioButtons(
            options=["Bt", "Bx", "By", "Bz", "Bg"],
            value="Bt",
            description="field",
            disabled=False,
        )
        target = widgets.RadioButtons(
            options=["Dipole", "Monopole (+)", "Monopole (-)"],
            value="Dipole",
            description="target",
            disabled=False,
        )

        inclination = widgets.FloatSlider(
            description="I", continuous_update=False, min=-90, max=90, step=1, value=90
        )
        declination = widgets.FloatSlider(
            description="D", continuous_update=False, min=-180, max=180, step=1, value=0
        )
        length = widgets.FloatSlider(
            description="length",
            continuous_update=False,
            min=2,
            max=200,
            step=1,
            value=72,
        )
        dx = widgets.FloatSlider(
            description="data spacing",
            continuous_update=False,
            min=0.1,
            max=15,
            step=0.1,
            value=2,
        )
        moment = widgets.FloatText(description="M", value=30)
        depth = widgets.FloatSlider(
            description="depth",
            continuous_update=False,
            min=0,
            max=50,
            step=1,
            value=10,
        )

        profile = widgets.RadioButtons(
            options=["East", "North", "None"],
            value="East",
            description="profile",
            disabled=False,
        )
        fixed_scale = widgets.Checkbox(
            value=False, description="fixed scale", disabled=False
        )

        show_halfwidth = widgets.Checkbox(
            value=False, description="half width", disabled=False
        )

        out = widgets.interactive_output(
            self.magnetic_dipole_applet,
            {
                "component": component,
                "target": target,
                "inclination": inclination,
                "declination": declination,
                "length": length,
                "dx": dx,
                "moment": moment,
                "depth": depth,
                "profile": profile,
                "fixed_scale": fixed_scale,
                "show_halfwidth": show_halfwidth,
            },
        )
        left = widgets.VBox(
            [component, profile],
            layout=Layout(width="20%", height="400px", margin="60px 0px 0px 0px"),
        )
        right = widgets.VBox(
            [
                target,
                inclination,
                declination,
                length,
                dx,
                moment,
                depth,
                fixed_scale,
                show_halfwidth,
            ],
            layout=Layout(width="50%", height="400px", margin="20px 0px 0px 0px"),
        )
        widgets.VBox(
            [out], layout=Layout(width="70%", height="400px", margin="0px 0px 0px 0px")
        )
        return widgets.HBox([left, out, right])

    def interact_plot_model_two_monopole(self):
        component = widgets.RadioButtons(
            options=["Bt", "Bx", "By", "Bz", "Bg"],
            value="Bt",
            description="field",
            disabled=False,
        )

        inclination = widgets.FloatSlider(
            description="I", continuous_update=False, min=-90, max=90, step=1, value=90
        )
        declination = widgets.FloatSlider(
            description="D", continuous_update=False, min=-180, max=180, step=1, value=0
        )
        length = widgets.FloatSlider(
            description="length",
            continuous_update=False,
            min=2,
            max=200,
            step=1,
            value=10,
        )
        dx = widgets.FloatSlider(
            description="data spacing",
            continuous_update=False,
            min=0.1,
            max=15,
            step=0.1,
            value=0.1,
        )
        moment = widgets.FloatText(description="M", value=30)
        depth_n = widgets.FloatSlider(
            description="depth$_{-Q}$",
            continuous_update=False,
            min=0,
            max=200,
            step=1,
            value=0,
        )
        depth_p = widgets.FloatSlider(
            description="depth$_{+Q}$",
            continuous_update=False,
            min=0,
            max=200,
            step=1,
            value=1,
        )
        profile = widgets.RadioButtons(
            options=["East", "North", "None"],
            value="East",
            description="profile",
            disabled=False,
        )
        fixed_scale = widgets.Checkbox(
            value=False, description="fixed scale", disabled=False
        )
        show_halfwidth = widgets.Checkbox(
            value=False, description="half width", disabled=False
        )

        out = widgets.interactive_output(
            self.magnetic_two_monopole_applet,
            {
                "component": component,
                "inclination": inclination,
                "declination": declination,
                "length": length,
                "dx": dx,
                "moment": moment,
                "depth_n": depth_n,
                "depth_p": depth_p,
                "profile": profile,
                "fixed_scale": fixed_scale,
                "show_halfwidth": show_halfwidth,
            },
        )
        left = widgets.VBox(
            [component, profile],
            layout=Layout(width="20%", height="400px", margin="60px 0px 0px 0px"),
        )
        right = widgets.VBox(
            [
                inclination,
                declination,
                length,
                dx,
                moment,
                depth_n,
                depth_p,
                fixed_scale,
                show_halfwidth,
            ],
            layout=Layout(width="50%", height="400px", margin="20px 0px 0px 0px"),
        )
        widgets.VBox(
            [out], layout=Layout(width="70%", height="400px", margin="0px 0px 0px 0px")
        )
        return widgets.HBox([left, out, right])

    def interact_plot_model_prism(self):
        plot = widgets.RadioButtons(
            options=["field", "model"],
            value="field",
            description="plot",
            disabled=False,
        )
        component = widgets.RadioButtons(
            options=["Bt", "Bx", "By", "Bz"],
            value="Bt",
            description="field",
            disabled=False,
        )

        inclination = widgets.FloatSlider(
            description="I", continuous_update=False, min=-90, max=90, step=1, value=90
        )
        declination = widgets.FloatSlider(
            description="D", continuous_update=False, min=-180, max=180, step=1, value=0
        )
        length = widgets.FloatSlider(
            description="length",
            continuous_update=False,
            min=2,
            max=200,
            step=1,
            value=72,
        )
        dx = widgets.FloatSlider(
            description="data spacing",
            continuous_update=False,
            min=0.1,
            max=15,
            step=0.1,
            value=2,
        )
        kappa = widgets.FloatText(description="$\kappa$", value=0.1)
        B0 = widgets.FloatText(description="B$_0$", value=56000)
        depth = widgets.FloatSlider(
            description="depth",
            continuous_update=False,
            min=0,
            max=50,
            step=1,
            value=10,
        )
        profile = widgets.RadioButtons(
            options=["East", "North", "None"],
            value="East",
            description="profile",
            disabled=False,
        )
        fixed_scale = widgets.Checkbox(
            value=False, description="fixed scale", disabled=False
        )

        show_halfwidth = widgets.Checkbox(
            value=False, description="half width", disabled=False
        )
        prism_dx = widgets.FloatText(description="$\\triangle x$", value=1)
        prism_dy = widgets.FloatText(description="$\\triangle y$", value=1)
        prism_dz = widgets.FloatText(description="$\\triangle z$", value=1)
        prism_inclination = widgets.FloatSlider(
            description="I$_{prism}$",
            continuous_update=False,
            min=-90,
            max=90,
            step=1,
            value=0,
        )
        prism_declination = widgets.FloatSlider(
            description="D$_{prism}$",
            continuous_update=False,
            min=0,
            max=180,
            step=1,
            value=0,
        )

        out = widgets.interactive_output(
            self.magnetic_prism_applet,
            {
                "plot": plot,
                "component": component,
                "inclination": inclination,
                "declination": declination,
                "length": length,
                "dx": dx,
                "kappa": kappa,
                "B0": B0,
                "depth": depth,
                "profile": profile,
                "fixed_scale": fixed_scale,
                "show_halfwidth": show_halfwidth,
                "prism_dx": prism_dx,
                "prism_dy": prism_dy,
                "prism_dz": prism_dz,
                "prism_inclination": prism_inclination,
                "prism_declination": prism_declination,
            },
        )
        left = widgets.VBox(
            [plot, component, profile],
            layout=Layout(width="20%", height="400px", margin="60px 0px 0px 0px"),
        )
        right = widgets.VBox(
            [
                inclination,
                declination,
                length,
                dx,
                B0,
                kappa,
                depth,
                prism_dx,
                prism_dy,
                prism_dz,
                prism_inclination,
                prism_declination,
                fixed_scale,
                show_halfwidth,
            ],
            layout=Layout(width="50%", height="400px", margin="20px 0px 0px 0px"),
        )
        widgets.VBox(
            [out], layout=Layout(width="70%", height="400px", margin="0px 0px 0px 0px")
        )
        return widgets.HBox([left, out, right])
