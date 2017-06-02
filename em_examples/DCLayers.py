from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
from SimPEG import Mesh, Maps, Utils, SolverLU
from scipy.constants import epsilon_0

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ipywidgets import (
    IntSlider, FloatSlider, FloatText, ToggleButtons
)

from .Base import widgetify

# Mesh parameters
npad = 20
cs = 0.5
hx = [(cs, npad, -1.3), (cs, 200), (cs, npad, 1.3)]
hy = [(cs, npad, -1.3), (cs, 100)]
mesh = Mesh.TensorMesh([hx, hy], "CN")

# bounds on electrical resistivity
rhomin = 1e2
rhomax = 1e3

eps = 1e-9 # to stabilize division
infinity = 100 # what is "far enough"


def r(xyz, src_loc):
    """
    Distance from source to points on an xyz grid
    """
    return np.sqrt(
        (xyz[:, 0] - src_loc[0])**2 +
        (xyz[:, 1] - src_loc[1])**2 +
        (xyz[:, 2] - src_loc[2])**2
     )+ eps


def sum_term(rho1, rho2, h, r):
    m = Utils.mkvc(np.arange(1, infinity+1))
    k = (rho2-rho1) / (rho2+rho1)
    return np.sum(
        (
            (k**m.T)*np.ones_like(Utils.mkvc(r, 2))) /
            np.sqrt(1. + (2.*h*m.T/Utils.mkvc(r, 2))**2
        ), 1
    )

def sum_term_deriv(rho1, rho2, h, r):
    m = Utils.mkvc(np.arange(1, infinity+1))
    k = (rho2-rho1) / (rho2+rho1)
    return np.sum(
        ((k**m.T)*np.ones_like(Utils.mkvc(r, 2))) /
        (1. + (2.*h*m.T/Utils.mkvc(r, 2))**2)**(3./2.) *
        ((2.*h*m.T)**2 / Utils.mkvc(r, 2)**3) ,
        1
    )


def layer_potentials(rho1, rho2, h, A, B, xyz):

    """
    Compute analytic solution of surface potential for 2-layered Earth
    (Ref Telford 1990, section 8.3.4)s
    """

    def V(I, src_loc):
        return (
            (I*rho1 / (2.*np.pi*r(xyz, src_loc))) *
            (1 + 2*sum_term(rho1, rho2, h, r(xyz, src_loc)))
        )

    VA = V(1., A)
    VB = V(-1., B)

    return VA+VB

def layer_E(rho1, rho2, h, A, B, xyz):
    k = (rho2-rho1) / (rho2+rho1)

    dr_dx = lambda src_loc: (xyz[:, 0] - src_loc[0]) / r(xyz, src_loc)
    dr_dy = lambda src_loc: (xyz[:, 1] - src_loc[1]) / r(xyz, src_loc)
    dr_dz = lambda src_loc: (xyz[:, 2] - src_loc[2]) / r(xyz, src_loc)

    m = Utils.mkvc(np.arange(1, infinity+1))

    def deriv_1(r):
        return (-1./r) * (1. + 2.*sum_term(rho1, rho2, h, r))

    def deriv_2(r):
        return (2.*sum_term_deriv(rho1, rho2, h, r))

    def Er(I, r):
        return - (I*rho1 / (2.*np.pi*r)) * (deriv_1(r) + deriv_2(r))

    def Ex(I, src_loc):
        return Er(I, r(xyz, src_loc)) * dr_dx(src_loc)

    def Ey(I, src_loc):
        return Er(I, r(xyz, src_loc)) * dr_dy(src_loc)

    def Ez(I, src_loc):
        return Er(I, r(xyz, src_loc)) * dr_dz(src_loc)

    ex = Ex(1., A) + Ex(-1., B)
    ey = Ey(1., A) + Ey(-1., B)
    ez = Ez(1., A) + Ez(-1., B)

    return ex, ey, ez

def layer_J(rho1, rho2, h, A, B, xyz):
    ex, ey, ez = layer_E(rho1, rho2, h, A, B, xyz)

    sig = 1./rho2*np.ones_like(xyz[:, 0])

    # print sig
    sig[xyz[:, 1] >= -h] = 1./rho1 # since the model is 2D

    return sig * ex, sig * ey, sig * ez

def G(A, B, M, N):
    """
    Geometric factor
    """
    return (
        1. / (
            1./(np.abs(A-M)+eps) -
            1./(np.abs(M-B)+eps) -
            1./(np.abs(N-A)+eps) +
            1./(np.abs(N-B)+eps)
        )
    )

def rho_a (VM, VN, A, B, M, N):
    """
    Apparent Resistivity
    """
    return (VM-VN)*2.*np.pi*G(A, B, M, N)

def solve_2D_potentials(rho1, rho2, h, A, B):
    """
    Here we solve the 2D DC problem for potentials (using SimPEG Mesg Class)
    """
    sigma = 1./rho2*np.ones(mesh.nC)
    sigma[mesh.gridCC[:, 1] >= -h] = 1./rho1 # since the model is 2D

    q = np.zeros(mesh.nC)
    a = Utils.closestPoints(mesh, A[:2])
    b = Utils.closestPoints(mesh, B[:2])

    q[a] = 1./mesh.vol[a]
    q[b] = -1./mesh.vol[b]

    # q = q * 1./mesh.vol

    A = (
        mesh.cellGrad.T *
        Utils.sdiag(1./(mesh.dim * mesh.aveF2CC.T * (1./sigma))) *
        mesh.cellGrad
    )
    Ainv = SolverLU(A)

    V = Ainv * q
    return V

def solve_2D_E(rho1, rho2, h, A, B):
    """
    solve the 2D DC resistivity problem for electric fields
    """

    V = solve_2D_potentials(rho1, rho2, h, A, B)
    E = -mesh.cellGrad * V
    E = mesh.aveF2CCV * E
    ex = E[:mesh.nC]
    ez = E[mesh.nC:]
    return ex, ez, V

def solve_2D_J(rho1, rho2, h, A, B):

    ex, ez, V = solve_2D_E(rho1, rho2, h, A, B)
    sigma = 1./rho2*np.ones(mesh.nC)
    sigma[mesh.gridCC[:, 1] >= -h] = 1./rho1 # since the model is 2D

    return Utils.sdiag(sigma) * ex, Utils.sdiag(sigma) * ez, V


def plot_layer_potentials(rho1, rho2, h, A, B, M, N, imgplt='Model'):

    markersize = 6.
    fontsize = 10.
    ylim = np.r_[-1., 1.]*rhomax/(5*2*np.pi)*1.5

    fig, ax = plt.subplots(2, 1, figsize=(9, 7))

    fig.subplots_adjust(right=0.8)
    x = np.linspace(-40., 40., 200)
    z = np.linspace(x.min(), 0, 100)

    pltgrid = Utils.ndgrid(x, z)
    xplt = pltgrid[:, 0].reshape(x.size, z.size, order='F')
    zplt = pltgrid[:, 1].reshape(x.size, z.size, order='F')

    V = layer_potentials(
        rho1, rho2, h, np.r_[A, 0., 0.], np.r_[B, 0., 0.],
        Utils.ndgrid(x, np.r_[0.], np.r_[0.])
    )
    VM = layer_potentials(
        rho1, rho2, h, np.r_[A, 0., 0.], np.r_[B, 0., 0.],
        Utils.mkvc(np.r_[M, 0., 0], 2).T
    )
    VN = layer_potentials(
        rho1, rho2, h, np.r_[A, 0., 0.], np.r_[B, 0., 0.],
        Utils.mkvc(np.r_[N, 0., 0], 2).T
    )

    ax[0].plot(x, V, color=[0.1, 0.5, 0.1], linewidth=2)
    ax[0].grid(which='both', linestyle='-', linewidth=0.5, color=[0.2, 0.2, 0.2], alpha=0.5)
    ax[0].plot(A, 0, '+', markersize = 12, markeredgewidth = 3, color=[1., 0., 0])
    ax[0].plot(B, 0, '_', markersize = 12, markeredgewidth = 3, color=[0., 0., 1.])
    ax[0].set_ylabel('Potential, (V)', fontsize = 14)
    ax[0].set_xlabel('x (m)', fontsize = 14)
    ax[0].set_xlim([x.min(), x.max()])
    ax[0].set_ylim(ylim)

    ax[0].plot(M, VM, 'o', color='k')
    ax[0].plot(N, VN, 'o', color='k')

    props = dict(boxstyle='round', facecolor='grey', alpha=0.3)

    txtsp = 1

    xytextM = (M+0.5, np.max([np.min([VM, ylim.max()]), ylim.min()])+0.5)
    xytextN = (N+0.5, np.max([np.min([VN, ylim.max()]), ylim.min()])+0.5)


    props = dict(boxstyle='round', facecolor='grey', alpha=0.4)

    ax[0].annotate('%2.1e'%(VM), xy=xytextM, xytext=xytextM, fontsize = 14)
    ax[0].annotate('%2.1e'%(VN), xy=xytextN, xytext=xytextN, fontsize = 14)

    # ax[0].plot(np.r_[M, N], np.ones(2)*VN, color='k')
    # ax[0].plot(np.r_[M, M], np.r_[VM, VN], color='k')
    # ax[0].annotate('%2.1e'%(VM-VN) , xy=(M, (VM+VN)/2), xytext=(M-9, (VM+VN)/2.), fontsize = 14)

    props = dict(boxstyle='round', facecolor='grey', alpha=0.4)
    ax[0].text(x.max()+1, ylim.max()-0.1*ylim.max(), '$\\rho_a$ = %2.2f'%(rho_a(VM, VN, A, B, M, N)),
                verticalalignment='bottom', bbox=props, fontsize = 14)

    if imgplt == 'Model':
        model = rho2*np.ones(pltgrid.shape[0])
        model[pltgrid[:, 1] >= -h] = rho1
        model = model.reshape(x.size, z.size, order='F')
        cb = ax[1].pcolor(xplt, zplt, model, norm=LogNorm())
        ax[1].plot([xplt.min(), xplt.max()], -h*np.r_[1., 1], color=[0.5, 0.5, 0.5], linewidth = 1.5 )

        clim = [rhomin, rhomax]
        clabel = 'Resistivity ($\Omega$m)'

    # elif imgplt == 'potential':
    #     Vplt = layer_potentials(rho1, rho2, h, np.r_[A, 0., 0.], np.r_[B, 0., 0.], np.c_[pltgrid, np.zeros_like(pltgrid[:, 0])])
    #     Vplt = Vplt.reshape(x.size, z.size, order='F')
    #     cb = ax[1].pcolor(xplt, zplt, Vplt)
    #     ax[1].contour(xplt, zplt, np.abs(Vplt), np.logspace(-2., 1., 10), colors='k', alpha=0.5)
    #     ax[1].set_ylabel('z (m)', fontsize=14)
    #     clim = ylim
    #     clabel = 'Potential (V)'

    elif imgplt == 'Potential':
        Pc = mesh.getInterpolationMat(pltgrid, 'CC')

        V = solve_2D_potentials(rho1, rho2, h, np.r_[A, 0., 0.], np.r_[B, 0., 0.])

        Vplt = Pc * V
        Vplt = Vplt.reshape(x.size, z.size, order='F')

        # since we are using a strictly 2D code, the potnetials at the surface
        # do not match the analytic, so we scale the potentials to match the
        # analytic 2.5D result at the surface.
        fudgeFactor = layer_potentials(
            rho1, rho2, h, np.r_[A, 0., 0.], np.r_[B, 0., 0.],
            np.c_[x.min(), 0., 0.]
        ) / Vplt[0, 0]

        cb = ax[1].pcolor(xplt, zplt, Vplt * fudgeFactor, cmap="viridis")
        ax[1].plot([xplt.min(), xplt.max()], -h*np.r_[1., 1], color=[0.5, 0.5, 0.5], linewidth = 1.5 )
        ax[1].contour(xplt, zplt, np.abs(Vplt), colors='k', alpha=0.5)
        ax[1].set_ylabel('z (m)', fontsize=14)
        clim = np.r_[-15., 15.]
        clabel = 'Potential (V)'

    elif imgplt == 'E':

        Pc = mesh.getInterpolationMat(pltgrid, 'CC')

        ex, ez, V = solve_2D_E(rho1, rho2, h, np.r_[A, 0., 0.], np.r_[B, 0., 0.])

        ex, ez = Pc * ex, Pc * ez
        Vplt = (Pc*V).reshape(x.size, z.size, order='F')
        fudgeFactor = layer_potentials(rho1, rho2, h, np.r_[A, 0., 0.], np.r_[B, 0., 0.], np.c_[x.min(), 0., 0.] ) / Vplt[0, 0]


        # ex, ez, _ = layer_E(rho1, rho2, h, np.r_[A, 0., 0.], np.r_[B, 0., 0.], np.c_[pltgrid, np.zeros_like(pltgrid[:, 0])])
        ex = fudgeFactor * ex.reshape(x.size, z.size, order='F')
        ez = fudgeFactor * ez.reshape(x.size, z.size, order='F')
        e = np.sqrt(ex**2.+ez**2.)

        cb = ax[1].pcolor(xplt, zplt, e, cmap="viridis", norm=LogNorm())
        ax[1].plot([xplt.min(), xplt.max()], -h*np.r_[1., 1], color=[0.5, 0.5, 0.5], linewidth = 1.5 )
        clim = np.r_[3e-3, 1e1]

        ax[1].streamplot(x, z, ex.T, ez.T, color = 'k', linewidth= 2*(np.log(e.T) - np.log(e).min())/(np.log(e).max() - np.log(e).min()))


        clabel = 'Electric Field (V/m)'

    elif imgplt == 'J':

        Pc = mesh.getInterpolationMat(pltgrid, 'CC')

        Jx, Jz, V = solve_2D_J(rho1, rho2, h, np.r_[A, 0., 0.], np.r_[B, 0., 0.])

        Jx, Jz = Pc * Jx, Pc * Jz

        Vplt = (Pc*V).reshape(x.size, z.size, order='F')
        fudgeFactor = layer_potentials(
            rho1, rho2, h, np.r_[A, 0., 0.], np.r_[B, 0., 0.], np.c_[x.min(), 0., 0.]
        ) / Vplt[0, 0]

        Jx = fudgeFactor * Jx.reshape(x.size, z.size, order='F')
        Jz = fudgeFactor * Jz.reshape(x.size, z.size, order='F')

        J = np.sqrt(Jx**2.+Jz**2.)

        cb = ax[1].pcolor(xplt, zplt, J, cmap="viridis", norm=LogNorm())
        ax[1].plot([xplt.min(), xplt.max()], -h*np.r_[1., 1], color=[0.5, 0.5, 0.5], linewidth = 1.5 )
        ax[1].streamplot(x, z, Jx.T, Jz.T, color = 'k', linewidth = 2*(np.log(J.T)-np.log(J).min())/(np.log(J).max() - np.log(J).min()) )
        ax[1].set_ylabel('z (m)', fontsize=14)

        clim = np.r_[3e-5, 3e-2]
        clabel = 'Current Density (A/m$^2$)'

    ax[1].set_xlim([x.min(), x.max()])
    ax[1].set_ylim([z.min(), 5.])
    ax[1].set_ylabel('z (m)', fontsize=14)
    cbar_ax = fig.add_axes([1., 0.08, 0.04, 0.4])
    plt.colorbar(cb, cax=cbar_ax, label=clabel)
    if 'clim' in locals():
        cb.set_clim(clim)
    ax[1].set_xlabel('x(m)', fontsize=14)

    xytextA1 = (A-0.5,2)
    xytextB1 = (B-0.5,2)
    xytextM1 = (M-0.5,2)
    xytextN1 = (N-0.5,2)

    ax[1].plot(A,1.,marker = 'v',color='red', markersize=markersize)
    ax[1].plot(B,1.,marker = 'v',color='blue', markersize=markersize)
    ax[1].plot(M,1.,marker = '^',color='yellow', markersize=markersize)
    ax[1].plot(N,1.,marker = '^',color='green', markersize=markersize)

    ax[1].annotate('A', xy=xytextA1, xytext=xytextA1,fontsize=fontsize)
    ax[1].annotate('B', xy=xytextB1, xytext=xytextB1,fontsize=fontsize)
    ax[1].annotate('M', xy=xytextM1, xytext=xytextM1,fontsize=fontsize)
    ax[1].annotate('N', xy=xytextN1, xytext=xytextN1,fontsize=fontsize)


    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_layer_potentials_app():
    plot_layer_potentials_interact = (
        lambda A, B, M, N, rho1, rho2, h, Plot:
        plot_layer_potentials(rho1, rho2, h, A, B, M, N, Plot)
    )
    app = widgetify(
        plot_layer_potentials_interact,
        A = FloatSlider(
            min=-30., max=30., step=1., value=-30., continuous_update=False
        ),
        B = FloatSlider(
            min=-30., max=30., step=1., value=30., continuous_update=False
        ),
        M = FloatSlider(
            min=-30., max=30., step=1., value=-10., continuous_update=False
        ),
        N = FloatSlider(
            min=-30., max=30., step=1., value=10., continuous_update=False
        ),
        rho1 = FloatSlider(
            min=rhomin, max=rhomax, step=10., value = 500.,
            continuous_update=False, description='$\\rho_1$'
        ),
        rho2 = FloatSlider(
            min=rhomin, max=rhomax, step=10., value = 500.,
            continuous_update=False, description='$\\rho_2$'
        ),
        h = FloatSlider(
            min=0., max=40., step=1., value=5., continuous_update=False
        ),
        Plot = ToggleButtons(
            options =['Model', 'Potential', 'E', 'J', ], value='Model'
        ),
    )
    return app

if __name__ == '__main__':
    rho1, rho2 = rhomin, rhomax
    h = 5.
    A, B = -30., 30.
    M, N = -10., 10.
    Plot =  'e'
    plot_layer_potentials(rho1, rho2, h, A, B, M, N, Plot)
