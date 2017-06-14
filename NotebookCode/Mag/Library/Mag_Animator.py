import Mag as MAG
import simpegPF as PF
import simpegCoordUtils as Utils
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import animation
from JSAnimation import HTMLWriter


fig = plt.figure(figsize=(10,5))
ax1 = plt.subplot(121, projection='3d')
ax2 = plt.subplot(222)

# Define the dimensions of the prism (m)
dx, dy, dz = 1., 1, 1.
# Set the depth of burial (m)
depth = 0.5
pinc, pdec = 0., 0.
npts2D, xylim = 20., 3.
rx_h, View_elev, View_azim = 1., 20, -115
Einc, Edec, Bigrf = 45., 45., 50000.
x1, x2, y1, y2 = x1, x2, y1, y2 = -xylim, xylim, 0., 0.
comp = 'bz'
irt = 'total'
Q, rinc,rdec = 0., 0., 90.
susc = 0.1


ax1.axis('equal')
ax1.set_title(irt+' '+comp)
# Define the problem interactively
p = MAG.definePrism()
p.dx, p.dy, p.dz, p.z0 = dx, dy, dz, -depth
p.pinc, p.pdec = pinc, pdec

srvy = PF.survey()
srvy.rx_h, srvy.npts2D, srvy.xylim = rx_h, npts2D, xylim

# Create problem
prob = PF.problem()
prob.prism = p
prob.survey = srvy

X, Y = np.meshgrid(prob.survey.xr, prob.survey.yr)

x, y = MAG.linefun(x1, x2, y1, y2, prob.survey.npts2D)
xyz_line = np.c_[x, y, np.ones_like(x)*prob.survey.rx_h]



ax1.plot(x,y,xyz_line[:,2], 'w.', ms=3,lw=2)
ax1.text(-1,0., -3.25, 'B-field', fontsize=14, color='k', horizontalalignment='left')
im1 = ax1.contourf(X,Y,X)
im2 = ax2.plot(x,y)
im3 = ax2.plot(x,y)
im4 = ax2.plot(x,y)
im5 = ax1.text(0,0,0,'')
clim = np.asarray([-125,125])
def animate(ii):

    removePlt()

    if ii<19:
        dec = 0
        inc = -90. + ii*10.
        
    else:

        dec = 90.
        inc = 90. - (ii-19)*10.
        
    MAG.plotObj3D(p, rx_h, View_elev, View_azim, npts2D, xylim, profile="X", fig= fig, axs = ax1, plotSurvey=False)

    block_xyz = np.asarray([[-.2, -.2, .2, .2, 0],
                           [-.25, -.25, -.25, -.25, 0.5],
                           [-.2, .2, .2, -.2, 0]])

    # rot = Utils.mkvc(Utils.dipazm_2_xyz(pinc, pdec))

    # xyz = Utils.rotatePointsFromNormals(block_xyz.T, np.r_[0., 1., 0.], rot,
    #                                     np.r_[p.xc, p.yc, p.zc])

    R = Utils.rotationMatrix(inc, dec)

    xyz = R.dot(block_xyz).T

    #print xyz
    # Face 1
    ax1.add_collection3d(Poly3DCollection([zip(xyz[:4, 0]-1,
                                               xyz[:4, 1],
                                               xyz[:4, 2]-4)], facecolors='w'))

    ax1.add_collection3d(Poly3DCollection([zip(xyz[[1, 2, 4], 0]-1,
                                               xyz[[1, 2, 4], 1],
                                               xyz[[1, 2, 4], 2]-4)], facecolors='b'))

    ax1.add_collection3d(Poly3DCollection([zip(xyz[[0, 1, 4], 0]-1,
                                               xyz[[0, 1, 4], 1],
                                               xyz[[0, 1, 4], 2]-4)], facecolors='r'))

    ax1.add_collection3d(Poly3DCollection([zip(xyz[[2, 3, 4], 0]-1,
                                               xyz[[2, 3, 4], 1],
                                               xyz[[2, 3, 4], 2]-4)], facecolors='k'))

    ax1.add_collection3d(Poly3DCollection([zip(xyz[[0, 3, 4], 0]-1,
                                           xyz[[0, 3, 4], 1],
                                           xyz[[0, 3, 4], 2]-4)], facecolors='y'))


    

    # Create problem
    prob = PF.problem()
    prob.prism = p
    prob.survey = srvy

    prob.Bdec, prob.Binc, prob.Bigrf = dec, inc, Bigrf
    prob.Q, prob.rinc, prob.rdec = Q, rinc, rdec
    prob.uType, prob.mType = comp, 'total'
    prob.susc = susc

    # Compute fields from prism
    b_ind, b_rem = prob.fields()

    if irt == 'total':
        out = b_ind + b_rem

    elif irt == 'induced':
        out = b_ind

    else:
        out = b_rem



    #out = plogMagSurvey2D(prob, susc, Einc, Edec, Bigrf, x1, y1, x2, y2, comp, irt,  Q, rinc, rdec, fig=fig, axs1=ax2, axs2=ax3)



    #dat = axs1.contourf(X,Y, np.reshape(out, (X.shape)).T
    global im1
    im1 = ax1.contourf(X,Y,np.reshape(out, (X.shape)).T,zdir='z',offset=rx_h-0.1, alpha=0.75, clim=clim, vmin=clim[0],vmax=clim[1])

    global im5
    im5 = ax1.text(-.5,0,-4,'I: ' + str(inc) + ' D: ' + str(dec), horizontalalignment='left')

    # Create problem
    prob1D = PF.problem()
    srvy1D = PF.survey()
    srvy1D._rxLoc = xyz_line

    prob1D.prism = prob.prism
    prob1D.survey = srvy1D

    prob1D.Bdec, prob1D.Binc, prob1D.Bigrf = dec, inc, Bigrf
    prob1D.Q, prob1D.rinc, prob1D.rdec = Q, rinc, rdec
    prob1D.uType, prob1D.mType = comp, 'total'
    prob1D.susc = susc

    # Compute fields from prism
    out_linei, out_liner = prob1D.fields()

    #out_linei, out_liner = getField(p, xyz_line, comp, 'total')
    #out_linei = getField(p, xyz_line, comp,'induced')
    #out_liner = getField(p, xyz_line, comp,'remanent')

    out_linet = out_linei+out_liner

    distance = np.sqrt((x-x1)**2.+(y-y1)**2.)

    global im2
    im2 =ax2.plot(distance, out_linei, 'b.-')

    global im3
    im3 =ax2.plot(distance, out_liner, 'r.-')

    global im4
    im4 =ax2.plot(distance, out_linet, 'k.-')

    ax2.set_xlim(distance.min(), distance.max())
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Magnetic field (nT)")
    ax2.set_ylim(clim)

    ax2.legend(("induced", "remanent", "total"), bbox_to_anchor=(0.5, -0.3))
    ax2.grid(True)
    plt.show()



def removePlt():
    #global im1
    #im1.remove()

    global im1
    for coll in im1.collections:
        coll.remove()

    for cc in range(6):
        for coll in ax1.collections:
            ax1.collections.remove(coll)


    global im2
    im2.pop(0).remove()


    global im3
    im3.pop(0).remove()

    global im4
    im4.pop(0).remove()

    global im5
    im5.remove()

anim = animation.FuncAnimation(fig, animate,
                               frames=37 , interval=200,repeat=False)

anim.save('animation.html', writer=HTMLWriter(embed_frames=True,fps=10,default_mode = 'reflect'))
