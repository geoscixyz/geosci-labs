from importMag import *
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def arrowfun(b, arrowstyle="-|>", color='g', linestyle="dashed"):
    return Arrow3D([0.,b[0]],[0.,b[1]],[0.,b[2]], mutation_scale=20, lw=2, arrowstyle=arrowstyle, color=color,linestyle=linestyle)

def arrowfun2(a, b, arrowstyle="-|>", color='g', linestyle="dashed", lw=2):
    a = a
    b = b
    return Arrow3D([a[0],b[0]],[a[1],b[1]],[a[2],b[2]], mutation_scale=20, lw=lw, arrowstyle=arrowstyle, color=color,linestyle=linestyle)

def anglefun(a, b, arrowstyle="-|>", color='g', linestyle="dashed", lw=2):
    a = a*0.5
    b = b*0.5
    return Arrow3D([a[0],b[0]],[a[1],b[1]],[a[2],b[2]], mutation_scale=20, lw=lw, arrowstyle=arrowstyle, color=color,linestyle=linestyle)
    
def vectrogram(inc, dec):        
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca(projection='3d')
    y = Arrow3D([0,1],[0,0],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    x = Arrow3D([0,0],[0,1],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    z = Arrow3D([0,0],[0,0],[0,1], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    y0 = Arrow3D([-1,1],[1,1],[0,0], mutation_scale=20, lw=1, arrowstyle="-", color="k", linestyle="dashed")
    x0 = Arrow3D([1,1],[-1,1],[0,0], mutation_scale=20, lw=1, arrowstyle="-", color="k", linestyle="dashed")
    y1 = Arrow3D([-1,1],[-1.,-1.],[0,0], mutation_scale=20, lw=1, arrowstyle="-", color="k", linestyle="dashed")
    x1 = Arrow3D([-1.,-1.],[-1,1],[0,0], mutation_scale=20, lw=1, arrowstyle="-", color="k", linestyle="dashed")

    test = fatiandoUtils.ang2vec(1.,inc, dec)
    x0_vec0 = Arrow3D([0.,test[0]],[test[1],test[1]],[0., 0.], mutation_scale=20, lw=1, arrowstyle="-", color="g", linestyle="dashed")
    x0_vec1 = Arrow3D([0.,test[0]],[0., 0.],[test[2], test[2]], mutation_scale=20, lw=1, arrowstyle="-", color="g", linestyle="dashed")
    x0_vec2 = Arrow3D([0.,test[0]],[0., 0.],[0., 0.], mutation_scale=20, lw=1, arrowstyle="-", color="g", linestyle="dashed")

    y0_vec0 = Arrow3D([test[0],test[0]],[0.,test[1]],[0., 0.], mutation_scale=20, lw=1, arrowstyle="-", color="g", linestyle="dashed")
    y0_vec1 = Arrow3D([0., 0.],[0.,test[1]],[test[2],test[2]], mutation_scale=20, lw=1, arrowstyle="-", color="g", linestyle="dashed")
    y0_vec2 = Arrow3D([0., 0.],[0.,test[1]],[0.,0.], mutation_scale=20, lw=1, arrowstyle="-", color="g", linestyle="dashed")
    x0_vec = Arrow3D([0.,test[0]],[test[1],test[1]],[test[2],test[2]], mutation_scale=20, lw=1, arrowstyle="-", color="g", linestyle="dashed")
    y0_vec = Arrow3D([test[0],test[0]],[0.,test[1]],[test[2],test[2]], mutation_scale=20, lw=1, arrowstyle="-", color="g", linestyle="dashed")
    z0_vec = Arrow3D([test[0],test[0]],[test[1],test[1]],[0.,test[2]], mutation_scale=20, lw=1, arrowstyle="-", color="g", linestyle="dashed")
    z0_vec0 = Arrow3D([test[0],test[0]],[0.,0.],[0.,test[2]], mutation_scale=20, lw=1, arrowstyle="-", color="g", linestyle="dashed")
    z0_vec1 = Arrow3D([0.,0.],[test[1],test[1]],[0.,test[2]], mutation_scale=20, lw=1, arrowstyle="-", color="g", linestyle="dashed")
    z0_vec2 = Arrow3D([0.,0.],[0.,0.],[0.,test[2]], mutation_scale=20, lw=1, arrowstyle="-", color="g", linestyle="dashed")
    xy_vec = arrowfun(np.r_[test[0], test[1], 0.], arrowstyle="-", color='k', linestyle="solid")
    inc_vec = anglefun(np.r_[test[0], test[1], 0.],test,arrowstyle="->", color='r', linestyle="solid",lw=1)
    dec_vec = anglefun(np.r_[1.,0.,0.],np.r_[test[0], test[1], 0.],arrowstyle="->", color='b', linestyle="solid",lw=1)
    vec = arrowfun(test)
    if test[2]>0.:
        z0 = Arrow3D([1,1],[1,1],[0,1], mutation_scale=20, lw=1, arrowstyle="-", color="k", linestyle="dashed")
    else:
        z0 = Arrow3D([1,1],[1,1],[0,-1], mutation_scale=20, lw=1, arrowstyle="-", color="k", linestyle="dashed")
        y2 = Arrow3D([-1,1],[1.,1.],[-1,-1], mutation_scale=20, lw=1, arrowstyle="-", color="k", linestyle="dashed")
        x2 = Arrow3D([1.,1.],[-1,1],[-1,-1], mutation_scale=20, lw=1, arrowstyle="-", color="k", linestyle="dashed")
        ax.add_artist(x2)
        ax.add_artist(y2)        
    
    ax.add_artist(x)
    ax.add_artist(y)
    ax.add_artist(z)
    ax.add_artist(x0)
    ax.add_artist(y0)
    ax.add_artist(x1)
    ax.add_artist(y1)    
    ax.add_artist(z0)
    ax.add_artist(vec)
    ax.add_artist(xy_vec)
    ax.add_artist(inc_vec)
    ax.add_artist(dec_vec)
    
    ax.add_artist(y0_vec)
    ax.add_artist(x0_vec)
    ax.add_artist(y0_vec0)
    ax.add_artist(x0_vec0)
    ax.add_artist(y0_vec1)
    ax.add_artist(x0_vec1)
    ax.add_artist(y0_vec2)
    ax.add_artist(x0_vec2)

    ax.add_artist(z0_vec)
    ax.add_artist(z0_vec0)
    ax.add_artist(z0_vec1)
    ax.add_artist(z0_vec2)    

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("Northing (X)", color='k', fontsize = 16)
    ax.set_ylabel("Easting (Y)", color='k', fontsize = 16)
    ax.set_zlabel("Depth (Z)", color='k', fontsize = 16)
    ax.invert_zaxis()
    ax.invert_yaxis()
    ax.view_init(30,200)
    return 

def vectorwidget(inc0, dec0):
    Q = interactive(vectrogram, inc=widgets.FloatSlider(min=-90,max=90,step=10,value=inc0)
                ,dec=widgets.FloatSlider(min=-180,max=180,step=10,value=dec0))
    return Q

def projatob(a, b):
    bunja = (a*b).sum()
    bunmo = (b*b).sum()
    out = b*bunja/bunmo
    return out

def projgram(inc1, dec1, inc2, dec2):

    fig = plt.figure(figsize=(8,8))
    ax = fig.gca(projection='3d')
    x1 = fatiandoUtils.ang2vec(1., inc1, dec1)
    x2 = fatiandoUtils.ang2vec(1., inc2, dec2)
    x3 = projatob(x2, x1)
    
    vec1 = arrowfun(x1, arrowstyle="-|>", color='g', linestyle="solid")
    vec2 = arrowfun(x2, arrowstyle="-|>", color='r', linestyle="solid")
    vec3 = arrowfun(x3, arrowstyle="-|>", color='k', linestyle="solid")
    vec4 = arrowfun2(x2, x3, arrowstyle="-", color='k', linestyle="dashed")

    ax.add_artist(vec1)
    ax.add_artist(vec2)
    ax.add_artist(vec3)
    ax.add_artist(vec4)

      
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("Northing (X)", color='k', fontsize = 16)
    ax.set_ylabel("Easting (Y)", color='k', fontsize = 16)
    ax.set_zlabel("Depth (Z)", color='k', fontsize = 16)
    ax.invert_zaxis()
    ax.invert_yaxis()
    ax.view_init(30,-70)

def projwidget(inc1, dec1, inc2, dec2):
    Q = interactive(projgram
                ,inc1=widgets.FloatSlider(min=-90,max=90,step=10,value=inc1)
                ,dec1=widgets.FloatSlider(min=-180,max=180,step=10,value=dec1)
                ,inc2=widgets.FloatSlider(min=-90,max=90,step=10,value=inc2)
                ,dec2=widgets.FloatSlider(min=-180,max=180,step=10,value=dec2)
                )
    return Q
    
# from MagVector import *