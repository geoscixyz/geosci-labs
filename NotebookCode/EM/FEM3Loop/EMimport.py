import warnings
warnings.filterwarnings('ignore') # ignore warnings: only use this once you are sure things are working
from IPython.display import HTML, set_matplotlib_formats, display
import matplotlib
set_matplotlib_formats('png')
matplotlib.rcParams['savefig.dpi'] = 100 # Change this to adjust figure size

try:
    from IPython.html.widgets import *
    pass
except Exception, e:    
    from ipywidgets import interact, interactive, IntSlider, widget, FloatText, FloatSlider

from FEM3loop import fem3loop, interactfem3loop

