import warnings
warnings.filterwarnings('ignore') 
import numpy as np
from IPython.display import HTML
from IPython.display import set_matplotlib_formats
from GPRlab1 import PrimaryWidget, PrimaryFieldWidget, PipeWidget, WallWidget
try:
    from IPython.html.widgets import  interact, interactive, IntSlider, widget, FloatText, FloatSlider
    pass
except Exception, e:    
    from ipywidgets import interact, interactive, IntSlider, widget, FloatText, FloatSlider
import matplotlib
set_matplotlib_formats('png')
matplotlib.rcParams['savefig.dpi'] = 100 # Change this to adjust figure size
from PIL import Image
