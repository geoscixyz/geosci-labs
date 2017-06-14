from IPython.display import set_matplotlib_formats
import warnings
warnings.filterwarnings('ignore') # ignore warnings: only use this once you are sure things are working

# from IPython.html.widgets import *
import matplotlib
from syntheticSeismogram import *
from EOSC350widget import wiggle, ViewWiggle
set_matplotlib_formats('png')
matplotlib.rcParams['savefig.dpi'] = 100 # Change this to adjust figure size
import numpy as np
