from EOSC350widget import CleanNMOWidget, NoisyNMOWidget, NMOstackSingle, NMOstackthree, wiggle, ViewWiggle
import warnings
warnings.filterwarnings('ignore') # ignore warnings: only use this once you are sure things are working
from IPython.display import set_matplotlib_formats
from IPython.html.widgets import *
import matplotlib
set_matplotlib_formats('png')
matplotlib.rcParams['savefig.dpi'] = 70 # Change this to adjust figure size
import numpy as np
syndata = np.load('syndata1.npy')
obsdata = np.load('obsdata1.npy')
