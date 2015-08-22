import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import warnings
warnings.filterwarnings('ignore') # ignore warnings: only use this once you are sure things are working

from IPython.html.widgets import *
from IPython.display import display
from fromFatiando import *
from fromSimPEG import *
from scipy.constants import mu_0
from Mag import *

# from importMag import *