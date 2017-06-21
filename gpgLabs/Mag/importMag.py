import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import subprocess

import warnings
warnings.filterwarnings('ignore') # ignore warnings: only use this once you are sure things are working

from IPython.html.widgets import *
from IPython.display import display
from Library import *
from scipy.constants import mu_0

#subprocess.call("jupyter dashboards install --user --symlink --overwrite")
# from importMag import *
#subprocess.call("jupyter dashboards activate")
