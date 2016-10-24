#from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore') # ignore warnings: only use this once you are sure things are working
from IPython.display import HTML
from IPython.display import set_matplotlib_formats
import matplotlib
set_matplotlib_formats('png')
matplotlib.rcParams['savefig.dpi'] = 100 # Change this to adjust figure size

from Attenuation import AttenuationWidgetTBL
