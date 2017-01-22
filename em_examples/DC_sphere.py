from SimPEG import *
from SimPEG.EM.Static import DC
from scipy.constants import epsilon_0

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    from ipywidgets import interact, IntSlider, FloatSlider, FloatText, ToggleButtons
    pass
except Exception, e:
    from IPython.html.widgets import  interact, IntSlider, FloatSlider, FloatText, ToggleButtons


# def define_model(sig0,sig1,)
# def plot_model()
# def 2D_cylinder_field
