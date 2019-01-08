from __future__ import print_function
from __future__ import absolute_import

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np


class MyApp(ipywidgets.Box):
    def __init__(self, widgets, kwargs):
        self._kwargs = kwargs
        self._widgets = widgets
        super(MyApp, self).__init__(widgets)
        self.layout.display = "flex"
        self.layout.flex_flow = "column"
        self.layout.align_items = "stretch"

    @property
    def kwargs(self):
        return dict(
            [
                (key, val.value)
                for key, val in self._kwargs.items()
                if isinstance(val, (ipywidgets.widget.Widget, ipywidgets.fixed))
            ]
        )


def widgetify(fun, layout=None, manual=False, **kwargs):

    f = fun

    if manual:
        app = ipywidgets.interact_manual(f, **kwargs)
        app = app.widget
    else:
        app = ipywidgets.interactive(f, **kwargs)

    # if layout is None:
    # TODO: add support for changing layouts
    w = MyApp(app.children, kwargs)

    f.widget = w
    # defaults =  #dict([(key, val.value) for key, val in kwargs.iteritems() if isinstance(val, Widget)])
    app.update()
    # app.on_displayed(f(**(w.kwargs)))

    return w


def clipsign(value, clip):
    clipthese = abs(value) > clip
    return value * ~clipthese + np.sign(value) * clip * clipthese


def wiggle(
    traces,
    skipt=1,
    scale=1.0,
    lwidth=0.1,
    offsets=None,
    redvel=0.0,
    manthifts=None,
    tshift=0.0,
    sampr=1.0,
    clip=10.0,
    dx=1.0,
    color="black",
    fill=True,
    line=True,
    ax=None,
):

    ns = traces.shape[1]
    ntr = traces.shape[0]
    t = np.arange(ns) * sampr

    def timereduce(offsets, redvel, shift):
        return [float(offset) / redvel + shift for offset in offsets]

    if offsets is not None:
        shifts = timereduce(offsets, redvel, tshift)
    elif manthifts is not None:
        shifts = manthifts
    else:
        shifts = np.zeros((ntr,))

    if ax is None:
        _, ax = plt.subplots(1, 1)

    for i in range(0, ntr, skipt):
        trace = traces[i].copy()
        trace[0] = 0
        trace[-1] = 0

        if line:
            ax.plot(
                i * dx + clipsign(trace / scale, clip),
                t - shifts[i],
                color=color,
                linewidth=lwidth,
            )
        if fill:
            for j in range(ns):
                if trace[j] < 0:
                    trace[j] = 0
            ax.fill(
                i * dx + clipsign(trace / scale, clip),
                t - shifts[i],
                color=color,
                linewidth=0,
            )
