from __future__ import print_function
from __future__ import absolute_import

import ipywidgets
import matplotlib.pyplot as plt


class MyApp(ipywidgets.Box):
    def __init__(self, widgets, kwargs):
        self._kwargs = kwargs
        self._widgets = widgets
        super(MyApp, self).__init__(widgets)
        self.layout.display = 'flex'
        self.layout.flex_flow = 'column'
        self.layout.align_items = 'stretch'

    @property
    def kwargs(self):
        return dict([
            (key, val.value) for key, val in self._kwargs.items()
            if isinstance(val, (ipywidgets.widget.Widget, ipywidgets.fixed))
        ])


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
    #app.on_displayed(f(**(w.kwargs)))

    return w
