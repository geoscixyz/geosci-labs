from ipywidgets import (
    interactive, VBox, HBox, Box, Widget
)
from IPython.display import display


class MyApp(Box):
    def __init__(self, widgets, kwargs):
        self._kwargs = kwargs

        super(MyApp, self).__init__(widgets)
        self.layout.display = 'flex'
        self.layout.flex_flow = 'column'
        self.layout.align_items = 'stretch'

    @property
    def kwargs(self):
        return dict(
            [(key, val.value) for key, val in self._kwargs.iteritems()
            if isinstance(val, Widget)]
        )



def widgetify(fun, layout=None, **kwargs):
    f = fun
    app = interactive(f, **kwargs)

    # if layout is None:
    # TODO: add support for changing layouts
    w = MyApp(app.children, kwargs)

    f.widget = w
    # defaults =  #dict([(key, val.value) for key, val in kwargs.iteritems() if isinstance(val, Widget)])
    app.on_displayed(f(**(w.kwargs)))

    return w
