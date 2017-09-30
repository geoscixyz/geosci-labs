em_apps
=======

.. .. image:: http://mybinder.org/badge.svg
..    :target: https://beta.mybinder.org/repo/geoscixyz/em_apps
..    :alt: binders

.. image:: https://travis-ci.org/geoscixyz/em_apps.svg?branch=master
    :target: https://travis-ci.org/geoscixyz/em_apps
    :alt: travis

This is a repo of notebooks and interactive examples for http://em.geosci.xyz. The examples are based on code available in
`em_examples <http://github.com/geoscixyz/em_examples>`_. Numerical simulations are based on `SimPEG <http://simpeg.xyz>`_

The notebooks are available on
.. - `binder <https://beta.mybinder.org/repo/geoscixyz/em_apps>`_
- `Azure Notebooks <https://notebooks.azure.com/lheagy/libraries/em_apps>`_

Notebook Structure
------------------

Each notebook has the following structure

- **Purpose** : Motivation and key concepts addressed by the notebook
- **Setup** : Overview of the relevant parameters in the problem
- **Questions** : Guiding questions related to the purpose
- **App** : interactive visualizations
- **Explore** : further questions that can be explored with the app

Conventions
------------

For colormaps (http://matplotlib.org/examples/color/colormaps_reference.html)
- **fields** are plotted with the `viridis`
- **potentials** are plotted with `viridis`
- **sensitivities** are plotted with `viridis`
- **physical properties** are plotted with `jet`
- **charges** are plotted with `RdBu`

Order of widgets:

- geometry of survey
- geomerty target
- physical properties of target
- view options

For developers
--------------

- to develop code for these notebooks, please see http://github.com/geoscixyz/em_examples
- when you generate a new notebook, please make sure that the filepath to its location follows the same structure as in `EM GeoSci <http://em.geosci.xyz>`_
- add the notebook name and path to the index (index.ipynb)
.. - and update the `binder <https://beta.mybinder.org>`_ so it can be shared with the world!

.. .. image:: ./images/binders.png


