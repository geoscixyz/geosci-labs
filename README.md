# em_examples

[![binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/geoscixyz/em-apps/master?filepath=index.ipynb)
[![azure](https://notebooks.azure.com/launch.png)](https://notebooks.azure.com/import/gh/geoscixyz/em-apps)
[![travis](https://travis-ci.org/geoscixyz/em_examples.svg?branch=master)](https://travis-ci.org/geoscixyz/em_examples)
[![pypi](https://img.shields.io/pypi/v/em_examples.svg)](https://pypi.python.org/pypi/SimPEG)

This is a repository of code used to power the notebooks and interactive examples for http://em.geosci.xyz. The associated notebooks are in the repository [em_apps](http://github.com/geoscixyz/em_apps). 
The notebooks are available on
- [Binder](https://mybinder.org/v2/gh/geoscixyz/em-apps/master?filepath=index.ipynb)
- [Azure Notebooks](https://notebooks.azure.com/import/gh/geoscixyz/em-apps)

The examples are based on code available in [SimPEG](http://simpeg.xyz). 

## Why

Interactive visualizations are a powerful way to interrogate mathematical equations. The goal of this repository is to be the home for code that can be plugged into jupyter notebooks so that we can play with the governing equations of geophysical electromagnetics.

## Scope

The repository contains the python code to run the ipython-widget style apps in http://github.com/geoscixyz/em-apps. These are mainly plotting code and some simple analytics. More complex numerical simulations depend on `SimPEG <http://simpeg.xyz>`_

Installing
----------

`em_examples` is on PyPi

```
pip install em_examples
```

For developers you can clone this repo from github:

```
git clone https://github.com/geoscixyz/em_examples.git
```

and install

```
python setup.py install
```


