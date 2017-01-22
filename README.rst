em_examples
===========

.. image:: http://mybinder.org/badge.svg
      :target: http://mybinder.org/repo/geoscixyz/em_apps
      :alt: binder

.. image:: https://travis-ci.org/geoscixyz/em_examples.svg?branch=master
      :target: https://travis-ci.org/geoscixyz/em_examples
      :alt: build_status

This is a repository of code used to power the notebooks and interactive examples for http://em.geosci.xyz. The associated notebooks are in the repository `em_apps <http://github.com/geoscixyz/em_apps>`_ The examples are based on code available in `SimPEG <http://simpeg.xyz>`_

Why
---

Interactive visualizations are a powerful way to interrogate mathematical equations. The goal of this repository is to be the home for code that can be plugged into jupyter notebooks so that we can play with the governing equations of geophysical electromagnetics.

Scope
-----

The repository contains the python code to run the ipython-widget style apps in http://github.com/geoscixyz/em_apps. These are mainly plotting code and some simple analytics. More complex numerical simulations depend on `SimPEG <http://simpeg.xyz>`_

Installing
----------

:code:`em_examples` is on PyPi

.. code::

    pip install em_examples


For developers you can clone this repo from github:

.. code::

    git clone https://github.com/geoscixyz/em_examples.git

and install

.. code::

    python setup.py install


