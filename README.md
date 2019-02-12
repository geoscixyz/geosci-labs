# GeoSci Labs

[![binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/geoscixyz/geosci-labs/master?filepath=notebooks%2Findex.ipynb)
[![azure](https://notebooks.azure.com/launch.png)](https://notebooks.azure.com/import/gh/geoscixyz/geosci-labs)
[![pypi](https://img.shields.io/pypi/v/geoscilabs.svg)](https://pypi.python.org/pypi/geoscilabs)
[![travis](https://travis-ci.org/geoscixyz/geosci-labs.svg?branch=master)](https://travis-ci.org/geoscixyz/geosci-labs)
[![License](https://img.shields.io/github/license/geoscixyz/geosci-labs.svg)](https://github.com/geoscixyz/geosci-labs/blob/master/LICENSE)
[![SimPEG](https://img.shields.io/badge/powered%20by-SimPEG-blue.svg)](http://simpeg.xyz)

This is a repository of code used to power the notebooks and interactive examples for https://em.geosci.xyz and https://gpg.geosci.xyz.

The examples are based on code available in [SimPEG](http://simpeg.xyz).

## Why

Interactive visualizations are a powerful way to interrogate mathematical equations. The goal of this repository is to be the home for code that can be plugged into jupyter notebooks so that we can play with the governing equations of geophysical electromagnetics.

## Scope

The repository contains the python code to run the ipython-widget style apps in http://github.com/geoscixyz/geosci-labs. These are mainly plotting code and some simple analytics. More complex numerical simulations depend on [SimPEG](http://simpeg.xyz)

## Usage

The notebooks can be run online through [Binder](#Binder), or [downloaded and run locally](#Locally).

### Binder

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/geoscixyz/geosci-labs/master?filepath=notebooks%2Findex.ipyn)

1. Launch the binder by clicking on the badge above or going to: https://mybinder.org/v2/gh/geoscixyz/geosci-labs/master?filepath=notebooks%2Findex.ipynb.
   This can sometimes take a couple minutes, so be patient...

2. Select the notebook of interest from the contents

3. [Run the Jupyter notebook](#Running-the-notebooks)

![Binder-steps](https://em.geosci.xyz/_images/binder-steps.png)

### Locally

To run them locally, you will need to have python installed, preferably through [anaconda](https://www.anaconda.com/download/).

You can then clone this reposiroty. From a command line, run

```
git clone https://github.com/geoscixyz/geosci-labs.git
```

Then `cd` into `geosci-labs`

```
cd geosci-labs
```

To setup your software environment, we recommend you use the provided conda environment

```
conda env create -f environment.yml
conda activate geosci-labs
```

alternatively, you can install dependencies through pypi
```
pip install -r requirements.txt
```

You can then launch Jupyter
```
jupyter notebook
```

Jupyter will then launch in your web-browser.

## Running the notebooks

Each cell of code can be run with `shift + enter` or you can run the entire notebook by selecting `cell`, `Run All` in the toolbar.

![cell-run-all](https://em.geosci.xyz/_images/run_all_cells.png)

For more information on running Jupyter notebooks, see the [Jupyter Documentation](https://jupyter.readthedocs.io/en/latest/)

## Using in a course

## Issues

If you run into problems or bugs, please let us know by [creating an issue](https://github.com/geoscixyz/em-apps/issues/new) in this repository.

## For Contributors

### Notebook Structure

Each notebook has the following structure

- **Purpose** : Motivation and key concepts addressed by the notebook
- **Setup** : Overview of the relevant parameters in the problem
- **Questions** : Guiding questions related to the purpose
- **App** : interactive visualizations
- **Explore** : further questions that can be explored with the app

### Setting up your environment

- to develop code for these notebooks, please see https://github.com/geoscixyz/geosci-labs
- add the notebook name and path to the [index](index.ipynb)

## Contributing


