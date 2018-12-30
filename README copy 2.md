**[overview](#overview) | [launching the notebooks](#launching-the-notebooks) | [running the notebooks](#running-the-notebooks) | [issues](#issues) | [contributing](#for-contributors)**

# gpgLabs

[![binder](http://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/geoscixyz/gpgLabs/master?filepath=Notebooks%2Findex.ipynb)
[![travis](https://travis-ci.org/geoscixyz/gpgLabs.svg?branch=master)](https://travis-ci.org/geoscixyz/gpgLabs)
[![SimPEG](https://img.shields.io/badge/powered%20by-SimPEG-blue.svg)](http://simpeg.xyz)

## Overview

This is a repo of interactive examples for http://gpg.geosci.xyz.

The notebooks are available on
- [Binder](https://mybinder.org/v2/gh/geoscixyz/gpgLabs/master?filepath=Notebooks%2Findex.ipynb)

<!-- <img src="https://em.geosci.xyz/_images/DC_LayeredEarth_notebook.png" width=60% align="center">
 -->

## Launching the notebooks

The notebooks can be run online through [Binder](#Binder), or [downloaded and run locally](#Locally).

### Binder

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/geoscixyz/gpgLabs/master?filepath=Notebooks%2Findex.ipyn)

1. Launch the binder by clicking on the badge above or going to: https://mybinder.org/v2/gh/geoscixyz/gpgLabs/master?filepath=Notebooks%2Findex.ipynb.
   This can sometimes take a couple minutes, so be patient...

2. Select the notebook of interest from the contents

3. [Run the Jupyter notebook](#Running-the-notebooks)

![Binder-steps](https://em.geosci.xyz/_images/binder-steps.png)

### Locally

To run them locally, you will need to have python installed, preferably through [anaconda](https://www.anaconda.com/download/).

You can then clone this reposiroty. From a command line, run

```
git clone https://github.com/geoscixyz/gpgLabs.git
```

Then `cd` into `gpgLabs`

```
cd gpgLabs
```

To setup your software environment, we recommend you use the provided conda environment

```
conda env create -f environment.yml
conda activate gpgLabs-environment
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

- to develop code for these notebooks, please see https://github.com/geoscixyz/gpgLabs
- add the notebook name and path to the [index](index.ipynb)


