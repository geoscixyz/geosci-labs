**[overview](#overview) | [launching the notebooks](#launching-the-notebooks) | [running the notebooks](#running-the-notebooks) | [issues](#issues) | [contributing](#for-contributors)**

# em-apps

[![binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/geoscixyz/em-apps/master?filepath=index.ipynb)
[![azure](https://notebooks.azure.com/launch.png)](https://notebooks.azure.com/import/gh/geoscixyz/em-apps)
[![travis](https://travis-ci.org/geoscixyz/em-apps.svg?branch=master)](https://travis-ci.org/geoscixyz/em-apps)

## Overview

This is a repo of notebooks and interactive examples for http://em.geosci.xyz. The examples are based on code available in
[em_examples](http://github.com/geoscixyz/em_examples). Numerical simulations are based on [SimPEG](http://simpeg.xyz). 

The notebooks are available on
- [Binder](https://mybinder.org/v2/gh/geoscixyz/em-apps/master?filepath=index.ipynb)
- [Azure Notebooks](https://notebooks.azure.com/import/gh/geoscixyz/em-apps)

<img src="https://em.geosci.xyz/_images/DC_LayeredEarth_notebook.png" width=60% align="center">

## Launching the notebooks

The notebooks can be run online through [Binder](#Binder) or [Azure notebooks](#Azure), or [downloaded and run locally](#Locally). 

### Binder

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/geoscixyz/em-apps/master?filepath=index.ipynb)

1. Launch the binder by clicking on the badge above or going to: https://mybinder.org/v2/gh/geoscixyz/em-apps/master?filepath=index.ipynb.
   This can sometimes take a couple minutes, so be patient...

2. Select the notebook of interest from the contents

3. [Run the Jupyter notebook](#Running-the-notebooks)

![Binder-steps](https://em.geosci.xyz/_images/binder-steps.png)

### Azure

[![Azure](https://notebooks.azure.com/launch.png)](https://notebooks.azure.com/import/gh/geoscixyz/em-apps)

1. Navigate to Azure notebooks by clicking on the badge above or going to: https://notebooks.azure.com/import/gh/geoscixyz/em-apps,
   and select `Import` to import the library

2. Sign in to your microsoft account (or `Create One` if you do not already have an account)

3. Create the new library: Select Import

4. Select `index.ipynb` to view the notebook contents

5. Select the notebook of interest from the contents

6. [Run the Jupyter notebook](#Running-the-notebooks)

![Azure-steps](https://em.geosci.xyz/_images/azure-steps.png)

### Locally

To run them locally, you will need to have python installed, preferably through [anaconda](https://www.anaconda.com/download/).

You can then clone this reposiroty. From a command line, run

```
git clone https://github.com/geoscixyz/em-apps.git
```

Then `cd` into `em-apps`

```
cd heagy_2018_emcyl
```

To setup your software environment, we recommend you use the provided conda environment

```
conda env create -f environment.yml
source activate em-apps-environment
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

### Conventions

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

### Setting up your environment

- to develop code for these notebooks, please see https://github.com/geoscixyz/em_examples
- add the notebook name and path to the [index](index.ipynb)


