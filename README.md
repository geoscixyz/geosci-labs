# em_examples

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/ubcgif/em_examples)

This is a repo of notebooks and interactive examples for http://em.geosci.xyz. The examples are based on code available in [SimPEG](http://simpeg.xyz)

## For developers
- when you generate a new notebook, please make sure that the filepath to its location follows the same structure as in [EM GeoSci](http://em.geosci.xyz)
- add the notebook name and path to theindex (index.ipynb)  
- and update the [binder](http://mybinder.org) so it can be shared with the world! 
![BinderInstructions](./images/binders.png)

## In EM GeoSci

To add the binder badge to an rst file, include:

```
.. image:: http://mybinder.org/badge.svg :target: http://mybinder.org/repo/ubcgif/em_examples
```

and if you would like to point directly to a specific example, append the path to the url, ie. 

```
.. image:: http://mybinder.org/badge.svg :target: http://mybinder.org/repo/ubcgif/em_examples/notebooks/geophysical_surveys/DCR_Pseudo-section_Simulation.ipynb
```
