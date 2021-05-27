# Vertical Regridding and Remapping

[![Binder](https://binder.pangeo.io/badge_logo.svg)](https://binder.pangeo.io/v2/gh/cspencerjones/vertical_regridding/master)

*C. Spencer Jones, Julius Busecke, Takaya Uchida, Ryan Abernathey*

Many ocean and climate models output ocean variables (like velocity, temperature, oxygen concentration etc.) in depth space. Property transport in the ocean generally follows isopycnals, but isopycnals often move up and down in depth space. A small difference in the vertical location of isopycnals between experiments can cause a large apparent difference in ocean properties when the experiments are compared in depth space. As a result, it is often useful to compare ocean variables in density space.

This work compares two algorithms for plotting ocean properties in density coordinates, one written in FORTRAN with a python wrapper (xlayers), and one written in python (xarrayutils). Both algorithms conserve total salt content in the vertical, and both algorithms are easily parallelizable to enable plotting large datasets in density coordinates.

We apply these algorithms to plot salinity in density space in some of the CMIP-6 models. In general, areas with net precipitation today experience increasing precipitation in higher greenhouse-gas scenarios, and areas with net evaporation today experience a further reduction in net precipitation in higher greenhouse-gas scenarios. By plotting salinity in density space, we visualize how changes in evaporation and precipitation at the surface propagate along isopycnals to influence salinity concentrations in the ocean interior.

----

A Jupyter notebook to explore ways of transforming CMIP-6 model output from depth coordinates into density coordinates. Featuring [xlayers](https://github.com/cspencerjones/xlayers) and [xarrayutils](https://github.com/jbusecke/xarrayutils), both of which run in parallel on the cloud.

Click on "launch binder" at the top to run the notebook at [https://binder.pangeo.io/](https://binder.pangeo.io/).
