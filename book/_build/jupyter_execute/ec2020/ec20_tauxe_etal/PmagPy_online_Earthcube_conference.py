#!/usr/bin/env python
# coding: utf-8

# ## PmagPy Online: Jupyter Notebooks, the PmagPy Software Package and the Magnetics Information Consortium (MagIC) Database
# 
# Lisa Tauxe$^1$, Rupert Minnett$^2$, Nick Jarboe$^1$, Catherine Constable$^1$, Anthony Koppers$^2$, Lori Jonestrask$^1$, Nick Swanson-Hysell$^3$
# 
# $^1$Scripps Institution of Oceanography, United States of America;  $^2$   Oregon State University; $^3$ University of California, Berkely; ltauxe@ucsd.edu
# 
# The Magnetics Information Consortium (MagIC), hosted at http://earthref.org/MagIC is a database that serves as a Findable, Accessible, Interoperable, Reusable (FAIR) archive for paleomagnetic and rock magnetic data. It has a flexible, comprehensive data model that can accomodate most kinds of paleomagnetic data. The **PmagPy** software package is a cross-platform and open-source set of tools written in Python for the analysis of paleomagnetic data that serves as one interface to MagIC, accommodating various levels of user expertise. It is available through github.com/PmagPy. Because PmagPy requires installation of Python, several non-standard Python modules, and the PmagPy software package, there is a speed bump for many practitioners on beginning to use the software. In order to make the software and MagIC more accessible to the broad spectrum of scientists interested in paleo and rock magnetism, we have prepared a set of Jupyter notebooks, hosted on [jupyterhub.earthref.org](https://jupyterhub.earthref.org) which serve a set of purposes. 1) There is a complete course in Python for Earth Scientists, 2) a set of notebooks that introduce PmagPy (pulling the software package from the github repository) and illustrate how it can be used to create data products and figures for typical papers, and 3) show how to prepare data from the laboratory to upload into the MagIC database. The latter will satisfy expectations from NSF for data archiving and for example the AGU publication data archiving requirements.
# 
# 
# 

# ### Getting started
# 
# - To use the PmagPy notebooks online, go to  website at [https://jupyterhub.earthref.org/](https://jupyterhub.earthref.org/). Create an Earthref account using your ORCID and log on. \[This allows you to keep files in a private work space.\]
# - Open the PmagPy Online - Setup  notebook and execute the two cells.  Then click on File = > Open and click on the PmagPy_Online folder.  Open the PmagPy_online notebook and work through the examples.  There are other notebooks that are useful for the working paleomagnetist.
# - Alternatively, you can  install Python and the  PmagPy software package on your computer (see [https://earthref.org/PmagPy/cookbook](https://earthref.org/PmagPy/cookbook) for instructions).   Follow  the instructions for  "Full PmagPy install and update" through section 1.4 (Quickstart with PmagPy notebooks).  This notebook is in  the collection of PmagPy notebooks. 
# 

# ### Overview of   MagIC
# 
#  
# The Magnetics Information Consortium (MagIC), hosted at http://earthref.org/MagIC is a database that serves as a Findable, Accessible, Interoperable, Reusable (FAIR) archive for paleomagnetic and rock magnetic data. Its datamodel is fully described here: [https://www2.earthref.org/MagIC/data-models/3.0](https://www2.earthref.org/MagIC/data-models/3.0). Each contribution is associated with a publication via the DOI.  There are nine data tables:
# 
# - contribution: metadata of the associated publication.
# - locations: metadata for locations, which are groups of sites (e.g., stratigraphic section, region, etc.)
# - sites: metadata and derived data at the site level (units with a common expectation)
# - samples: metadata and derived data at the sample level.
# - specimens: metadata and derived data at the specimen level.
# - criteria: criteria by which data are deemed acceptable
# - ages: ages and metadata for sites/samples/specimens
# - images: associated images and plots.  
# 

# ### Overview of   PmagPy
# 
# The functionality of **PmagPy** is demonstrated within notebooks in the **PmagPy** repository:
# 
# - PmagPy_online.ipynb:  serves as an introdution to PmagPy and MagIC (this conference). It highlights the link between **PmagPy** and the Findable Accessible Interoperable Reusabe (FAIR) database maintained by the Magnetics Information Consortium (MagIC) at [https://earthref.org/MagIC](https://eathref.org/MagIC).  
# 
# Other notebooks of interest are: 
# 
# - PmagPy_calculations.ipynb:  demonstrates many of the PmagPy calculation functions such as those that rotate directions, return statistical parameters, and simulate data from specified distributions. 
# - PmagPy_plots_analysis.ipynb: demonstrates PmagPy functions that can be used to visual data as well as those that conduct statistical tests that have associated visualizations.
# - PmagPy_MagIC.ipynb: demonstrates how PmagPy can be used to read and write data to and from the MagIC database format including conversion from many individual lab measurement file formats.
# 
# Please see also our YouTube channel with more presentations from the 2020 MagIC workshop here: 
# [https://www.youtube.com/playlist?list=PLirL2unikKCgUkHQ3m8nT29tMCJNBj4kj](https://www.youtube.com/playlist?list=PLirL2unikKCgUkHQ3m8nT29tMCJNBj4kj)
# 
# 

# In[ ]:




