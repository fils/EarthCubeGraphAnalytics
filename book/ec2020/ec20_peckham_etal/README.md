## The BALTO Jupyter Notebook GUI

*Scott Dale Peckham, Maria Stoica, D. Sarah Stamps, James Gallagher, Nathan Potter, David Fulker*

The EarthCube BALTO project has built upon the proven and widely-used technology of OPeNDAP (Open-source Project for a Network Data Access Protocol) to provide a mechanism for sharing long-tail data sets, big data sets and supporting metadata for the geosciences. The OPeNDAP protocol was developed by a non-profit organization, also called OPeNDAP, along with a software package called Hyrax. Hyrax is installed on servers to enable them to share data sets using this protocol and is one of several server-side software packages for this purpose. In support of the BALTO project, Hyrax has been extended in various ways, including support for schema.org, GeoCODES, and JSON-LD. The BALTO extension to Hyrax makes it easier for search engines to discover data sets on OPeNDAP-enabled servers.

One of the products of the BALTO project is a graphical user interface (GUI) prototype that runs in a Jupyter notebook and provides convenient access to the data that is available on OPeNDAP servers. This notebook GUI is built using several Python packages, including ipywidgets (for GUI widgets), ipyleaflet (for interactive maps), and pydap (for basic access to OPeNDAP servers). When the notebook is made available with Binder, users are able to use the GUI in a browser on their computer without installing any additional software (such as Python or Python packages). They can also edit the code in the notebook to customize it for their specific needs, or to further analyze and visualize the geoscience data sets they retrieve. A demonstration of this BALTO notebook GUI will be given at the virtual EarthCube 2020 annual meeting. 

----

# balto_gui
An Interactive GUI for BALTO in a Jupyter notebook

This respository creates a GUI (graphical user interface) for the BALTO (Brokered Alignment of Long-Tail Observations) project. BALTO is funded by the NSF EarthCube program. The GUI aims to provide a simplified and customizable method for users to access data sets of interest on servers that support the OpenDAP data access protocol. This interactive GUI runs within a Jupyter notebook and uses the Python packages: <b>ipywidgets</b> (for widget controls), <b>ipyleaflet</b> (for interactive maps) and <b>pydap</b> (an OpenDAP client).

The Python source code to create the GUI and to process events is in a module called <b>balto_gui.py</b> that must be found in the same directory as this Jupyter notebook.  Python source code for visualization of downloaded data is given in a module called <b>balto_plot.py</b>.

This GUI consists of mulitiple panels, and supports both a <b>tab-style</b> and an <b>accordion-style</b>, which allows you to switch between GUI panels without scrolling in the notebook.

You can run the notebook in a browser window without installing anything on your computer, using something called Binder. Look for the Binder icon below and a link labeled "Launch Binder".  This sets up a server in the cloud that has all the required dependencies and lets you run the notebook on that server.  (Sometimes this takes a while, however.)

To run this Jupyter notebook without Binder, it is recommended to install Python 3.7 from an Anaconda distribution and to then create a conda environment called <b>balto</b>. Simple instructions for how to create a conda environment and install the software are given in Appendix 1 of version 2 (v2) of the notebook.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/peckhams/balto_gui/master?filepath=BALTO_GUI_v2.ipynb)
<br>


