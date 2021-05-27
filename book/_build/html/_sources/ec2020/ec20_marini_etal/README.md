# Semantic Annotation of Data using JSON Linked Data

*Luigi Marini, Diego Calderon, Praveen Kumar*

The Earthcube Geosemantics Framework (http://ecgs.ncsa.illinois.edu/) developed a prototype of a decentralized framework that combines the Linked Data and RESTful web services to annotate, connect, integrate, and reason about integration of geoscience resources. The framework allows the semantic enrichment of web resources and semantic mediation among heterogeneous geoscience resources, such as models and data.

This notebook will provide examples on how the Semantic Annotation Service can be used to manage linked controlled vocabularies using JSON Linked Data (JSON-LD), including how to query the built-in RDF graphs for existing linked standard vocabularies based on the Community Surface Dynamics Modeling System (CSDMS), Observations Data Model (ODM2) and Unidata udunits2 vocabularies, how to query build-in crosswalks between CSDMS and ODM2 vocabularies using SKOS, and how to add new linked vocabularies to the service. JSON-LD based definitions provided by these endpoints will be used to annotate sample data available within the IML Critical Zone Observatory data repository using the Clowder Web Service API (https://data.imlczo.org/). By supporting JSON-LD, the Semantic Annotation Service and the Clowder framework provide examples on how portable and semantically defined metadata can be used to better annotate data across repositories and services.

----

A simple Jupyter Notebook to explore the Web service API available through https://ecgs.ncsa.illinois.edu/ and https://clowderframework.org/.

There are two options to run this notebook:

1. Launch the notebook on https://mybinder.org by clicking on this link [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/earthcube2020/ec20_marini_etal.git/master?filepath=ecgs.ipynb)

2. Follow these steps to run the notebook on your local machine:

- Install python 3 on your local machine (you can try anaconda for managing your environments)
- Create a virtual environment and install the `requirements.txt` (using your terminal, navigate to the location of the notebook and run the command `pip install -r requirements.txt`
- Install Jupyter Notebook in the same virtual environment from the previous step
- Run the notebook `ecgs.ipynb` using `jupyter-lab` or `jupyter-notebook`.

A PDF copy of the notebook is available [here](ecgs.pdf).
