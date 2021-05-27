#!/usr/bin/env python
# coding: utf-8

# # Semantic Annotation of Data using JSON Linked Data
# 
# Luigi Marini (lmarini@illinois.edu), Diego Calderon (diegoac2@illinois.edu), Praveen Kumar (kumar1@illinois.edu)
# 
# University of Illinois at Urbana-Champaign

# ## Abstract
# 
# The Earthcube Geosemantics Framework (https://ecgs.ncsa.illinois.edu/) developed a prototype of a decentralized framework that combines the Linked Data and RESTful web services to annotate, connect, integrate, and reason about integration of geoscience resources. The framework allows the semantic enrichment of web resources and semantic mediation among heterogeneous geoscience resources, such as models and data. 
# 
# This notebook provides examples on how the Semantic Annotation Service can be used to manage linked controlled vocabularies using JSON Linked Data (JSON-LD), including how to query the built-in RDF graphs for existing linked standard vocabularies based on the Community Surface Dynamics Modeling System (CSDMS), Observations Data Model (ODM2) and Unidata udunits2 vocabularies, how to query build-in crosswalks between CSDMS and ODM2 vocabularies using SKOS, and how to add new linked vocabularies to the service. JSON-LD based definitions provided by these endpoints will be used to annotate sample data available within the IML Critical Zone Observatory data repository using the Clowder Web Service API (https://data.imlczo.org/). By supporting JSON-LD, the Semantic Annotation Service and the Clowder framework provide examples on how portable and semantically defined metadata can be used to better annotate data across repositories and services.

# ## Table of contents
# 1. [Introduction](#Introduction)
# 2. [Basic requirements](#Basic-Requirements)
# 3. [Geosemantics Integration Service](#Geosemantics-Integration-Service-(GSIS))
# 4. [RDF Graphs](#RDF-Graphs)
# 5. [Standard Vocabularies](#Standard-Vocabularies)
# 6. [List of CSDMS Standard Names and ODM2 as JSON Arrays](#List-of-CSDMS-Standard-Names-and-ODM2-as-JSON-Arrays)
# 7. [Crosswalks Between Standard Vocabularies](#Crosswalks-Between-Standard-Vocabularies)
# 8. [Units](#Units)
# 9. [Temporal Annotation](#Temporal-Annotation)
# 10. [Annotating Data in the IMLCZO Data Management System](#Annotating-Data-in-the-IMLCZO-Data-Management-System)
# 11. [Setting API key and request headers](#Setting-API-key-and-request-headers)
# 12. [Search by generic search query](#Search-by-generic-search-query)
# 13. [Search by metadata field](#Search-by-metadata-field)
# 14. [Create new dataset](#Create-new-dataset)
# 15. [Upload file to new dataset](#Upload-file-to-new-dataset)
# 16. [Add metadata to new file](#Add-metadata-to-new-file)
# 17. [Matching Models with Data](#Matching-Models-with-Data)
# 18. [Conclusions](#Conclusions)
# 19. [References](#References)
# 20. [License](#License)

# ## Introduction
# 
# We face many challenges in the process of extracting meaningful information from data. Frequently, these obstacle  compel scientists to perform the integration of models with data manually. Manual integration becomes exponentially difficult when a user aims to integrate long-tail data (data collected by individual researchers or small research groups) and long-tail models (models developed by individuals or small modeling communities). We focus on these long-tail resources because despite their often-narrow scope, they have significant impacts in scientific studies and present an opportunity for addressing critical gaps through automated integration. The goal of the Goesemantics Framework is to provide a framework rooted in semantic techniques and approaches to support “long-tail” models and data integration.
# 
# The Linked Data paradigm emerged in the context of Semantic Web technologies for publishing and sharing data over the Web. It connects related individual Web resources in a Graph database, where resources represent the graph nodes, and an edge connects a pair of nodes. Publishing and linking scientific resources using Semantic Web technologies require that the user community follows the three principles of Linked Data:
# 
# 1.  Each resource needs to be represented using a unique Uniform Resource Identifier (URI), which consists of: (i) A Uniform Resource Locator (URL) to define the server path over the Web, and (ii) A Uniform Resource Name (URN) to describe the exact name of the resource.
# 
# 2. The relationships between resources are described using the triple format, where a subject S has a predicate P with an object O. A predicate is either an undirected relationship (bi-directional), where it connects two entities in both ways or a directed relationship (uni-directional), where the presence of a relationship between two entities in one direction does not imply the presence of a reverse relationship. The triple format is the structure unit for the Linked Data system. 
# 
# 3. The HyperText Transfer Protocol (HTTP) is used as a universal access mechanism for resources on the Web. 
# 
# For more information about linked data, please visit https://www.w3.org/standards/semanticweb/data.
# 
# The Geosemantics Integration Service (GSIS) is a playground to show a lot of these principles in practice with respect to Earth science. Below we show many of the endpoints available in the GSIS and how they can enable new ways to manage metadata about data and models. By virtue of leveraging Semntic Web Technologies, the approaches below are compatible with other efforts such as the [ESIP science-on-schema](https://github.com/ESIPFed/science-on-schema.org) effort. Some of the endpoints shown below are currently used in production, others are proof of concept developement. The goal of this notebook is to show what is possible when using Linked Data approaches.

# ## Basic Requirements
# 
# We will be interacting with two web services. The first is the Geosemantics Integration Service (GSIS) available at [http://hcgs.ncsa.illinois.edu](http://hcgs.ncsa.illinois.edu). This service provides support for standandard vocabularies and methods for transforming typical strings used for tracking time, space and physical variables into well formed Linked Data documents. The second service is the [NSF Intensively Managed Landscape Critical Zone Observatory](http://criticalzone.org/iml/) data management system (https://data.imlczo.org/). We will be retrieving data from it and uploading data and metadata back to it using the Clowder web service API. Clowder is a customizable and scalable data management system to support any data format (Marini et al. 2018). It is under active development and deployed for a variety of research projects. For more information about Clowder please visit https://clowderframework.org/. 
# 
# We first setup some basic requirements used throughout the notebook. We use the ubiqutous [Requests](https://requests.readthedocs.io/) Python library to intereact with both APIs.

# In[1]:


import requests
import json

gsis_host = 'https://ecgs.ncsa.illinois.edu'

clowder_host = 'https://data.imlczo.org/clowder'


# ## Geosemantics Integration Service (GSIS)
# 
# Since geospatial data comes in many formats, from shapefiles to geotiffs to comma-delimited text files, it is often helpfull to annotate the files with portable metadata that can be used to identify what each files contains. For geospatial data the temporal, spatial, and physical properties dimensions are important and often used to search over a large collection of data. The Geosemantics Integration Service (GSIS) provides a series of endpoints to simplify annotating geospatial data. It includes temporal endpoints so that generic formats for date and times can be translated to well formted JSON-LD formats (Elag et al. 2015). It includes the ability to store standard vocabularies as generic RDF graphs and retrieve those as simple JSON documents for easy integration in external services (for example Clowder). It also includes the ability to make links between terms from two different standard vocabularies using SKOS and OWN same as predicates.

# ### RDF Graphs
# 
# All information stored in the GSIS is stored in the form of RDF graphs using [Apache Jena](https://jena.apache.org/). The following endpoints list all know RDF graphs and let the client retrieve each graph as JSON-LD. The content of these graphs can vary greatly, from standardad vocabularies to definitions of computational models. For a full list of methods please see the documentation available at http://hcgs.ncsa.illinois.edu/. New RDF graphs can be added to the system using private HTTP POST endpoints which accept RDF graphs serialized as JSON-LD or Turtle format.

# In[2]:


# Get a lists of the names of all graphs in the Knowledge base
r = requests.get(f"{gsis_host}/gsis/listGraphNames")
r.json()


# In[3]:


# List the content of the  Graph (for example, CSDMS Standard Names)
graph = 'csdms'
r = requests.get(f"{gsis_host}/gsis/read?graph={graph}")
r.json().get('@graph')[0:5] # We just show the top 10 results, to see all results remove the slice operator [0:5]


# ### Standard Vocabularies
# Two RDF graphs in the GSIS store two external standard vocabularies in RDF. The first one is the [CSDMS Standard Terms (CSN)](https://csdms.colorado.edu/wiki/CSDMS_Standard_Names) (Peckham et al. 2014). The second is the [ODM2 Variable Name Vocabulary](http://vocabulary.odm2.org/variablename/) (Horsburgh et al. 2016). To make it easier to query these RDF by graphs, the GSIS provide simplified methods to search across standard vocabularies by label and look attributes of a specific term in a vocabulary.

# In[4]:


# Search standard vocabularies by search query
query = 'wind speed'
r = requests.get(f'{gsis_host}/gsis/sas/vars/list?term={query}')
r.json()


# In[5]:


# Get all properties of a given CSDMS Standard Name from a specific graph
graph = 'csdms'
name = 'air__dynamic_shear_viscosity'
r = requests.get(f"{gsis_host}/gsis/CSNqueryName?graph={graph}&name={name}")
r.json()


# ### List of CSDMS Standard Names and ODM2 as JSON Arrays
# 
# To simplify clients' ability to parse these standard vocabularies, the GSIS provides ways to list all unique identifiers from both vocabularies as play JSON arrays. This for example is used by Clowder to show a list of standard term for each vocabulary in its interface. It is worth noting that Clowder lets users define these lists through its GUI both as local list, but more importantly as remote JSON endpoints so that, as lists are updated, the latest version is always shown to the user. Here is an example from the IMLCZO instance.
# 
# This is what a user sees when manually adding metadata to a file or dataset. The list is dynamically loaded at run time.
# 
# ![Add metadata](img/metadata-add.png)
# 
# This is what an adminstrator of the system or a data sharing space see when adding more options to what users can defined from the GUI. This is only for metadata added by users from the GUI. Later we will show how to programmatically add any type of metadata to a file or dataset using the web service API.
# 
# ![Define metadata](img/metadata-define.png)
# 
# 

# The widget listing options above is populated by calling the endpoints below. The `Definitions URL` is how the system is aware of which external endpoint to call. Any service providing the same interface can be utilized.

# In[6]:


# Get the CSDMS Standard Names as a flat list.
r = requests.get(f"{gsis_host}/gsis/sas/sn/csn")
csn_terms = r.json()
print(f'Found {len(csn_terms)} terms. Showing top 20 results:')
csn_terms[0:20] # We just show the top 20 results, to see all results remove the slice operator [0:20]


# In[7]:


# Get the ODM2 Variable Names as a flat list.
r = requests.get(f"{gsis_host}/gsis/sas/sn/odm2")
odm2_terms = r.json()
print(f'Found {len(odm2_terms)} terms. Showing top 20 results:')
r.json()[0:20] # We just show the top 20 results, to see all results remove the slice operator [0:20]


# With some simple Python we can search specific substrings from these lists.

# In[8]:


[i for i in csn_terms if 'temperature' in i]


# ### Crosswalks Between Standard Vocabularies
# 
# With so many standard vocabularies available, it is helpful to defined equivalency between terms from separate vocabularies. To this end, the GSIS provides the ability to establish mappings between terms in different graphs using the `skos:sameAs` predicate. The user can the query this graph of relationships using the following endpoint.

# In[9]:


# Given a term from one vocabulary, find equivalent ones in other vocabularies.
var = 'http://vocabulary.odm2.org/variablename/windSpeed'
r = requests.get(f'{gsis_host}/gsis/var/sameAs/skos?varName={var}')
r.json()


# ### Units
# 
# Physical variables are not the only type of standard vocabularies the GSIS stores. Following are examples of two different lists of standard units imported in the GSIS, [Unidata udunits2](https://www.unidata.ucar.edu/software/udunits/) and [Google Units](https://support.google.com/websearch/answer/3284611?hl=en).

# In[10]:


# Get the list of udunits2 units in JSON format.
r = requests.get(f"{gsis_host}/gsis/sas/unit/udunits2")
r.json()[0:20] # We just show the top 20 results, to see all results remove the slice operator [0:20]


# In[11]:


## Get the list of Google units in JSON format.
r = requests.get(gsis_host + "/gsis/sas/unit/google")
r.json()[0:20] # We just show the top 20 results, to see all results remove the slice operator [0:20]


# ### Temporal Annotation
# 
# To convert from strings representing time to a more formal definition, the GSIS provides three endpoints to represent instant, interval, and time series. Time values are represented in UTC (Coordinated Universal Time) format. Times are expressed in local time, together with a time zone offset in hours and minutes. For more information about date and time formats, please visit https://www.w3.org/TR/NOTE-datetime.

# #### Time Instant Annotation
# Query parameters: 
# * **time** (string): time value in UTC format

# In[12]:


# Get a temporal annotation for a time instant in a JSON-LD format.
time = '2014-01-01T08:01:01-09:00'
r = requests.get(f"{gsis_host}/gsis/sas/temporal?time={time}")
r.json()


# #### Time Interval Annotation
# Query parameters: 
# * **beginning** (string): time value in UTC format.
# * **end** (string): time value in UTC format.

# In[13]:


# Get a temporal annotation for a time interval in a JSON-LD format.
beginning = '2014-01-01T08:01:01-10:00'
end = '2014-12-31T08:01:01-10:00'
r = requests.get(f"{gsis_host}/gsis/sas/temporal?beginning={beginning}&end={end}")
r.json()


# #### Time Series Annotation
# Query parameters:
# * **beginning** (string): time value in UTC format.
# * **end** (string): time value in UTC format.
# * **interval** (float): time step.

# In[14]:


# Get a temporal annotation for a time series in a JSON-LD format.
beginning = '2014-01-01T08:01:01-10:00'
end = '2014-03-01T08:01:01-10:00'
interval = '4'
r = requests.get(f"{gsis_host}/gsis/sas/temporal?beginning={beginning}&end={end}&interval={interval}")
r.json()


# ## Annotating Data in the IMLCZO Data Management System

# The [IMLCZO data management system](https://data.imlczo.org/) is comprised of two different services. A Clowder instance stores raw data streaming in from sensors, manually uploaded by users, collected in the lab or in the field. A Geodashboard including a subset of all the data collected and presented using interactive maps and graphs. We will focus on the Clowder service and how users can store arbitrary metadata as JSON-LD on datasets and files stored within. We will leverage the GSIS to make sure that our metadata is based on existing standards.
# 
# Because we will be adding information to the Clowder instance, we will be required to register an account on the IMLCZO Clowder instance and create an API key and added to the cell below.

# ### Register an account on IMLCZO
# To register an account on the IMLCZO Clowder instance, please go to https://data.imlczo.org/clowder/signup and enter your email. You will receive an email from us. To activate your account, please reply back to the email saying you are using the ecgs jupyter notebook.
# Once your account is activated, you can generate an API key.

# ### Generate and use an API Key
# 1. Login to https://data.imlczo.org/clowder/login and navigate to your profile (click on the icon in the top right and select "View Profile").
# ![profile](img/profile.png)
# 2. Add a name to your key and hit the "Add" button. In the example below, we created a test_key.
# ![profile](img/added_key.png)
# 3. Copy your key from step 2. 
# 4. If you are running this notebook in your machine, create a .env file in the same directory as this notebook and, using your favorite text editor, add the following line:
# ```CLOWDER-KEY=paste-your-key-here```.
# 5. If you are running the notebook in Binder, uncomment and paste the key in line 5, `%env CLOWDER-KEY=` in the block below. 
# 6. Run the following block.
# 

# ### Setting API key and request headers
# 
# We use [python-dotenv](https://github.com/theskumar/python-dotenv) to set the Clowder API key for this session (you can also just manually set it in the notebook if you prefer). We also set the headers for most requests here and set the default content type to JSON and the Clowder API key to the one we just created. All calls will only provide information based on the user making the request. This means that the quality of the results could vary greatly. We will be creating a dataset and adding a file to it to make sure that the user can make the appropriate against this specific resource.

# In[15]:


# Please create an API key as described above
get_ipython().run_line_magic('load_ext', 'dotenv')
get_ipython().run_line_magic('dotenv', '')
# If using Binder, uncomment and paste your key below
# %env CLOWDER-KEY=paste your key here
import os
clowder_key = os.getenv("CLOWDER-KEY")

headers = {'Content-type': 'application/json', 'X-API-Key': clowder_key}


# ### Search by generic search query

# We will start by searching the system for a generic string `precipitation`. Depending on your permissions you might be able to see around 11 results. The results of this query are based on any information available on the resource (dataset, file, or collection).

# In[16]:


query = 'precipitation'
r = requests.get("{}/api/search?query={}".format(clowder_host, query), headers=headers)
r.raise_for_status()
r.json()


# ### Search by metadata field
# 
# To be more specific, we will search for any resource which contains metadata for `ODM2 Variable Name` that is equal to `precipitation`.

# In[17]:


query = '"ODM2 Variable Name":"precipitation"'
r = requests.get("{}/api/search?query={}".format(clowder_host, query), headers=headers)
datasets = r.json().get('results')

dataset = [d for d in datasets if d.get('name') == 'Trimpe East Site (Precip tipping bucket site)']

datasetId = dataset[0].get('id')

dataset[0]


# We list all files in the dataset and download the first in the list. This is just to provide us with a relevant file locally but if you prefer you can ignore this step and later on upload your own file to the system.

# In[18]:


# List files in dataset
url = "{}/api/datasets/{}/files".format(clowder_host, datasetId)
r = requests.get(url)
files = r.json()
# Download the first file
file_id = files[0].get('id')
file_name = files[0].get('filename')
url = "{}/api/files/{}/blob".format(clowder_host, file_id)
r = requests.get(url)
with open(file_name, 'wb') as f:
    f.write(r.content)
print(f'Downloaded file {file_name} to local disk')


# ### Create new dataset
# 
# Create a new dataset to contain the file we just downloaded (or a new one) and metadata.

# In[19]:


url = "{}/api/datasets/createempty".format(clowder_host)
payload = json.dumps({'name': 'Geosemantics Demo', 
                      'description': 'A dataset used for demoing basic metadata annotation functionality',
                      'access': 'PRIVATE',
                      'space': [],
                      'collection': []}) 

r = requests.post(url, data=payload, headers=headers)
r.raise_for_status()
new_dataset = r.json()
new_dataset_id = new_dataset.get('id')
print(f'Created new dataset {clowder_host}/datasets/{new_dataset_id}')


# ### Upload file to new dataset
# 
# We now upload a file to the dataset that we just created. If you prefer uploading a different file, change the file name below.

# In[20]:


url = "{}/api/uploadToDataset/{}".format(clowder_host, new_dataset_id)
# change file_name if you prefer uploading a different file from your local directory
files = {'file': open(file_name, 'rb')}
r = requests.post(url, files=files, headers={'X-API-Key': clowder_key})
r.raise_for_status()
uploaded_file_id = r.json().get('id')
print(f'Uploaded file {clowder_host}/files/{uploaded_file_id}')


# ### Add metadata to new file
# 
# We now upload metadata to the file we have just uploaded. Note that this operation can be executed multiple times with different payloads. Every time a new entry is added to the list of metadata documents associated with a file or dataset. The same user can update multiple values of a specific entry or different users can specify alternative values of the same metadata types. It is is up to the client to decide which version is the most accurate. Users can delete entries that are not valid anymore. This type of generic metadata is compatible to the [advanced publishing techniquies described in the ESIP science-on-schema.org](https://github.com/ESIPFed/science-on-schema.org/blob/master/guides/Dataset.md#advanced-publishing-techniques) and could be added to a DCAT Dataset as described there.

# In[21]:


url = "{}/api/files/{}/metadata.jsonld".format(clowder_host, uploaded_file_id)
payload = {
    "@context":[
        "https://clowder.ncsa.illinois.edu/contexts/metadata.jsonld",
        {
            "CSDMS Standard Name": "http://csdms.colorado.edu/wiki/CSN_Searchable_List#atmosphere_air__temperature",
            "Unit": "http://ecgs.ncsa.illinois.edu/gsis/sas/unit/udunits2#degree_Celsius"
        }
    ],
    "agent": {
        "@type":"cat:extractor",
        "name":"ECGS Notebook",
        "extractor_id":"https://clowder.ncsa.illinois.edu/api/extractors/ecgs"
    },
    "content": {
        "CSDMS Standard Name": "atmosphere_air__temperature",
        "Unit": "degree_Celsius"
    }
}
r = requests.post(url, headers = headers, data=json.dumps(payload))
r.raise_for_status()
print('Response ' + r.json())
print(f'View metadata you have just uploaded on the file page {clowder_host}/files/{uploaded_file_id}')


# We can also retrieve all the metadata available on the file, including metadata automatically created by the system.

# In[22]:


url = "{}/api/files/{}/metadata.jsonld".format(clowder_host, uploaded_file_id)
r = requests.get(url, headers = headers)
r.json()


# We have defined the context by including an external one that contains the basic elements of any Clowder metadata document such as `agent` as well as specific ones for the two entries in `content`. We can view the rest of the context here:

# In[23]:


# The context file describes the basic elements of a Clowder metadata document
r = requests.get('https://clowder.ncsa.illinois.edu/contexts/metadata.jsonld')
r.json()


# ### Matching Models with Data
# 
# The GSIS also prototyped a method to match simulations models developed using the Basic Model Interface (BMI) (Jiang et al. 2017) to input datasets. Given definitions for both, the system checks if a specific variable is compatible between data and model. The user provides the id of an RDF graph representing a model, one of an RDF graph representing a dataset, and the RDF predicates used to identify the variables in each graph. It then tries to match the two lists leveraging the `skos:sameAs` crosswalks defined above. The results includes whether a match was found, what variables are missing if so, and what variables will require a crosswalk and potentially a conversion.
# 
# For example, in the case below a match was found but the crosswalk between `http://vocabulary.odm2.org/variablename/windSpeed` and `http://csdms.colorado.edu/wiki/CSN_Searchable_List/land_surface_air_flow__speed` was required.

# In[24]:


model_id = 'model-3'
model_var_property_name = 'http://ecgs.ncsa.illinois.edu/bmi_models/temp/hasStandardName'
data_id = 'data-3'
data_var_property_name = 'http://ecgs.ncsa.illinois.edu/gsis/sas/vars'
        
r = requests.get(f'{gsis_host}/gsis/modelDataMatches?model={model_id}&modelVarPropertyName={model_var_property_name}&data={data_id}&dataVarPropertyName={data_var_property_name}')
r.json()


# For reference here are the two RDF graphs being compared:

# In[25]:


r = requests.get(f"{gsis_host}/gsis/read?graph={model_id}")
r.json()


# In[26]:


r = requests.get(f"{gsis_host}/gsis/read?graph={data_id}")
r.json()


# ## Conclusions
# 
# This short notebook provides a few simple examples of developing web applications around the principles of Linked Data. By developing services built around interoperability, we hope it will be easier in the future to build services that can easily interoperate. Earth sciences provide unique challenges in that the way researchers store their data can vary greatly. The Linked Data approach can be useful in overcoming some of these challenges even thought it provides its own technical challenges in terms of adoption. Over time efforts like [Schema.org](https://schema.org/) are showing that the principles of Linked Data are important but that simplifying some of their approaches might be the key to widespread adoption. Even though the GSIS stores information as RDF graphs, it provides simple HTTP web services to make it easier to be used in the existing ecosystem of tools. Furthermore, the Clowder data framework provides simple GUI and APIs to store rich metadata documents next to the raw bytes, but it tries to find a good compromise between expressivenes of the metadata and simplicity of use.

# ## References
# 
# 1. Marini, L., I. Gutierrez-Polo, R. Kooper, S. Puthanveetil Satheesan, M. Burnette, J. Lee, T. Nicholson, Y. Zhao, and K. McHenry. 2018. Clowder: Open Source Data Management for Long Tail Data. In Proceedings of the Practice and Experience on Advanced Research Computing (PEARC '18). ACM, New York, NY, USA, Article 40, 8 pages. DOI: https://doi.org/10.1145/3219104.3219159
# 2. Jiang, Peishi and Elag, Mostaf and Kumar, Praveen and Peckham, Scott and Marini, Luigi and Liu, Rui, “A service-oriented architecture for coupling web service models using the Basic Model Interface (BMI)”, Environmental Modelling & Software, 2017
# 3. Elag, M.M., P. Kumar, L. Marini, S.D. Peckham (2015) Semantic interoperability of long-tail geoscience resources over the Web, In: Large-Scale Machine Learning in the Earth Sciences, Eds. A.N. Srivastava, R. Nemani and K. Steinhaeuser, Taylor and Francis
# 4. Peckham, S.D. (2014a) The CSDMS Standard Names: Cross-domain naming conventions for describing process models, data sets and their associated variables, Proceedings of the 7th Intl. Congress on Env. Modelling and Software, International Environmental Modelling and Software Society (iEMSs), San Diego, CA. (Eds. D.P. Ames, N.W.T. Quinn, A.E. Rizzoli)
# 5. Horsburgh, J. S., Aufdenkampe, A. K., Mayorga, E., Lehnert, K. A., Hsu, L., Song, L., Spackman Jones, A., Damiano, S. G., Tarboton, D. G., Valentine, D., Zaslavsky, I., Whitenack, T. (2016). Observations Data Model 2: A community information model for spatially discrete Earth observations, Environmental Modelling & Software, 79, 55-74, http://dx.doi.org/10.1016/j.envsoft.2016.01.010

# ## License
# 
# This notebook is licensed under the [Apache License](http://www.apache.org/licenses/LICENSE-2.0).

# 
