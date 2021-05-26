#!/usr/bin/env python
# coding: utf-8

# # SciKit K-means Clustering

# Exploring SciKit-Learn (https://scikit-learn.org/stable/) for semantic search.  Here I am looking at the k-means approach (https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py).  Specifically the Mini-Batch K-Means clustering (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html).  
# 
# There are MANY approaches (https://scikit-learn.org/stable/auto_examples/index.html#cluster-examples) and it would be nice to get some guidance on what might make a good approach for building a similarity matrix of descriptive abstracts for datasets. 
# 
# ## Notes:
# 
# Some search concepts are:
# 
# * Geolocation: "Gulf of Mexico", "Bay of Fundy", "Georges Bank"
# * Parameter: "chlorophyll", "abundance", "dissolved organic carbon"
# * Species: "calanus finmarchicus"
# * Instrument": "CTD", "bongo net"
# * Project: "C-DEBI"
# 
# The worry is these by themselves just become "frequncy searches".   What we want are search phrases that we can pull semantics from.  
# 
# ```
# !apt-get install libproj-dev proj-data proj-bin -qq
# !apt-get install libgeos-dev -qq
# ```

# ## Imports and Inits

# In[4]:


get_ipython().run_cell_magic('capture', '', '!pip install -q PyLD\n!pip install -q boto3\n!pip install -q s3fs\n!pip install -q minio\n!pip install -q rdflib==4.2.2\n!pip install -q cython\n!pip install -q cartopy\n!pip install -q SPARQLWrapper\n!pip install -q geopandas\n!pip install -q contextily==1.0rc2\n!pip install -q rdflib-jsonld==0.5.0\n!pip install -q sklearn')


# In[5]:


import requests
import json
import rdflib
import pandas as pd
from pandas.io.json import json_normalize 
import concurrent.futures
import urllib.request
import dask, boto3
from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np
import geopandas
import matplotlib.pyplot as plt
import shapely

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
 

dbsparql = "http://dbpedia.org/sparql"
okn = "http://graph.openknowledge.network/blazegraph/namespace/samplesearth/sparql"
whoi = "https://lod.bco-dmo.org/sparql"    
    
# Pandas options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# ## Gleaner Data
# 
# First lets load up some of the data Gleaner has collected.  This is just simple data graph objects and not any graphs or other processed products from Gleaner. 

# In[6]:


# Set up our S3FileSystem object
import s3fs 
oss = s3fs.S3FileSystem(
      anon=True,
      client_kwargs = {"endpoint_url":"https://oss.geodex.org"}
   )
# oss.ls('gleaner/summoned')


# In[7]:


# # A simple example of grabbing one item...  
# import json 

# jld = ""
# with oss.open('gleaner/summoned/opentopo/231f7fa996be8bd5c28b64ed42907b65cca5ee30.jsonld', 'rb') as f:
#   #print(f.read())
#    jld = f.read().decode("utf-8", "ignore").replace('\n',' ')
#    json = json.loads(jld)

# document = json['description']
# print(document)


# In[8]:


import json

@dask.delayed()
def read_a_file(fn):
    # or preferably open in text mode and json.load from the file
    with oss.open(fn, 'rb') as f:
        #return json.loads(f.read().replace('\n',' '))
        return json.loads(f.read().decode("utf-8", "ignore").replace('\n',' '))

# buckets = ['gleaner/summoned/dataucaredu', 'gleaner/summoned/getiedadataorg', 'gleaner/summoned/iris', 'gleaner/summoned/opentopo', 'gleaner/summoned/ssdb', 'gleaner/summoned/wikilinkedearth', 'gleaner/summoned/wwwbco-dmoorg', 'gleaner/summoned/wwwhydroshareorg', 'gleaner/summoned/wwwunavcoorg']

buckets = ['gleaner/summoned/opentopo']

filenames = []

for d in range(len(buckets)):
  print("indexing {}".format(buckets[d]))
  f = oss.ls(buckets[d])
  filenames += f

#filenames = oss.cat('gleaner/summoned/opentopo', recursive=True)
output = [read_a_file(f) for f in filenames]
print(len(filenames))


# In[9]:


get_ipython().run_cell_magic('time', '', '\ngldf = pd.DataFrame(columns=[\'name\', \'url\', "keywords", "description", "object"])\n\n#for key in filenames:\n\nfor doc in range(len(output)):\n#for doc in range(10):\n#for key in filenames:\n  #if ".jsonld" in key:\n  if "/.jsonld" not in filenames[doc] :\n    try:\n      jld = output[doc].compute()\n    except:\n      print(filenames[doc])\n      print("Doc has bad encoding")\n\n    # TODO  Really need to flatten and or frame this\n    try:\n      desc = jld["description"]\n    except:\n      desc = "NA"\n      continue\n    kws = "keywords" #jld["keywords"]\n    name = jld["name"]\n    url = "NA" #jld["url"]\n    object = filenames[doc]\n\n    gldf = gldf.append({\'name\':name, \'url\':url, \'keywords\':kws, \'description\': desc, \'object\': object}, ignore_index=True)')


# In[10]:


gldf.info()
# gldf.to_parquet('index.parquet.gzip',  compression='gzip') # optional save state here ...  one master parquet for Geodex? 


# ## Feature extraction
# 
# Let's just worry about the description section for now.
# 

# In[11]:


vec = TfidfVectorizer(stop_words="english")
vec.fit(gldf.description.values)
features = vec.transform(gldf.description.values)


# ### Kmeans clusters
# 
# Guess at the number a few times since we don't have a prior idea who many natural clusterings we might expect

# In[12]:


random_state = 0 

cls = MiniBatchKMeans(n_clusters=20, random_state=random_state)
cls.fit(features)


# In[13]:


# predict cluster labels for new dataset
cls.predict(features)

# to get cluster labels for the dataset used while
# training the model (used for models that does not
# support prediction on new dataset).
cls.labels_


# In[14]:


# reduce the features to 2D
pca = PCA(n_components=2, random_state=random_state)
reduced_features = pca.fit_transform(features.toarray())

# reduce the cluster centers to 2D
reduced_cluster_centers = pca.transform(cls.cluster_centers_)


# In[15]:


plt.scatter(reduced_features[:,0], reduced_features[:,1], c=cls.predict(features))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')


# ## Nearest Neighbor testing

# In[16]:


from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=10, metric='cosine')
knn.fit(features)


# In[17]:


knn.kneighbors(features[0:1], return_distance=False)


# In[18]:


knn.kneighbors(features[0:1], return_distance=True)


# ## Search testing
# 
# run a few test searches and then plot the first n (4) results

# In[19]:


input_texts = ["New Zeland lidar data", "California housing", "new madrid seismic zone"]
input_features = vec.transform(input_texts)

D, N = knn.kneighbors(input_features, n_neighbors=4, return_distance=True)

for input_text, distances, neighbors in zip(input_texts, D, N):
    print("Input text = ", input_text[:200], "\n")
    for dist, neighbor_idx in zip(distances, neighbors):
        print("Distance = ", dist, "Neighbor idx = ", neighbor_idx)
        print(gldf.name[neighbor_idx])
        print(gldf.description[neighbor_idx][:400])
        print("-"*200)
    print("="*200)
    print()

