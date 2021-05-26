#!/usr/bin/env python
# coding: utf-8

# # EarthCube Graph Analytics Exploration
# 
# ## About
# 
# This is the start of learning a bit about leveraging graph analytics to assess the 
# EarthCube graph and explore both the relationships but also look for methods to 
# better search the graph for relevant connections.
# 
# ## Thoughts
# It seems we don't care about the triples with literal objects.  Only the triples 
# that represent connections between types.   Could we use a CONSTRUCT call to 
# remove the unwanted triples?  Filter on only IRI to IRI.    
# 
# ## References
# 
# * [RDFLib](https://github.com/RDFLib/rdflib)
# * [NetworkX](https://networkx.org/)
# * [iGraph](https://igraph.org/)
# * [NetworkX link analysis](https://networkx.org/documentation/latest/reference/algorithms/link_analysis.html?highlight=page%20rank#)
# * https://faculty.math.illinois.edu/~riveraq2/teaching/simcamp16/PageRankwithPython.html
# * https://docs.dask.org/en/latest/
# * https://examples.dask.org/bag.html
# * https://s3fs.readthedocs.io/en/latest/
# * https://docs.dask.org/en/latest/remote-data-services.html

# ## Installs

# In[14]:


get_ipython().system('pip -q install mimesis')
get_ipython().system('pip -q install minio ')
get_ipython().system('pip -q install s3fs')
get_ipython().system('pip -q install SPARQLWrapper')
get_ipython().system('pip -q install boto3')
get_ipython().system("pip -q install 'fsspec>=0.3.3'")
get_ipython().system('pip -q install rdflib')
get_ipython().system('pip -q install rdflib-jsonld')
get_ipython().system('pip -q install PyLD==2.0.2')
get_ipython().system('pip -q install networkx')


# In[2]:


import sys
sys.path.append("../lib/")  # path contains python_file.py

import sparqlPandas


# ## Imports
# 

# In[3]:


import dask, boto3
import dask.dataframe as dd
import pandas as pd
import json

from SPARQLWrapper import SPARQLWrapper, JSON

sweet = "http://cor.esipfed.org/sparql"
dbsparql = "http://dbpedia.org/sparql"
ufokn = "http://graph.ufokn.org/blazegraph/namespace/ufokn-dev/sparql"


# ## Code inits

# ### Helper function(s)
# The following block is a SPARQL to Pandas feature.  You may need to run it to load the function per standard notebook actions.

# In[40]:


#@title
def get_sparql_dataframe(service, query):
    """
    Helper function to convert SPARQL results into a Pandas data frame.
    """
    sparql = SPARQLWrapper(service)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    result = sparql.query()

    processed_results = json.load(result.response)
    cols = processed_results['head']['vars']

    out = []
    for row in processed_results['results']['bindings']:
        item = []
        for c in cols:
            item.append(row.get(c, {}).get('value'))
        out.append(item)

    return pd.DataFrame(out, columns=cols)


# ### Set up some Pandas Dataframe options

# In[5]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# ### Set up the connection to the object store to access the graph objects from

# In[30]:


import s3fs 

oss = s3fs.S3FileSystem(
      anon=True,
      key="",
      secret="",
      client_kwargs = {"endpoint_url":"https://oss.geodex.org"}
   )


# In[31]:


# Simple command to list objects in the current results bucket prefix
oss.ls('gleaner/results/cdfv3')


# ## Pull a graph and load
# 
# Let's pull back an example graph and load it up into an RDFLib graph so we can test out a SPARQL call on it.

# In[44]:


import rdflib
import gzip

with oss.open('gleaner/results/cdfv3/opentopo_graph.nq', 'rb') as f:
  #print(f.read())
  file_content = f.read()  #.decode("utf-8", "ignore").replace('\n',' ')

# print(file_content)    
# with gzip.open('./oceanexperts_graph.nq.gz', 'rb') as f:
#     file_content = f.read()


# In[45]:


g = rdflib.Graph()
parsed = g.parse(data = file_content, format="nquads")


# ## Note
# 
# When we start to do the network analysis we don't really care about the links to literal strings.
# Rather, we want to see connections between various types.  More specidically, types connecting to types.
# 
# Note, the isIRI filter removes the blank nodes since the rdf:type can point to both IRI and blank nodes.
# 
#            BIND("Nca34627a4b6d4272be7e2d22bab3becd" as ?s)
# 
# 
# ```SPARQL
#     prefix schema: <http://schema.org/>
#     SELECT DISTINCT ?s ?o 
#        WHERE {
#            ?s  a schema:Dataset. 
#            ?s ?p ?o .
#            ?o a ?type
#            FILTER isIRI(?o)
#        }
#      LIMIT 1000
# ```

# In[46]:


qres = g.query(
    """prefix schema: <https://schema.org/>
    SELECT DISTINCT ?s ?o 
       WHERE {
           ?s  a schema:Dataset. 
           ?s ?p ?o .
           ?o a ?type  
           FILTER isIRI(?o)
       }
       LIMIT 1000
       """)

qrdf = pd.DataFrame(columns=['s', 'o'])

for row in qres:
    qrdf = qrdf.append({'s': row[0], 'o': row[1]}, ignore_index=True)
#     print("%s : %s " % row)


# In[47]:


qrdf.head()


# ## Convert to NetworkX
# 
# Convert this to a networkx graph so we can explore some analytics calls

# In[48]:


import rdflib
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
from rdflib.extras.external_graph_libs import rdflib_to_networkx_digraph
import networkx as nx
import matplotlib.pyplot as plt

G = rdflib_to_networkx_digraph(parsed)
G=nx.from_pandas_edgelist(qrdf, source='s', target='o')


# In[49]:


# Pointless to draw for a graph this big..   just a black ball
nx.draw_circular(G, with_labels = False)
plt.show() # display


# In[50]:


plt.hist([v for k,v in nx.degree(G)])


# In[51]:


plt.hist(nx.centrality.betweenness_centrality(G).values())


# In[52]:


#  nx.diameter(G)  # found infinite  


# In[53]:


nx.cluster.average_clustering(G)


# ### Pagerank
# 
# Test a page rank call and see if we can load the results into Pandas and sort.

# In[54]:


import pandas as pd

pr = nx.pagerank(G,alpha=0.9)

prdf = pd.DataFrame.from_dict(pr, orient='index')
prdf.sort_values(by=0,ascending=False, inplace=True,)
prdf.head(10)


# ### NetworkX hits 
# 

# In[55]:


hits = nx.hits(G)  # can also be done with a nodelist (which is interesting.   provide with SPARQL call?)


# In[56]:


hitsdf = pd.DataFrame.from_dict(hits)
hitsdf.head()


# In[57]:


bc = nx.betweenness_centrality(G)


# In[58]:


bcdf = pd.DataFrame.from_dict(bc, orient='index')
bcdf.sort_values(by=0,ascending=False, inplace=True,)
bcdf.head(10)


# In[ ]:





# In[ ]:




