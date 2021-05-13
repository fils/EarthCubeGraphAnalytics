#!/usr/bin/env python
# coding: utf-8

# # Gleaner & txtai
# 
# ## About
# 
# Exploring TXTAI (https://github.com/neuml/txtai) as yet another canidate in generating a set of nodes (V) that could be fed into a graph as the initial node set.  Essentially looking at semantic search for the initial full text index search and then moving on to a graph database (triplestore in my case) fort he graph search / analysis portion.
# 
# This is the "search broker" concept I've been trying to resolve. 
# 
# ## References
# 
# * https://github.com/neuml/txtai
# 

# ## Imports and Installs

# In[1]:


# %%capture
get_ipython().system('pip install -q git+https://github.com/neuml/txtai')
get_ipython().system("pip install -q  'fsspec>=0.3.3'")
get_ipython().system('pip install -q  s3fs')
get_ipython().system('pip install -q  boto3')
get_ipython().system('pip install -q  spacy')
get_ipython().system('pip install -q  pyarrow')
get_ipython().system('pip install -q  fastparquet')


# In[24]:


import pprint
import spacy
from spacy import displacy
import pandas as pd
import dask, boto3
import dask.dataframe as dd
from txtai.embeddings import Embeddings

# Create embeddings model, backed by sentence-transformers & transformers
embeddings = Embeddings({"method": "transformers", "path": "sentence-transformers/bert-base-nli-mean-tokens"})


# ## Gleaner Data
# 
# First lets load up some of the data Gleaner has collected.  This is just simple data graph objects and not any graphs or other processed products from Gleaner. 

# In[29]:


# Set up our S3FileSystem object
import s3fs 
oss = s3fs.S3FileSystem(
      anon=True,
      client_kwargs = {"endpoint_url":"https://oss.geodex.org"}
   )
# oss.ls('gleaner/summoned')


# In[30]:


# # A simple example of grabbing one item...  
# import json 

# jld = ""
# with oss.open('gleaner/summoned/opentopo/231f7fa996be8bd5c28b64ed42907b65cca5ee30.jsonld', 'rb') as f:
#   #print(f.read())
#    jld = f.read().decode("utf-8", "ignore").replace('\n',' ')
#    json = json.loads(jld)

# document = json['description']
# print(document)


# In[31]:


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


# In[32]:


get_ipython().run_cell_magic('time', '', '\ngldf = pd.DataFrame(columns=[\'name\', \'url\', "keywords", "description", "object"])\n\n#for key in filenames:\n\nfor doc in range(len(output)):\n#for doc in range(10):\n#for key in filenames:\n  #if ".jsonld" in key:\n  if "/.jsonld" not in filenames[doc] :\n    try:\n      jld = output[doc].compute()\n    except:\n      print(filenames[doc])\n      print("Doc has bad encoding")\n\n    # TODO  Really need to flatten and or frame this\n    try:\n      desc = jld["description"]\n    except:\n      desc = "NA"\n      continue\n    kws = "keywords" #jld["keywords"]\n    name = jld["name"]\n    url = "NA" #jld["url"]\n    object = filenames[doc]\n\n    gldf = gldf.append({\'name\':name, \'url\':url, \'keywords\':kws, \'description\': desc, \'object\': object}, ignore_index=True)')


# In[42]:


gldf.info()


# In[36]:


gldf.to_parquet('index.parquet.gzip',  compression='gzip') 


# ## Erratta 

# In[37]:


import re

text_corpus = []

# for i in range(len(gldf)):
#   text_corpus += gldf.at[i,'description']

# for i in range(len(gldf)):
for i in range(10):
  d = gldf.at[i,'description']
  # d.replace('(', '').replace(')', '').replace('\"', '')
  dp = re.sub(r'[^A-Za-z0-9 ]+', '', str(d))
  text_corpus.append(str(dp))

  # if not "http" in d:
  #   if not "(" in d:
  #     if not "<" in d:
  #       text_corpus.append(str(d))

# for x in range(len(text_corpus)):
#   print(text_corpus[x])


# In[38]:


# Not needed for textai

# Create a set of frequent words
stoplist = set('for a of the and to in'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in text_corpus]

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
# pprint.pprint(processed_corpus)


# ## txtai section

# In[41]:


import numpy as np

# Create an index for the list of text_corpus
embeddings.index([(gldf.at[uid,'object'], text, None) for uid, text in enumerate(text_corpus)])
embeddings.save("index")
embeddings = Embeddings()
embeddings.load("index")

results = embeddings.search("lidar data ", 3)
for r in results:
  uid = r[0]
  score = r[1]
  print('score:{} -- {}\n\n'.format(score, uid)) #text_corpus[uid]))
  #print(gldf.at[uid,'object'])
  


# In[ ]:




