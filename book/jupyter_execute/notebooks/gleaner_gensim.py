#!/usr/bin/env python
# coding: utf-8

# # Gensim
# 
# This is an exploration of Gensim as a potential to create the "node set", V,  results from a semantic search.  That would wouild be fed into a graph database and used to start the path searches and or analysis to create the desired results set for an interface.
# 
# This V_semsearch might be intersected with a V_spatial and or others to form a node set for the graph.  This is essentially a search "preprocessor". Another potential set might be V_text that usses more classical full text index approaches.  
# 
# ## References
# 
# * https://github.com/topics/document-similarity
# * https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html
# 

# In[1]:


# %%capture
get_ipython().system('pip install -q --upgrade gensim')
get_ipython().system('pip install -q  dask[dataframe] --upgrade')
get_ipython().system('pip install -q s3fs')
get_ipython().system('pip install -q boto3')
get_ipython().system('pip install -q python-Levenshtein')


# In[2]:


import pprint
import spacy
from spacy import displacy
import pandas as pd
import dask, boto3
import dask.dataframe as dd


# ## Gleaner Data
# 
# First lets load up some of the data Gleaner has collected.  This is just simple data graph objects and not any graphs or other processed products from Gleaner. 

# In[3]:


# Set up our S3FileSystem object
import s3fs 
oss = s3fs.S3FileSystem(
      anon=True,
      client_kwargs = {"endpoint_url":"https://oss.geodex.org"}
   )


# ## Further examples

# In[4]:


import json

@dask.delayed()
def read_a_file(fn):
    # or preferably open in text mode and json.load from the file
    with oss.open(fn, 'rb') as f:
        #return json.loads(f.read().replace('\n',' '))
        return json.loads(f.read().decode("utf-8", "ignore").replace('\n',' '))

filenames = oss.ls('gleaner/summoned/opentopo')
output = [read_a_file(f) for f in filenames]


# In[5]:


gldf = pd.DataFrame(columns=['name', 'url', "keywords", "description"])

for doc in range(len(output)):
#for doc in range(10):
  try:
    jld = output[doc].compute()
  except:
    print("Doc has bad encoding")

  # TODO  Really need to flatten and or frame this

  desc = jld["description"]
  kws = jld["keywords"]
  name = jld["name"]
  url = jld["url"]  
  gldf = gldf.append({'name':name, 'url':url, 'keywords':kws, 'description': desc}, ignore_index=True)


# In[9]:


# gldf.info()


# In[10]:


import re

# document = "Human machine interface for lab abc computer applications"

# text_corpus = [
#     "Human machine interface for lab abc computer applications",
#     "A survey of user opinion of computer system response time",
#     "The EPS user interface management system",
#     "System and human system engineering testing of EPS",
#     "Relation of user perceived response time to error measurement",
#     "The generation of random binary unordered trees",
#     "The intersection graph of paths in trees",
#     "Graph minors IV Widths of trees and well quasi ordering",
#     "Graph minors A survey",
# ]

text_corpus = []

# for i in range(len(gldf)):
#   text_corpus += gldf.at[i,'description']

for i in range(len(gldf)):
# for i in range(10):
  d = gldf.at[i,'description']
  # d.replace('(', '').replace(')', '').replace('\"', '')
  dp = re.sub(r'[^A-Za-z0-9 ]+', '', str(d))
  text_corpus.append(str(dp))

  # if not "http" in d:
  #   if not "(" in d:
  #     if not "<" in d:
  #       text_corpus.append(str(d))


# In[12]:


# for x in range(len(text_corpus)):
#   print(text_corpus[x])


# In[13]:


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


# In[14]:


from gensim import corpora

dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)


# In[15]:


# pprint.pprint(dictionary.token2id)


# In[16]:


# Side demo
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)


# In[17]:


bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
# pprint.pprint(bow_corpus)


# In[19]:


from gensim import models

# train the model
tfidf = models.TfidfModel(bow_corpus)

# transform the "system minors" string
words = "system minors".lower().split()
print(tfidf[dictionary.doc2bow(words)])


# In[20]:


from gensim import similarities

index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)


# In[ ]:


query_document = 'Airborne Laser Mapping'.split()
query_bow = dictionary.doc2bow(query_document)
sims = index[tfidf[query_bow]]
print(list(enumerate(sims)))


# In[1]:


for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
    print(document_number, score)


# In[ ]:




