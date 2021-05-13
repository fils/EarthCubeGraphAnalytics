# Notebooks
## About

The following notebooks are exploring both document store access but 
also approaches to semantic search, clustering and semantic similarity.

```{admonition} Note to the reader
:class: tip
The following are not good notebooks to learn from..  as I am learning too.  ;)
```

### SciKit K-means Clustering

Exploring SciKit-Learn (https://scikit-learn.org/stable/) for semantic search. Here I am looking at the k-means approach (https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py). Specifically the Mini-Batch K-Means clustering (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html).

There are MANY approaches (https://scikit-learn.org/stable/auto_examples/index.html#cluster-examples) and it would be nice to get some guidance on what might make a good approach for building a similarity matrix of descriptive abstracts for datasets.

### Gensim

This is an exploration of Gensim as a potential to create the “node set”, V, results from a semantic search. That would would be fed into a graph database and used to start the path searches and or analysis to create the desired results set for an interface.

```{note}
I've not been able to get Gensim to produce the final output I expect due to a error
from one of the library dependencies.  I've tried clean environments and even a clean VM
and not been able to get 64 bit linux to work.  
```

This V_semsearch might be intersected with a V_spatial and or others to form a node set for the graph. This is essentially a search “preprocessor”. Another potential set might be V_text that uses a more classical full text index approaches.

### TXTAI

Exploring TXTAI (https://github.com/neuml/txtai) as yet another candidate in generating a set of nodes (V) that could be fed into a graph as the initial node set. Essentially looking at semantic search for the initial full text index search and then moving on to a graph database (triplestore in my case) fort he graph search / analysis portion.

This is the “search broker” concept I’ve been trying to resolve.

### EarthCube Graph Analytics Exploration

This is the start of learning a bit about leveraging graph analytics to assess the EarthCube graph and explore both the relationships but also look for methods to better search the graph for relevant connections.