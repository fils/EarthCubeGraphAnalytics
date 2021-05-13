Exploring GeoCODES
============================

These pages introduce some work related to exploring the GeoCODES graph and 
document store.  

The document store is an AWS S3 API compliant store leveraging the Minio 
open source project.  It could leverage any such system such as AWS S3, 
Ceph, GCS, Wasbi, etc.  The graph database is an RDF based triples accessed via
SPARQL.  The document store is synchronized to the graph and acts as the 
source of truth for the setup.  

Both serve different functions and compliment each other.  At this time
most of these examples are using the document store but that will change.
Mostly this is due to me exploring a few concepts such as:

* S3Select calls for data inspection, validation and sub-setting
* Dask based concurrent object access for updates and validation
  * SHACL
  * JSON inspection
  