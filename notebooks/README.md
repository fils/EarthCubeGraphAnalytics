# About

## Disclaimer

The graph analytics is rather lame.  It's not my forte so any guidance or advice is welcome.  

## gleaner_*

The three gleaner_* document represent some exploration I have been doing into using semantic search and 
semantic clustering to develop a set of initial node sets (V) that I can use.  The issue has been that 
when using a triplestore as the search entry point we suffer from issues around spatial and text indexing 
in those triplestores not being up to existing best practices.  Also, integration, like triplestore plus 
elasticsearch can be a bit complex for commodity deployment.  

I've been working on the idea that I could use special tools like semantic search, spatial (geohash) and 
text search to develop a set of node sets.   The intersect of these node sets results in a new node set 
that can then be passed to the graph store at the end to do the graph search and analytics that are sent
on to the client.  The client could be web based or machine based.  

So the union (ugh..  no math symbols in markdown)

V_semsearch U V_spaital = V_union

This V_union can be taken to the triplestore for use then.

Obviously this is FAR more complex than this for cases where the union is nil for all but not a subset of potential initial node sets.  

I'll try and detail out this "search pre-processor" concept more and why I feel the need to introduce that seems like
large scale complexity.  :) 