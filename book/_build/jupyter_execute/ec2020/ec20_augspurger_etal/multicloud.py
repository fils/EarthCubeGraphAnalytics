#!/usr/bin/env python
# coding: utf-8

# # Multi-Cloud Workflow with Pangeo
# 
# <img src="https://cdn-images-1.medium.com/max/1200/1*uJwhZbe_Brx09yb_DKlI-g.png" width="20%" align="right">
# 
# This example demonstrates a workflow using analysis-ready data provided in two public clouds.
# 
# * [LENS](https://registry.opendata.aws/ncar-cesm-lens/) (Hosted on AWS in the us-west-2 region)
# * [ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) (Hosted on Google Cloud Platform in multiple regions)
# 
# We'll perform a similar analysis on each of the datasets, a histogram of the total precipitation, compare the results. Notably, this computation reduces a large dataset to a small summary. The reduction can happen on a cluster in the cloud.
# 
# By placing a compute cluster in the cloud next to the data, we avoid moving large amounts of data over the public internet. The large analysis-ready data only needs to move within a cloud region: from the machines storing the data in an object-store like `S3` to the machines performing the analysis. The compute cluster reduces the large amount of data to a small histogram summary. At just a handful of KBs, the summary statistics can easily be moved back to the local client, which might be running on a laptop. This also avoids costly egress charges from moving large amounts of data out of cloud regions.

# In[1]:


import getpass

import dask
from distributed import Client
from dask_gateway import Gateway, BasicAuth
import intake
import numpy as np
import s3fs
import xarray as xr
from xhistogram.xarray import histogram


# # Create Dask Clusters

# We've deployed [Dask Gateway](https://gateway.dask.org/) on two Kubernetes clusters, one in AWS and one in GCP. We'll use these to create [Dask](https://dask.org/) clusters in the same cloud region as the data. We'll connect to both of them from the same interactive notebook session.

# In[2]:


password = getpass.getpass()
auth = BasicAuth("pangeo", password)


# In[3]:


# Create a Dask Cluster on AWS
aws_gateway = Gateway(
    "http://a00670d37945911eab47102a1da71b1b-524946043.us-west-2.elb.amazonaws.com",
    auth=auth,
)
aws = aws_gateway.new_cluster()
aws_client = Client(aws, set_as_default=False)
aws_client


# In[4]:


# Create a Dask Cluster on GCP
gcp_gateway = Gateway(
    "http://34.72.56.89",
    auth=auth,
)
gcp = gcp_gateway.new_cluster()
gcp_client = Client(gcp, set_as_default=False)
gcp_client


# We'll enable adaptive mode on each of the Dask clusters. Workers will be added and removed as needed by the current level of computation.

# In[5]:


aws.adapt(minimum=1, maximum=200)
gcp.adapt(minimum=1, maximum=200)


# # ERA5 on Google Cloud Storage
# 
# We'll use `intake` and pangeo's data catalog to discover the dataset.

# In[6]:


cat = intake.open_catalog(
    "https://raw.githubusercontent.com/pangeo-data/pangeo-datastore/master/intake-catalogs/master.yaml"
)
cat


# The next cell loads the *metadata* as an xarray dataset. No large amount of data is read or transfered here. It  will be loaded on-demand when we ask for a concrete result later.

# In[7]:


era5 = cat.atmosphere.era5_hourly_reanalysis_single_levels_sa(
    storage_options={"requester_pays": False, "token": "anon"}
).to_dask()
era5


# We're computing the histogram on the total precipitation for a specific time period. xarray makes selecting this subset of data quite natural. Again, we still haven't loaded the data.

# In[8]:


tp = era5.tp.sel(time=slice('1990-01-01', '2005-12-31'))
tp


# To compare to the 6-hourly LENS dataset, we'll aggregate to 6-hourly totals.

# In[9]:


# convert to 6-hourly precip totals
tp_6hr = tp.coarsen(time=6).sum()
tp_6hr


# We'll bin this data into the following bins.

# In[10]:


tp_6hr_bins = np.concatenate([[0], np.logspace(-5,  0, 50)])
tp_6hr_bins


# The next cell applies the histogram to the `longitude` dimension and takes the mean over `time`.
# We're still just building up the computation here, we haven't actually loaded the data or executed it yet.

# In[11]:


tp_hist = histogram(
    tp_6hr.rename('tp_6hr'), bins=[tp_6hr_bins], dim=['longitude']
).mean(dim='time')
tp_hist.data


# In total, we're going from the ~1.5TB raw dataset down to a small 288 kB result that is the histogram summarizing the total precipitation. We've built up a large sequence of operations to do that reduction (over 110,000 individual tasks), and now it's time to actually execute it. There will be some delay between running the next cell, the scheduler receiving the task graph, and the cluster starting to process it, but work is happening in the background. After a minute or so, tasks will start appearing on the Dask dashboard.
# 
# One thing to note: we request this result with the `gcp_client`, the client for the cluster in the same cloud region as the data.

# In[12]:


era5_tp_hist_ = gcp_client.compute(tp_hist, retries=5)


# `gcp_tp_hist_` is a `Future` pointing to the result on the cluster. The actual computation is happening in the background, and we'll call `.result()` to get the concrete result later on.

# In[13]:


era5_tp_hist_


# Because the Dask cluster is in adaptive mode, this computation has kicked off a chain of events: Dask has noticed that it suddenly has many tasks to compute, so it asks the cluster manager (Kubernetes in this case) for more workers. THe Kubernetes cluster then asks it's compute backend (Google Compute Engine in this case) for more virtual machines. As these machines come online, our workers will come to life and the cluster will start progressing on our computation.

# ## LENS on AWS
# 
# This computation is very similar to the ERA5 computation. The primary difference is that the LENS dataset is an ensemble. We'll histogram a single member of that ensemble.
# 
# The Intake catalog created by NCAR includes many things, so we'll use `intake-esm` to search for the URL we want.

# In[14]:


col = intake.open_esm_datastore(
    "https://raw.githubusercontent.com/NCAR/cesm-lens-aws/master/intake-catalogs/aws-cesm1-le.json"
)
res = col.search(frequency='hourly6-1990-2005', variable='PRECT')
res.df


# In[15]:


url = res.df.loc[0, "path"]
url


# We'll (lazily) load that data from S3 using s3fs, xarray, and zarr.

# In[16]:


fs = s3fs.S3FileSystem(anon=True)
lens = xr.open_zarr(fs.get_mapper(url), consolidated=True)
lens


# In[17]:


hour = 60*60

precip_in_m = lens.PRECT * (6 * hour)
precip_in_m


# We'll select the first member for comparison with the ERA5 histogram.

# In[18]:


lens_hist = histogram(
    precip_in_m.isel(member_id=0).rename("tp_6hr"),
    bins=[tp_6hr_bins], dim=["lon"]
).mean(dim=('time'))


# In[19]:


lens_hist.data


# Note that we're using the `aws_client`, because LENS is stored in an AWS region.

# In[20]:


lens_hist_ = aws_client.compute(lens_hist)


# ## Compare results
# 
# Let's plot the histograms for both the ERA5 and LENS data. These are small results so it's safe to transfer the result from the cluster to the client machine for plotting.

# In[21]:


lens_tp_hist_ = lens_hist_.result()
era5_tp_hist_ = era5_tp_hist_.result()


# For ERA5:

# In[22]:


era5_tp_hist_[: ,1:].plot(xscale='log');


# And for LENS:

# In[23]:


lens_tp_hist_[: ,1:].plot(xscale='log');


# # Cleanup
# 
# Closing the clients will free all our resources.

# In[24]:


aws_client.close()
aws.close()

gcp_client.close()
gcp.close()


# ## Behind the Scenes
# 
# We deployed some infrastructure to make this notebook runnable. In line with one of Pangeo's guiding principles, each of these technologies has an [open architechture](https://medium.com/pangeo/closed-platforms-vs-open-architectures-for-cloud-native-earth-system-analytics-1ad88708ebb6).
# 
# ![](multi-cloud.png)
# 
# From low-level to high-level
# 
# * **Terraform** provides the tools for provisioning the cloud resources needed for the clusters.
# * **Kubernetes** provides the container orchestration for deploying the Dask Clusters. We created kubernetes clusters in AWS's `us-west-2` and GCP's `us-central1` regsions.
# * **Dask-Gatway** provides centralized, secure access to Dask Clusters. These clusters were deployed using [helm](https://helm.sh/) on two Kubernetes clusters.
# * **Dask** provides scalable, distributed computation for analyzing these large datasets
# * **xarray** provides high-level APIs and high-performance data structures for working with this data
# * **Intake, gcsfs, s3fs** provide catalogs for data discover and libraries for loading that data
# * **Jupyterlab** provides a user interface for interactive computing. The client laptop interacts with the clusters through Jupyterlab.
# 
# 
# All of the resources for this demo are available at https://github.com/pangeo-data/multicloud-demo.
