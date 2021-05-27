#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import numpy as np
import pandas as pd

import cmocean
import matplotlib.pylab as plt
from scipy.interpolate import griddata
from scipy import interpolate
from datetime import datetime
import pdb
import os
import csv

from datetime import datetime, timedelta
import calendar

import matplotlib
matplotlib.font_manager._rebuild()

#used for map projections
from cartopy import config
import cartopy.crs as ccrs
import matplotlib.patches as mpatches

get_ipython().run_line_magic('matplotlib', 'inline')

#sets plot styles
import seaborn as sns
from matplotlib import rc
from matplotlib import rcParams
import matplotlib.ticker as mtick
rc('text', usetex=False)
rcStyle = {"font.size": 10,
           "axes.titlesize": 20,
           "axes.labelsize": 20,
           'xtick.labelsize': 16,
           'ytick.labelsize': 16}
sns.set_context("paper", rc=rcStyle)
sns.set_style("whitegrid", {'axes.grid' : False})
myColors = ["windows blue", "amber", "dusty rose", "prussian blue", "faded green", "dusty purple", "gold", "dark pink", "green", "red", "brown"]
colorsBW = ["black", "grey"]
sns.set_palette(sns.xkcd_palette(myColors))

curDir = os.getcwd()
dataDir = os.path.join(curDir, 'data')

if not os.path.exists(dataDir):
    os.mkdir(dataDir)
    
import warnings
warnings.filterwarnings('ignore')


# # 1. Get a BGC profile
# 
# If you know that a profile contains BGC parameters, the standard profile api contains the bgc measurements under the field bgcMeas.

# In[56]:


def get_profile(profile_number):
    url = 'https://argovis.colorado.edu/catalog/profiles/{}'.format(profile_number)
    resp = requests.get(url)
    # Consider any status other than 2xx an error
    if not resp.status_code // 100 == 2:
        return "Error: Unexpected response {}".format(resp)
    profile = resp.json()
    return profile

def json2dataframe(profiles, measKey='measurements'):
    """ convert json data to Pandas DataFrame """
    # Make sure we deal with a list
    if isinstance(profiles, list):
        data = profiles
    else:
        data = [profiles]
    # Transform
    rows = []
    for profile in data:
        keys = [x for x in profile.keys() if x not in ['measurements', 'bgcMeas']]
        meta_row = dict((key, profile[key]) for key in keys)
        for row in profile[measKey]:
            row.update(meta_row)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


# In[45]:


profileId = "5901069_270"
profile = get_profile(profileId)
df = json2dataframe([profile], 'bgcMeas')


# In[7]:


df.head(5)


# # 2. Get a BGC Platform, two variables at a time
# Platform metadata is queried separatly from the BGC data. This is to keep the payload small enough for the server to operate efficiently.
# Platform BGC data is queried by two parameters at a time.

# In[57]:


def get_platform_profile_metadata(platform_number):
    url = f'https://argovis.colorado.edu/catalog/platform_profile_metadata/{platform_number}'
    print(url)
    resp = requests.get(url)
    # Consider any status other than 2xx an error
    if not resp.status_code // 100 == 2:
        return "Error: Unexpected response {}".format(resp)
    platformMetadata = resp.json()
    return platformMetadata

def get_platform_profile_data(platform_number, xaxis='doxy', yaxis='pres'):
    url = 'https://argovis.colorado.edu/catalog/bgc_platform_data/{0}/?xaxis={1}&yaxis={2}'.format(platform_number, xaxis, yaxis)
    print(url)
    resp = requests.get(url)
    # Consider any status other than 2xx an error
    if not resp.status_code // 100 == 2:
        return "Error: Unexpected response {}".format(resp)
    platformData = resp.json()
    return platformData

def join_platform_data(platformMetadata, platformData):
    platforms = []
    for idx, platform in enumerate(platformMetadata):
        metadata_id = platform['_id']
        data_id = platformData[idx]['_id']
        if (metadata_id == data_id) and ('bgcMeas' in platformData[idx].keys()) and isinstance(platformData[idx]['bgcMeas'], list):
            platform['bgcMeas'] = platformData[idx]['bgcMeas']
            platforms.append(platform)
    return platforms


# We merge the metadata and data and convert it into a dataframe

# In[58]:


platformMetadata = get_platform_profile_metadata(5901464)
platformData = get_platform_profile_data(5901464, 'doxy', 'pres')
platforms = join_platform_data(platformMetadata, platformData)
df = json2dataframe(platforms, 'bgcMeas')


# In[59]:


df.head()


# # 3. Get a BGC selection
# 
# https://argovis.colorado.edu/selection/bgc_data_selection?xaxis=temp&yaxis=pres&startDate=2020-03-01&endDate=2020-03-11&presRange=[0,50]&shape=[[[-155.929898,27.683528],[-156.984448,13.752725],[-149.468316,8.819693],[-142.15318,3.741443],[-134.922845,-1.396838],[-127.660888,-6.512815],[-120.250934,-11.523088],[-110.056944,-2.811371],[-107.069051,12.039321],[-118.141833,20.303418],[-125.314828,22.509761],[-132.702476,24.389053],[-140.290513,25.90038],[-148.048372,27.007913],[-155.929898,27.683528]]]

# In[64]:


def get_bgc_selection_profiles(startDate, endDate, shape, xaxis, yaxis, presRange=None, printUrl=True):
    url = 'https://argovis.colorado.edu/selection/bgc_data_selection'
    url += '?startDate={}'.format(startDate)
    url += '&endDate={}'.format(endDate)
    url += '&shape={}'.format(shape)
    url += '&xaxis={}'.format(xaxis)
    url += '&yaxis={}'.format(yaxis)
    if presRange:
        pressRangeQuery = '&presRange='.format(presRange)
        url += pressRangeQuery
    url = url.replace(' ', '')
    if printUrl:
        print(url)
    resp = requests.get(url)
    # Consider any status other than 2xx an error
    if not resp.status_code // 100 == 2:
        return "Error: Unexpected response {}".format(resp)
    selectionProfiles = resp.json()
    return selectionProfiles


# In[65]:


startDate = '2020-03-01'
endDate = '2020-03-11'
presRange = [0, 50]
shape = [[[-155.929898,27.683528],[-156.984448,13.752725],[-149.468316,8.819693],[-142.15318,3.741443],[-134.922845,-1.396838],          [-127.660888,-6.512815],[-120.250934,-11.523088],[-110.056944,-2.811371],[-107.069051,12.039321],[-118.141833,20.303418],          [-125.314828,22.509761],[-132.702476,24.389053],[-140.290513,25.90038],[-148.048372,27.007913],[-155.929898,27.683528]]]
xaxis='doxy'
yaxis='pres'
profiles = get_bgc_selection_profiles(startDate, endDate, shape, xaxis, yaxis, presRange, printUrl=True)


# In[66]:


df = json2dataframe(profiles, 'bgcMeas')


# In[67]:


df.head()


# In[ ]:




