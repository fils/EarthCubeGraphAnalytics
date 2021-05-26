#!/usr/bin/env python
# coding: utf-8

# # CMIP6 without the interpolation: Grid-native analysis with Pangeo in the cloud
# *Julius Busecke*<sup>1</sup> and *Ryan Abernathey*<sup>2</sup>
# 
# (<sup>1</sup> Princeton University  <sup>2</sup> Columbia University)
# 
# The coupled model intercomparison project (CMIP; currently in its [6th iteration](https://www.geosci-model-dev.net/9/1937/2016/)) involves contribution from countries around the world, constituting a combined dataset of of 20+ Petabytes.
# 
# Many years of planning go into the inception of the different scenarios, various model intercomparison projects (MIPs) and the delivery of the output data from the modeling community around the globe. A major challenge for the earth science community now, is to effectively analyze this dataset and answer science question that improve our understanding of the earth system for the coming years.
# 
# The Pangeo project recently introduced large parts of the CMIP6 data archive into the [cloud](https://medium.com/pangeo/cmip6-in-the-cloud-five-ways-96b177abe396). This enables, for the first time, centralized, reproducible science of state-of-the-art climate simulations without the need to own large storage or a supercomputer as a user.
# 
# The data itself however, still presents significant challenges for analysis, one of which is applying operations across many models. Two of the major hurdles are different naming/metadata between modeling centers and complex grid layouts, particularly for the ocean components of climate models. Common workflows in the past often included interpolating/remapping desired variables and creating new files, creating organizational burden, and increasing storage requirements.
# 
# We will demonstrate two pangeo tools which enable seamless calculation of common operations like vector calculus operators (grad, curl, div) and weighted averages/integrals across a wide range of CMIP6 models directly on the data stored in the cloud. `cmip6_preprocessing` provides numerous tools to unify naming conventions and reconstruct grid information and metrics (like distances). This information is used by `xgcm` to enable finite volume analysis on the native model grids. The combination of both tools facilitates fast analysis while ensuring a reproducible and accurate workflow.

# In[1]:


from dask.distributed import Client, progress
from dask_gateway import Gateway

gateway = Gateway()
cluster = gateway.new_cluster()
cluster.scale(60)
cluster


# In[2]:


client = Client(cluster)
client


# # Loading the CMIP6 data from the cloud

# In order to load the cmip data from the cloud storage, we use `intake-esm`, and choose to load all monthly ocean model output (`table_id='Omon'`) of sea surface temperature (`variable_id='tos'`) on the native grid (`grid_label='gn'`) and for the 'historical' run (`experiment_id='historical'`). 
# For more details on how to use intake-esm we refer to this [Earth Cube presentation]( https://github.com/andersy005/intake-pangeo-catalog-EarthCube-2020). 

# In[3]:


import intake
import warnings
warnings.filterwarnings("ignore")

url = "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"
col = intake.open_esm_datastore(url)
models = col.df['source_id'].unique()
# we will have to eliminate some models that do not *yet* play nice with cmip6_preprocessing
# ignore_models = ['FGOALS-f3-L','EC-Earth3','EC-Earth3-Veg-LR', 'CESM2-FV2',
#                  'GFDL-ESM4', 'MPI-ESM-1-2-HAM', 'MPI-ESM1-2-LR', 'NorCPM1',
#                  'GISS-E2-1-H', 'IPSL-CM6A-LR', 'MRI-ESM2-0', 'CESM2-WACCM',
#                  'GISS-E2-1-G', 'FIO-ESM-2-0', 'MCM-UA-1-0', 'FGOALS-g3']
# models = [m for m in models if 'AWI' not in m and m not in ignore_models]

# These might change in the future so it is better to explicitly allow the models that work

models = ['TaiESM1','BCC-ESM1','CNRM-ESM2-1','MIROC6','UKESM1-0-LL','NorESM2-LM',
          'BCC-CSM2-MR','CanESM5','ACCESS-ESM1-5','E3SM-1-1-ECA','E3SM-1-1',
          'MIROC-ES2L','CESM2','CNRM-CM6-1','GFDL-CM4','CAMS-CSM1-0','CAS-ESM2-0',
          'CanESM5-CanOE','IITM-ESM','CNRM-CM6-1-HR','ACCESS-CM2','E3SM-1-0',
          'EC-Earth3-LR','EC-Earth3-Veg','INM-CM4-8','INM-CM5-0','HadGEM3-GC31-LL',
          'HadGEM3-GC31-MM','MPI-ESM1-2-HR','GISS-E2-1-G-CC','GISS-E2-2-G','CESM2-WACCM-FV2',
          'NorESM1-F','NorESM2-MM','KACE-1-0-G','GFDL-AM4','NESM3',
          'SAM0-UNICON','CIESM','CESM1-1-CAM5-CMIP5','CMCC-CM2-HR4','CMCC-CM2-VHR4',
          'EC-Earth3P-HR','EC-Earth3P','ECMWF-IFS-HR','ECMWF-IFS-LR','INM-CM5-H',
          'IPSL-CM6A-ATM-HR','HadGEM3-GC31-HM','HadGEM3-GC31-LM','MPI-ESM1-2-XR','MRI-AGCM3-2-H',
          'MRI-AGCM3-2-S','GFDL-CM4C192','GFDL-OM4p5B']

cat = col.search(table_id='Omon', grid_label='gn', experiment_id='historical', variable_id='tos', source_id=models)


# We have to eliminate some models from this analysis: The `AWI` models, due to their unstructured native grid, as well as others for various subtleties that have yet to be resolved. This will be addressed in a future update of `cmip6_preprocessing`. If you do not see your favorite model, please consider raising an issue on [github](https://github.com/jbusecke/cmip6_preprocessing/issues).

# In[4]:


cat.df['source_id'].nunique() # This represents the number of models as of 6/14/2020.
                              # More models might be added in the future. See commented parts in previous code cell.


# This gives us 27 different models ('source_ids') to work with. Lets load them into a dictionary and inspect them closer.

# In[5]:


# Fix described here: https://github.com/intake/filesystem_spec/issues/317
# Would cause `Name (gs) already in the registry and clobber is False` error somewhere in `intake-esm`
import fsspec
fsspec.filesystem('gs')

ddict = cat.to_dataset_dict(zarr_kwargs={'consolidated':True, 'decode_times':False})


# What we ultimately want is to apply an analysis or just a visualization across all these models. So before we jump into that, lets inspect a few of the datasets:

# In[6]:


ddict['CMIP.NCAR.CESM2.historical.Omon.gn']


# In[7]:


ddict['CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.Omon.gn']


# In[8]:


ddict['CMIP.CSIRO.ACCESS-ESM1-5.historical.Omon.gn']


# We can immediately spot problems:
# 
# - There is no consistent convention for the labeling of dimensions (note the 'logical 1D dimension in the x-direction is called `x`, `nlon`, `i`)
# 
# - Some models (here: `CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.Omon.gn`) are missing the 'vertex'/'vertices' dimension, due to the fact that the cell geometry is given by longitude and latitude 'bounds' centered on the cell face compared to the corners of the cell.
# 
# There are more issues that both make working with the data cumbersome and can quickly introduce errors. For instance, some models give depth in units of cm, not m, and many other issues. Instead of focusing on problems, here we would like to illustrate how easy analysis across models with different grids can be, using `cmip6_preprocessing`.

# ## Use `cmip6_preprocessing` to harmonize different model output 
# 
# [`cmip6_preprocessing`](https://github.com/jbusecke/cmip6_preprocessing) was born out of the [cmip6-hackathon](https://cmip6hack.github.io/#/) and aims to provide a central package through which these conventions can be harmonized. For convenience we will make use of the `preprocess` functionality and apply the 'catch-all' function `combined_preprocessing` to the data before it gets aggregated. For a more detailed description of the corrections applied we refer to the [documentation and examples therein](https://github.com/jbusecke/cmip6_preprocessing/blob/master/doc/tutorial.ipynb).

# In[9]:


from cmip6_preprocessing.preprocessing import combined_preprocessing

ddict_pp = cat.to_dataset_dict(
    zarr_kwargs={'consolidated':True, 'decode_times':False},
    preprocess=combined_preprocessing
)


# > Note that all functions in the `cmip6_preprocessing.preprocessing` modules can also be used with 'raw' datasets, like e.g. a local netcdf file, as long as the metadata is intact.
# 
# Now lets look at the preprocessed version of the same datasets from above:

# In[10]:


ddict_pp['CMIP.NCAR.CESM2.historical.Omon.gn']


# In[11]:


ddict_pp['CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.Omon.gn']


# In[12]:


ddict_pp['CMIP.CSIRO.ACCESS-ESM1-5.historical.Omon.gn']


# Much better! 
# 
# As you can see, they all have consistent dimension names and coordinates. Time to see if this works as advertised, by plotting the sea surface temperature (SST) for all models

# In[13]:


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig, axarr = plt.subplots(
    ncols=5, nrows=6, figsize=[15, 15], subplot_kw={"projection": ccrs.Robinson(190)}
)
for ax, (k, ds) in zip(axarr.flat, ddict_pp.items()):
    # Select a single member for each model
    if 'member_id' in ds.dims:
        ds = ds.isel(member_id=-1)

    # select the first time step
    da = ds.tos.isel(time=0)
    
    # some models have large values instead of nans, so we mask unresonable values
    da = da.where(da < 1e30)
    
    # and plot the resulting 2d array using xarray
    pp = da.plot(
        ax=ax,
        x="lon",
        y="lat",
        vmin=-1,
        vmax=28,
        transform=ccrs.PlateCarree(),
        infer_intervals=False,
        add_colorbar=False,
    )
    
    ax.set_title(ds.attrs['source_id'])
    ax.add_feature(cfeature.LAND, facecolor='0.6')
    ax.coastlines()
fig.subplots_adjust(hspace=0.05, wspace=0.05)
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])
fig.colorbar(pp, cax=cbar_ax, label="Sea Surface Temperature [$^\circ C$]")


# That was very little code to plot such a comprehensive figure, even if the models all have different resolution and grid architectures.
# 
# 
# `cmip6_preprocessing` helps keeping the logic needed for simple analysis and visualization to a minimum, but in the newest version is also helps to facilitate the use of [`xgcm`](https://github.com/xgcm/xgcm) with the native model grids, making it a powerful tool for more complex analysis across the models.

# ## Calculations on native model grids using `xgcm`.
# 
# Most modern circulation models discretize the partial differential equations needed to simulate the earth system on a logically [rectangular grid](https://xgcm.readthedocs.io/en/latest/grids.html). This means the grid for a single time step can be represented as a 3-dimensional array of cells. Operations like e.g., a derivative are then approximated by a finite difference between neighboring cells, divided by the appropriate distance.
# 
# Calculating operators like *gradient*, *curl* or *divergence* is usually associated with a lot of bookkeeping to make sure that the difference is taken between the correct grid points, etc. This often leads to users preferring interpolated grids (on regular lon/lat grids), which have to be processed and stored next to the native grid data, both increasing storage requirements and potentially losing some of the detailed structure in the high-resolution model fields.
# 
# The combination of `cmip6_preprocessing` and `xgcm` enables the user to quickly calculate operators like *gradient* or *curl* (shown in detail below) on the native model grids, preserving the maximum detail the models provide and preventing unnecessary storage burden. `cmip6_preprocessing` parses the detailed grid information for the different models so that `xgcm` can carry out the computations, without requiring the user to dive into the details of each model grid.

# As an example, let us compute and plot the gradient magnitude of the SST.
# 
# $ \overline{\nabla(SST)} = \sqrt{\frac{\partial SST}{\partial x}^2 + \frac{\partial SST}{\partial y}^2} $
# 
# We will start with an example of just computing the zonal gradient $\frac{\partial SST}{\partial x}$:
# 
# First, we need to create a suitable grid, with dimensions both for the cell center (where our tracer is located) and the cell faces. These are needed for xgcm to calculate the finite distance version of the above equation. 
# 
# The function `combine_staggered_grid` parses a dataset so that it is compatible with xgcm. It also provides a 'ready-to-use' [xgm Grid object](https://xgcm.readthedocs.io/en/latest/grids.html#creating-grid-objects) for convenience.

# In[14]:


# we pick one of the many models
ds = ddict_pp['CMIP.CNRM-CERFACS.CNRM-CM6-1-HR.historical.Omon.gn']
ds


# In[15]:


from cmip6_preprocessing.grids import combine_staggered_grid

# Now we parse the necessary additional dimensions
grid,ds = combine_staggered_grid(ds)
ds


# Note how the new dimensions `x_left` and `y_right` are added, which are associated with the 'eastern' and 'northern' cell faces. 
# We can now easily calculate the finite difference in the logical `X` direction:

# In[16]:


delta_t_x = grid.diff(ds.tos, 'X')
delta_t_x


# This array is now located on the center of the eastern cell face. But in order to recreate a derivative, we need to divide this difference by an appropriate distance. Usually these distances are provided with model output, but currently there is no such data publicly available for CMIP models. 
# 
# To solve this problem, `cmip6_preprocessing` can recalculate grid distances.
# > Be aware that this can lead to rather large biases when the grid is strongly warped, usually around the Arctic. More on that at the end of the notebook.

# In[17]:


grid, ds = combine_staggered_grid(ds, recalculate_metrics=True)


# In[18]:


ds


# Now there are additional coordinates, representing distances in the x and y direction at different grid locations. We can now calculate the derivative with respect to x.

# In[19]:


dt_dx = grid.diff(ds.tos, 'X') / ds.dx_gx
dt_dx


# `xgcm` can handle these cases even smarter, by automatically picking the right [grid metric](https://xgcm.readthedocs.io/en/latest/grid_metrics.html), in this case the zonal distance.

# In[20]:


dt_dx_auto = grid.derivative(ds.tos, 'X')
dt_dx_auto


# In[21]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=[13,5])
dt_dx.isel(time=0).sel(x_left=slice(120, 140), y=slice(20,40)).plot(ax=ax1, vmax=1e-4)
dt_dx_auto.isel(time=0).sel(x_left=slice(120, 140), y=slice(20,40)).plot(ax=ax2, vmax=1e-4)


# The results are identical, and the user does not have to remember the names of the distance coordinates, `xgcm` takes care of it. The derivative in the y-direction can be calculated similarly.
# 
# Then, let's compute the full gradient amplitude for **all** the models in one loop, averaging the values over all members and 5 years at the beginning of the historical period.

# In[22]:


fig, axarr = plt.subplots(
    ncols=4, nrows=7, figsize=[15, 15], subplot_kw={"projection": ccrs.Robinson(190)}
)
for ax, (k, ds) in zip(axarr.flat, ddict_pp.items()):
    
    # the cmip6_preprocessing magic: 
    # create an xgcm grid object and dataset with reconstructed grid metrics
    grid, ds = combine_staggered_grid(ds, recalculate_metrics=True)
    
    da = ds.tos

    # some models have large values instead of nans, so we mask unresonable values
    da = da.where(da < 1e30)

    # calculate the zonal temperature gradient
    dt_dx = grid.derivative(da, 'X')
    dt_dy = grid.derivative(da, 'Y', boundary='extend')
    # these values are now situated on the cell faces, we need to 
    # interpolate them back to the center to combine them

    dt_dx = grid.interp(dt_dx, 'X')
    dt_dy = grid.interp(dt_dy, 'Y', boundary='extend')
    grad = (dt_dx**2 + dt_dy**2)**0.5

    ds['grad'] = grad

    # take an average over the first 5 years of the run
    ds = ds.isel(time=slice(0,12*5)).mean('time', keep_attrs=True)

    # take the average over all available model members
    if "member_id" in ds.dims:
        ds = ds.mean('member_id', keep_attrs=True)

    # and plot the resulting 2d array using xarray
    pp = ds.grad.plot(
        ax=ax,
        x="lon",
        y="lat",
        vmin=1e-9,
        vmax=1e-5,
        transform=ccrs.PlateCarree(),
        infer_intervals=False,
        add_colorbar=False,
    )
    ax.set_title(ds.attrs['source_id'])
    ax.add_feature(cfeature.LAND, facecolor='0.6')
    ax.coastlines()
    
fig.subplots_adjust(hspace=0.05, wspace=0.05)
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])
fig.colorbar(pp, cax=cbar_ax, label="Sea Surface Temperature gradient magnitude [$^\circ C/m$]")


# Not bad for five additional lines of code! 
# 
# Features like the western boundary currents and the fronts across along the Antarctic Circumpolar Current and the equatorial upwelling zones are clearly visible and present in most models.

# ## Calculate surface vorticity in the North Atlantic
# 
# But this was still just operating on a tracer field. What about velocity fields? We can also combine different variables into an xgcm compatible dataset!
# 
# The recommended workflow is to first load datasets seperately for each desired variable:

# In[23]:


variables = ['thetao', 'uo', 'vo']
ddict_multi = {var:{} for var in variables}
for var in variables:
    cat_var = col.search(table_id='Omon',
                         grid_label='gn',
                         experiment_id='historical',
                         variable_id=var,
                         source_id=models)

    ddict_multi[var] = cat_var.to_dataset_dict(
        zarr_kwargs={'consolidated':True, 'decode_times':False},
        preprocess=combined_preprocessing
    )


# Now we need to make sure to choose only the intersection between the keys of each sub-dictionary, because we need all 3 variables for each model.

# In[24]:


ddict_multi_clean = {var:{} for var in variables}
for k in ddict_multi['thetao']:
    if all([k in ddict_multi[var] for var in variables]) and 'lev' in ddict_multi['thetao'][k].dims:
        for var in variables:
            ddict_multi_clean[var][k] = ddict_multi[var][k]


# And now similar as before we can create a complete grid dataset, but we can pass additional datasets to be combined using the `other_ds` argument.  This can be a list of different variables, `cmip6_preprocessing` automatically detects the appropriate grid position.

# In[25]:


import matplotlib.path as mpath
import numpy as np
from xgcm import Grid
import xarray as xr

fig, axarr = plt.subplots(
    ncols=4, nrows=6, figsize=[25, 20], subplot_kw={"projection": ccrs.PlateCarree()}
)
for ai ,(ax, k) in enumerate(zip(axarr.flat, ddict_multi_clean['thetao'].keys())):
    ds_t = ddict_multi_clean['thetao'][k] # One tracer should always be the reference dataset!
    ds_u = ddict_multi_clean['uo'][k]
    ds_v = ddict_multi_clean['vo'][k]

    if 'lev' in ds_t: # show only models with depth coordinates in the vertical
        # aling only the intersection of the member_ids
        ds_t, ds_u, ds_v = xr.align(ds_t, ds_u, ds_v, join='inner', exclude=[di for di in ds_t.dims if di != 'member_id'])

        # combine all datasets and create grid with metrics
        grid, ds = combine_staggered_grid(ds_t, other_ds=[ds_u, ds_v], recalculate_metrics=True)


        # interpolate grid metrics at the corner points (these are not *yet* constructed by cmip6_preprocessing)
        ds.coords['dx_temp'] = grid.interp(ds['dx_gx'], 'Y', boundary='extend')
        ds.coords['dy_temp'] = grid.interp(ds['dy_gy'], 'X')
        
        ds = ds.chunk({'x':360})
        
        # selecting the surface value
        ds = ds.isel(lev=0)
        
        # For demonstration purposes select the first member and a single monthly timestep
        ds = ds.isel(time=10)
        
        if "member_id" in ds.dims:
            ds = ds.isel(member_id=0)

        grid = Grid(ds, periodic=['X'], metrics={'X':[co for co in ds.coords if 'dx' in co],
                                                'Y':[co for co in ds.coords if 'dy' in co]})


        dv_dx = grid.derivative(ds.vo, 'X')
        du_dy = grid.derivative(ds.uo, 'Y',boundary='extend')

        # check the position of the derivatives and interpolate back to tracer point for plotting
        if any(['x_' in di for di in dv_dx.dims]):
            dv_dx = grid.interp(dv_dx, 'X')
        if any(['y_' in di for di in dv_dx.dims]):
            dv_dx = grid.interp(dv_dx, 'Y', boundary='extend')

        if any(['x_' in di for di in du_dy.dims]):
            du_dy = grid.interp(du_dy, 'X')
        if any(['y_' in di for di in du_dy.dims]):
            du_dy = grid.interp(du_dy, 'Y', boundary='extend')

        curl = dv_dx - du_dy

        ds['curl'] = curl
        
        # Select the North Atlantic
        ds = ds.sel(x=slice(270,335), y=slice(10,55))

        # and plot the resulting 2d array using xarray
        pp = ds.curl.plot.contourf(
            levels=51,
            ax=ax,
            x="lon",
            y="lat",
            vmax=1e-5,
            transform=ccrs.PlateCarree(),
            infer_intervals=False,
            add_colorbar=False,
            add_labels=False
        )
        ax.text(0.2,0.9,ds.attrs['source_id'],horizontalalignment='center',verticalalignment='center',
                transform=ax.transAxes, fontsize=10)
        ax.coastlines()
        ax.set_extent([-90, -45, 20, 45], ccrs.PlateCarree())
fig.subplots_adjust(wspace=0.01, hspace=0.01)
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])
fig.colorbar(pp, cax=cbar_ax, label="Sea Surface Vorticity [$1/s$]")


# ## A caveat: Reconstructed metrics
# 
# For all the examples shown here, we have relied on a simplified reconstruction of the grid metrics (in this case, the distances between different points on the grid). We can check the quality of the reconstruction indirectly by comparing these to the only horizontal grid metric that is saved with the ocean model output: The horizontal grid cell area.
# 
# We reconstruct our area naively as the product of the x and y distances centered on the tracer cell, `dx_t`, and `dy_t`.

# In[26]:


ds = ddict_multi_clean['thetao']['CMIP.NOAA-GFDL.GFDL-CM4.historical.Omon.gn']
_,ds = combine_staggered_grid(ds, recalculate_metrics=True)
area_reconstructed = ds.dx_t * ds.dy_t


# Now lets load the actual area and compare the two.

# In[27]:


url = "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"
col = intake.open_esm_datastore(url)
cat = col.search(table_id='Ofx', variable_id='areacello', source_id='GFDL-CM4' , grid_label='gn') # switch to `grid_label='gr'` for regridded file
ddict = cat.to_dataset_dict(zarr_kwargs={'consolidated':True}, preprocess=combined_preprocessing)
_,ds_area = ddict.popitem()
area = ds_area.areacello.isel(member_id=0).squeeze()


# In[28]:


fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=[20,5])
area_reconstructed.plot(ax=ax1, vmax=1e9)
area.plot(ax=ax2, vmax=1e9)
((area_reconstructed - area) / area * 100).plot(ax=ax3, vmax=0.5)
ax1.set_title('reconstructed area')
ax2.set_title('saved area')
ax3.set_title('difference in %')


# You can see that for this particular model (`GFDL-CM4`), the reconstruction does reproduce the grid area with a less than 0.5% error between approximately 60S-60N. North of that, the grid geometry gets vastly more complicated, and the simple reconstruction fails. The area over which the reconstruction varies from model to model and as such caution should always be exercised when analyzing data using reconstructed metrics.
# 
# At this point, this is, however, the only way to run these kinds of analyses, since the original grid metrics are not routinely provided with the CMIP6 output. In order to truly reproduce analyses like e.g., budgets, the community requires the native model geometry.

# ## Conclusions

# We show that using `cmip6_preprocessing` and `xgcm` users can analyze complex native ocean model grids without the need to interpolate data or keep track of the intricacies of single model grids. Using these tools already enables users to calculate common operators like the gradient and curl, [weighted averages](https://xgcm.readthedocs.io/en/latest/grid_metrics.html#Grid-aware-(weighted)-average) and more, in a 'grid-agnostic' way, with decent precision outside of the polar regions.
# 
# Pending on the availability of more grid metric output and building on these tools, complex analyses like various budgets could become a matter of a few lines and be calculated across all models in the CMIP6 archive. 
# 
# By combining generalizable and reproducible analysis with the [publicly available CMIP6 data](https://medium.com/pangeo/cmip6-in-the-cloud-five-ways-96b177abe396), more users will be able to analyze the data efficiently, leading to faster understanding and synthesis of the vast amount of data provided by modern climate modeling. 

# ### Acknowledgements
# Julius Busecke's contributions to this work are motivated and developed as part of an ongoing project together with his postdoc advisor [Prof. Laure Resplandy](http://resplandy.princeton.edu/). This project investigates the influence of [equatorial dynamics on possible expansions of Oxygen Minimum Zones in the Pacific](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2019GL082692) and is funded under the NOAA Cooperative Institute for Climate Science agreement NA14OAR4320106.
