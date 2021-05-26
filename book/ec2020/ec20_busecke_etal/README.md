[![Binder](https://binder.pangeo.io/badge_logo.svg)](https://binder.pangeo.io/v2/gh/earthcube2020/ec20_busecke_etal/master?filepath=busecke_abernathey_earthcube2020.ipynb)
[![Speakerdeck](https://img.shields.io/badge/SPEAKERDECK-Slides-<COLOR>.svg)](https://speakerdeck.com/jbusecke/earthcube-2020-cmip6-without-the-interpolation-grid-native-analysis-with-pangeo-in-the-cloud)


# ec20_busecke_etal 

## CMIP6 without the interpolation: Grid-native analysis with Pangeo in the cloud.

*Julius Busecke, Ryan Abernathey*

The Pangeo project recently introduced large parts of the CMIP6 data archive into the cloud. This enables, for the first time, centralized, reproducible science of state-of-the-art climate simulations without the need to own large storage or a supercomputer as a user.
The data itself however, still presents significant challenges for analysis, one of which is applying operations across many models. Two of the major hurdles are different naming/metadata between modeling centers and complex grid layouts, particularly for the ocean components of climate models.
Common workflows in the past often included interpolating/remapping desired variables and creating new files, creating organizational burden, and increasing storage requirements.

We will demonstrate two Pangeo tools which enable seamless calculation of common operations like vector calculus operators (grad, curl, div) and weighted averages/integrals across a wide range of CMIP6 models directly on the data stored in the cloud. [`cmip6_preprocessing`](https://github.com/jbusecke/cmip6_preprocessing) provides numerous tools to unify naming conventions and parse grid information and metrics (like cell area). This information is used by [`xgcm`](https://xgcm.readthedocs.io/en/latest/) to enable finite volume analysis on the native model grids. The combination of both tools facilitates fast analysis while ensuring a reproducible and accurate workflow.
