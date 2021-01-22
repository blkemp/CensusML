# CensusML
Data dashboards utilising tools of Machine Learning and data from the Australian census.

## Intent
Use this repository as a base for functions to be deployed in the use of analysing the Australian Census data published by the Australian Bureau of Statistics. Current framework is building off prior analysis of solely 2016 data, however may be expanded to other years or other data sources. The intent is to build tools to allow for ease of integration into machine learning models, as well as general data exploration/charting, in addition to incorporating these tools into web app frameworks (using Dash, Plotly and Flask) to allow access by the general public and extending some of the functionality already provided by https://www.abs.gov.au/databyregion

## Requirements
* NumPy
* Pandas
* MatPlotLib
* SKLearn
* os
* TextWrap
* pickle
* re
* operator
* scipy
* tqdm
* statsmodels
* Dash 1.18.1
* * dash_core_components
* * dash_html_components
* * dash_bootstrap_components
* plotly 4.14.1
* importlib (optional - used in development when refining custom .py functions)