# time_series_forecast_nn
An easy to use template to predict timeseries via a neural network with keras (tensorflow backend)

This template loads two time series, one for input data (data that correlates with the prediction of interest) and one for output data (data to predict). This data is used to train a neural network (deep neural network or GRU neural network can be chosen). The template contains two hidden layers. 

## Usage
In order to use this template, it simply can be run. All necessary data for a test run are in this repository. In order for tensorflow to work it needs a Python version of 3.3 - 3.6.

### Test run
The repository contains two time series that can be used for a test run. The input data contains the power of renewables in the German energy grid as well as the total electrical load. The ouput data, that should be predicted, is the spot market price of the EEX. Both time series are for the year 2015. 