# D4R2018
From mobile phone data to population density
In March 2018, Turk Telekom shared part of its call detail records and lauched the Data4Refugees Big-Data Challenge (http://d4r.turktelekom.com.tr/). Here we show how, using these data, is possible to estimate the population density at the cell tower level.

This repo contains a simple code to train and test the best model that describe the population distribution (as proposed by Deville et al. in https://doi.org/10.1073/pnas.1408439111).
The input data are mobile phone activity densities and the population data with which the model is tested.
The output are the parameters of the model that can be used as general estimators for population densities over the same geographical area the model was trained.

