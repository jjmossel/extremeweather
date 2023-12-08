# Statistics of Extreme Weather
We use Extreme Value Theory (EVT) to study the statistics of extreme weather events (eg max daily rainfall). 

## Example: Extreme Rainfall in New York (Maximum likelihood approach)
We use a Maximum Likelihood Estimation (MLE) approach to model hourly rainfall in New York. Weather data is from single weather station.

[Example notebook: Maximum Likelihood Estimation](/notebooks/GEV_demo_PRCP_US01.ipynb)

## Example: Bayesian Approach for Extreme Rainfall Data
In this notebook we use a Bayesian Approach (implemented in PyMC) to model extreme weather events. We start with a univariate study (which parallels the MLE approach) for a single weather station. Next we train a spatial model, by incorporating data from many (nearby) weather stations, using a Gaussian processes as prior distribution for the GEV model parameters.

[Example notebook: Bayesian Approach with Gaussian Process](/notebooks/GEV_probabilistic_presentation.ipynb)

[PyData 2023 Talk](PyData2023_JornMossel_Modeling%20Extreme%20Events%20with%20PyMC.pdf)


## References
* [An Introduction to Statistical Modeling of Extreme Values - Coles (2001)](https://link.springer.com/book/10.1007/978-1-4471-3675-0)
* [Statistical Modeling of Spatial Extremes - Davison, Padoan and Ribatet (2012)](https://arxiv.org/pdf/1208.3378.pdf)
* [Generalized Extreme Value Distribution in PyMC - Caprani (2021)](https://www.pymc.io/projects/examples/en/latest/case_studies/GEV.html)
* [Modeling spatial data with Gaussian processes in PyMC - Paz (2022)](https://www.pymc-labs.com/blog-posts/spatial-gaussian-process-01/)
