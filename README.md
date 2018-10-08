A repository for models of local election results at local authority level. The only model I've got so far is written in Stan - analysis is in Python in a Jupyter notebook.

To run the notebook you'll need to have the following python packages installed:

- pandas
- numpy
- matplotlib
- pystan
- arviz

The notebook downloads most of the data it needs directly, but unfortunately this wasn't possible in every single case. Earnings data must be downloaded to `data/data_in/earning_data.csv` from [here](https://download.beta.ons.gov.uk/downloads/datasets/ashe-table-8-earnings/editions/time-series/versions/1.csv) and population data must be downloaded to `data/data_in/population_estimates.xls` from [here](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/wardlevelmidyearpopulationestimatesexperimental)
