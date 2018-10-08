import functions
import numpy as np
import pandas as pd


def fetch_vote_shares():
    # hardcoded strings
    url_2018 = "https://candidates.democracyclub.org.uk/media/csv-archives/results-2018-05-03.csv"
    at_url = "http://www.andrewteale.me.uk/pdf/{0}/{0}_results.zip"
    at_colnames = ['ward', 'council', 'candidate', 'party_at', 'votes', 'elected']
    dc_ballot_colnames = ['election_type', 'council', 'ward', 'date', '???']
    final_colnames = ['council', 'ward', 'year', 'party', 'votes']
    at_party_names = {'Lab': 'labour', 'C': 'conservative', 'LD': 'libdem', 'UKIP': 'ukip', 'Grn': 'green'}
    dc_party_names = {'Labour Party': 'labour', 'Labour and Co-operative Party': 'labour',
                      'Conservative and Unionist Party': 'conservative', 'Liberal Democrats': 'libdem',
                      'UK Independence Party (UKIP)': 'ukip', 'Green Party': 'green'}
    # get 2018 results from democracy club website
    results_2018 = pd.read_csv(url_2018, header=1).assign(year=2018)
    info_2018 = pd.DataFrame.from_records(results_2018['ballot_paper_id'].str.split('.').tolist())
    info_2018.columns = dc_ballot_colnames
    info_2018['council'] = info_2018['council'].str.replace('-', ' ')
    info_2018['ward'] = info_2018['ward'].str.replace('-', ' ')
    results_2018 = results_2018.join(info_2018).loc[lambda df: df['election_type'] == 'local']
    results_2018['party'] = results_2018['party_name'].map(dc_party_names).fillna('other')
    results_2018['votes'] = results_2018['ballots_cast'].astype(int)
    results_2018['candidate'] = results_2018['person_name']
    # get older results from Andrew Teale's website
    results_at = pd.concat([
        pd.read_csv(at_url.format(str(year)),
                    header=None,
                    names=at_colnames,
                    skiprows=1 if year == 2014 else None)
        .assign(year=year,
                party=lambda df: df['party_at'].map(at_party_names).fillna('other'),
                ward=lambda df: df['ward'].str.lower(),
                council=lambda df: df['council'].str.lower(),
                votes=lambda df: df['votes'].astype(int))
        for year in [2011, 2012, 2014]
    ])
    # put all results together
    results = pd.concat([results_at[final_colnames], results_2018[final_colnames]])
    # aggregate results into vote shares
    shares = (results
              .groupby(['year', 'council', 'party'])
              ['votes']
              .sum()
              .unstack()
              .pipe(functions.proportionise)
              .fillna(0))
    # calculate labour share of 2 party vote
    shares['lab_2p'] = shares['labour'] / (shares['labour'] + shares['conservative'])
    return shares


def fetch_earnings(authorities: list):
    """First download the (very big) csv from https://download.beta.ons.gov.uk/downloads/datasets/ashe-table-8-earnings/editions/time-series/versions/1.csv
    and save it to data/data_in/earning_data.csv
    :param authorities: list of local authorities NB must be all lower case
    """
    return (pd.read_csv("../data/data_in/earning_data.csv")
            .assign(authority=lambda df: df['Geography'].str.lower())
            .loc[lambda df:
                 (df['authority'].isin(authorities))
                 & (df['Sex_codelist'] == 'all')
                 & (df['Statistics_codelist'] == 'median')
                 & (df['Workingpattern_codelist'] == 'all')
                 & (df['Earnings'] == 'Annual pay - Gross')]
            .set_index(['authority', 'Time'])['V4_2'] .unstack())


def fetch_authority_to_region():
    url = "http://geoportal1-ons.opendata.arcgis.com/datasets/c457af6314f24b20bb5de8fe41e05898_0.csv"
    return (pd.read_csv(url)
            .assign(authority=lambda df: df['LAD17NM'].str.lower(),
                    region=lambda df: df['RGN17NM'].str.lower())
            .set_index('authority')
            ['region'])


def fetch_housing_waiting_lists():
    url = "https://www.gov.uk/government/uploads/system/uploads/attachment_data/file/674350/LT_600.xlsx"
    return (
        pd.read_excel(url, header=0, skiprows=3, usecols=[4, 19, 26])
        .dropna()
        .rename(columns={'Lower and Single Tier Authority Data': 'authority'})
        .assign(authority=lambda df: df['authority'].str.lower())
        .set_index('authority')
        .replace('..', np.nan)
        .astype(float)
        .assign(change=lambda df: df[2017] - df[2010],
                ratio=lambda df: df[2017] / df[2010])
    )


def fetch_leave_vote_share():
    url = "https://www.electoralcommission.org.uk/__data/assets/file/0014/212135/EU-referendum-result-data.csv"
    return (pd.read_csv(url)
            .assign(authority=lambda df: df['Area'].str.lower())
            .groupby('authority')['Pct_Leave'].first()
            .rename('leave_vote_share'))


def fetch_population_by_authority():
    """First download data from https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/wardlevelmidyearpopulationestimatesexperimental
    and save as data/data_in/population_estimates.xls"
    """
    path = "../data/data_in/population_estimates.xls"
    population_by_ward = (
        pd.read_excel(path, sheet_name="Mid-2016 Persons", skiprows=4, header=0)
        .assign(authority=lambda df: df['Local Authority'].str.lower())
        .drop('Local Authority', axis=1)
    )
    return (population_by_ward
            .groupby('authority')['All Ages'].sum()
            .astype(int)
            .rename('population'))
