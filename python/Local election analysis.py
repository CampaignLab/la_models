
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import pystan
from matplotlib import pyplot as plt
plt.style.use('fr_sparse')


# ## Vote shares by authority in the 2018, 2014, 2012 and 2011 local elections

# In[2]:

def proportionise(df):
    return df.div(df.sum(axis=1), axis=0)

url_2018 = "https://candidates.democracyclub.org.uk/media/csv-archives/results-2018-05-03.csv"
at_url = "http://www.andrewteale.me.uk/pdf/{0}/{0}_results.zip"
at_colnames = ['ward', 'council', 'candidate', 'party_at', 'votes', 'elected']
dc_ballot_colnames = ['election_type', 'council', 'ward', 'date', '???']
final_colnames = ['council', 'ward', 'year', 'party', 'votes']

at_party_names = {
    'Lab': 'labour',
    'C': 'conservative',
    'LD': 'libdem',
    'UKIP': 'ukip',
    'Grn': 'green'
}
dc_party_names = {
    'Labour Party': 'labour',
    'Labour and Co-operative Party': 'labour',
    'Conservative and Unionist Party': 'conservative',
    'Liberal Democrats': 'libdem',
    'UK Independence Party (UKIP)': 'ukip',
    'Green Party': 'green'
}

results_2018 = pd.read_csv(url_2018, header=1).assign(year=2018)
info_2018 = pd.DataFrame.from_records(results_2018['ballot_paper_id'].str.split('.').tolist())
info_2018.columns = dc_ballot_colnames
info_2018['council'] = info_2018['council'].str.replace('-', ' ')
info_2018['ward'] = info_2018['ward'].str.replace('-', ' ')

results_2018 = results_2018.join(info_2018).loc[lambda df: df['election_type'] == 'local']
results_2018['party'] = results_2018['party_name'].map(dc_party_names).fillna('other')
results_2018['votes'] = results_2018['ballots_cast'].astype(int)
results_2018['candidate'] = results_2018['person_name']

results_at = pd.concat([
    pd.read_csv(at_url.format(str(year)), header=None, names=at_colnames, skiprows=None if year != 2014 else 1)
    .assign(year=year,
            party=lambda df: df['party_at'].map(at_party_names).fillna('other'),
            ward=lambda df: df['ward'].str.lower(),
            council=lambda df: df['council'].str.lower(),
            votes=lambda df: df['votes'].astype(int))
    for year in [2011, 2012, 2014]
])

results = pd.concat([results_at[final_colnames], results_2018[final_colnames]])
vote_shares = results.groupby(['year', 'council', 'party'])['votes'].sum().unstack().pipe(proportionise).fillna(0)
vote_shares['lab_2p'] = vote_shares['labour'] / (vote_shares['labour'] + vote_shares['conservative'])

vote_shares.head()


# In[3]:

vote_shares.reset_index().to_csv("~/Downloads/vote_shares.csv")


# ## Change in vote share 2014-2018 by party (where available)

# In[4]:

both = vote_shares.loc[2014:2018].unstack(level=0).dropna().stack().swaplevel().sort_index()
vote_share_change = (both.loc[2018] - both.loc[2014]).add_suffix("_change")
display(vote_share_change.sort_values('lab_2p_change', ascending=False).head())
hist = plt.hist(vote_share_change['lab_2p_change'], bins=15)
plt.xlabel("Change in Labour share of Lab/Con vote 2014-2018")
plt.ylabel("Frequency")


# In[5]:

vote_share_change.to_csv("~/Downloads/vote_share_change_2014_2018.csv")


# ## Median annual pay 2012-2017

# In[6]:

earnings = (
    # download the very big csv from here: https://download.beta.ons.gov.uk/downloads/datasets/ashe-table-8-earnings/editions/time-series/versions/1.csv
    # I got a 'permission denied' message from the ons website when I tried to read it with `read_excel` :(
    pd.read_csv("../data/data_in/earning_data.csv")
    .loc[lambda df: (df['Geography'].str.lower().isin(vote_share_change.index))
         & (df['Sex_codelist'] == 'all')
         & (df['Statistics_codelist'] == 'median')
         & (df['Workingpattern_codelist'] == 'all')
         & (df['Earnings'] == 'Annual pay - Gross')]
    .assign(authority=lambda df: df['Geography'].str.lower())
    .set_index(['authority', 'Time'])['V4_2']
    .unstack()
)
earnings.sort_values(2017, ascending=True).head()


# In[7]:

earnings.to_csv("~/Downloads/median_annual_pay.csv")


# ## Map of local authorities to regions
# 
# Useful for hierarchical models...

# In[8]:

region_lookup_url = "http://geoportal1-ons.opendata.arcgis.com/datasets/c457af6314f24b20bb5de8fe41e05898_0.csv"
authority_to_region = (
    pd.read_csv(region_lookup_url)
    .assign(authority=lambda df: df['LAD17NM'].str.lower(),
            region=lambda df: df['RGN17NM'].str.lower())
    .set_index('authority')
    ['region']
)
authority_to_region.head()


# In[9]:

authority_to_region.reset_index().to_csv("~/Downloads/authority_to_region.csv")


# ## Housing waiting lists 2010 vs 2017

# In[10]:

waiting_list_url = "https://www.gov.uk/government/uploads/system/uploads/attachment_data/file/674350/LT_600.xlsx"

waiting_list = (
    pd.read_excel(waiting_list_url, header=0, skiprows=3, usecols=[4, 19, 26])
    .dropna()
    .rename(columns={'Lower and Single Tier Authority Data': 'authority'})
    .assign(authority=lambda df: df['authority'].str.lower())
    .set_index('authority')
    .replace('..', np.nan)
    .astype(float)
    .assign(change=lambda df: df[2017] - df[2010],
            ratio=lambda df: df[2017] / df[2010])

)


# In[11]:

waiting_list.reset_index().to_csv("~/Downloads/housing_waiting_lists_2010_2017")


# ## Leave vote share in the referendum

# In[12]:

eu_ref_url = "https://www.electoralcommission.org.uk/__data/assets/file/0014/212135/EU-referendum-result-data.csv" 

leave_vote_share = (
    pd.read_csv(eu_ref_url)
    .assign(authority=lambda df: df['Area'].str.lower())
    .groupby('authority')['Pct_Leave'].first()
    .rename('leave_vote_share')
)


# In[13]:

leave_vote_share.reset_index().to_csv("~/Downloads/leave_vote_share_by_authority.csv")


# ## 2016 population estimates by local authority 
# 
# These might be useful for scaling some of the other statistics.

# In[14]:

# Another access denied - has to be downloaded from here: https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/wardlevelmidyearpopulationestimatesexperimental
population_sheet_path = "../data/data_in/population_estimates.xls"

population_by_ward = (pd.read_excel(population_sheet_path, sheet_name="Mid-2016 Persons", skiprows=4, header=0)
                      .assign(authority=lambda df: df['Local Authority'].str.lower())
                      .drop('Local Authority', axis=1))

total_population_by_authority = (population_by_ward
                                 .groupby('authority')['All Ages'].sum()
                                 .astype(int)
                                 .rename('population'))
total_population_by_authority.sort_values(ascending=False).head()


# In[15]:

total_population_by_authority.reset_index().to_csv("~/Downloads/population_estimate_2016_by_local_authority.csv")


# In[16]:

def stanify_series(s):
    return pd.Series(s.factorize()[0] + 1, index=s.index)

# design matrix
dm = (
    vote_share_change
    .join(waiting_list.add_prefix('waiting_list_'), how='left')
    .join(total_population_by_authority, how='left')
    .join(earnings.add_prefix('median_income_'), how='left')
    .join(authority_to_region)
    .join(leave_vote_share)
    .assign(region_stan=lambda df: stanify_series(df['region']))
    .dropna()
)
dm.head()


# In[17]:

f, axes = plt.subplots(3, 2, figsize=[10, 8], sharey=True, sharex=True)
axes = axes.ravel()
x_col = 'leave_vote_share'

f.suptitle("Change in local election vote share 2014-2018  vs {}".format(x_col.replace('_', ' ')))

for ax, y_col in zip(axes, ['green_change', 'libdem_change', 'conservative_change', 
                            'labour_change', 'ukip_change', 'lab_2p_change']):
    ax.scatter(dm[x_col], dm[y_col], s=dm['population']/10000)
    extremes = (list(dm.sort_values(y_col)[y_col].dropna().iloc[np.r_[0, -1]].index)
                + list(dm.sort_values(x_col)[x_col].dropna().iloc[np.r_[0, -1]].index))
    for i, r in dm.loc[extremes].iterrows():
        if all(np.isfinite(dm.loc[i, [x_col, y_col]].astype(float))):
            ax.text(dm.loc[i, x_col], dm.loc[i, y_col], i.capitalize(), fontsize=8)

    ax.set_title(y_col.capitalize(), y=0.8)
    if ax in [axes[0], axes[2], axes[4]]:
        ax.set_ylabel("Change in vote share".format(y_col))
axes[4].set_xlabel("{}".format(x_col))
axes[5].set_xlabel("{}".format(x_col))
# plt.tight_layout()


# In[18]:

t = vote_share_change.join(earnings[2017].rename('earnings')).join(total_population_by_authority)

plt.scatter(t['earnings'], t['lab_2p_change'], s=t['population']/10000)
extremes = (list(t.sort_values('earnings').dropna().iloc[np.r_[0:4, -4:0]].index)
            + list(t.sort_values('lab_2p_change').dropna().iloc[np.r_[0:4, -4:0]].index))
for i, r in t.loc[extremes].iterrows():
    if all(np.isfinite(t.loc[i, ['earnings', 'lab_2p_change']].astype(float))):
        plt.text(t.loc[i, 'earnings'], t.loc[i, 'lab_2p_change'], i, fontsize=8)


plt.xlabel("Median annual pay")
plt.ylabel("Change in labour vote share")


# In[19]:

t = vote_share_change.join(waiting_list).join(total_population_by_authority)

plt.scatter(t['ratio'], t['labour_change'], s=t['population']/10000)

extremes = (list(t.sort_values('ratio').iloc[np.r_[0:4, -4:0]].index)
            + list(t.sort_values('labour_change').iloc[np.r_[0:4, -4:0]].index))
for i, r in t.loc[extremes].iterrows():
    if all(np.isfinite(t.loc[i, ['ratio', 'labour_change']])):
        plt.text(t.loc[i, 'ratio'], t.loc[i, 'labour_change'], i, fontsize=8)

plt.semilogx()
plt.xlabel("Housing waitlist 2010 / housing waitlist 2017 (log scale)")
plt.ylabel("Change in labour vote share")


# In[20]:

model = pystan.StanModel(file="../stan/model.stan")


# In[21]:

predictors = ['waiting_list_ratio', 'median_income_2017', 'leave_vote_share']
model_input = {
    'N': len(dm),
    'M': len(predictors),
    'R': dm['region'].nunique(),
    'x': dm[predictors].astype(float),
    'region': dm['region_stan'],
    'y': dm['lab_2p_change']
}
fit = model.sampling(data=model_input, control={'adapt_delta':0.99})
samples = fit.to_dataframe()
samples.head()


# In[22]:

print(fit.stansummary(pars=[p for p in fit.model_pars 
                            if p not in ['log_lik', 'y_tilde']
                            and 'z' not in p]))


# In[23]:

region_stan_to_region = dm.groupby('region_stan')['region'].first()


# In[24]:

b_samples = samples[[c for c in samples.columns if c[:2] == 'b[']].copy()
l = list(map(lambda s: s.strip('b[').strip(']').split(','), b_samples.columns))
b_samples.columns = pd.MultiIndex.from_tuples(
    [(predictors[int(i) - 1], region_stan_to_region.loc[int(j)]) for [i, j] in l]
)
print("Posterior mean effects by region:")
mean_regression_effects = b_samples.sort_index(axis=1).mean().unstack().T

mean_regression_effects.style.highlight_min(axis=0)


# In[25]:

output = dm.copy()
ppc_samples = samples[[c for c in samples.columns if 'y_tilde' in c]].copy()
log_lik_samples = samples[[c for c in samples.columns if 'log_lik' in c]].copy()
output['log_likelihood_mean'] = log_lik_samples.mean().values
output['ppc_lower'] = ppc_samples.quantile(0.1).values
output['ppc_mean'] = ppc_samples.mean().values
output['ppc_upper'] = ppc_samples.quantile(0.9).values
log_lik_samples.mean().mean()


# In[26]:

f, ax = plt.subplots(figsize=[12, 8])

x_col = 'leave_vote_share'

ax.scatter(output[x_col], output['lab_2p_change'], label='observation')
ax.vlines(output[x_col], output['ppc_lower'], output['ppc_upper'], 
          color='tab:orange', zorder=0, alpha=0.6, label='posterior predictive quantiles 10%-90%')
ax.axhline(0, color='r', linestyle='--', alpha=0.5)

extremes = list(output.sort_values('log_likelihood_mean')[:8].index)
for i, r in output.loc[extremes].iterrows():
    if all(np.isfinite(output.loc[i, [x_col, 'lab_2p_change']].astype(float))):
        plt.text(output.loc[i, x_col], output.loc[i, 'lab_2p_change'], i.capitalize(), fontsize=8)


ax.legend(frameon=False)
ax.set_xlabel('Leave vote share in the 2016 referendum')
ax.set_ylabel('Change in Labour share of Lab/Con vote from 2014 to 2018')
ax.set_title('Labour did worse vs Tories in leave-voting areas\n (Except Adur and Worthing - what happened there??)', fontsize=14)


# In[27]:

f, axes = plt.subplots(3, 3, figsize=[12, 8], sharex=True, sharey=True)
f.suptitle("The effect was similar across the country", fontsize=14)

axes = axes.ravel()
x_col = 'leave_vote_share'

for ax, (reg, df) in zip(axes, output.groupby('region')):
    ax.scatter(df[x_col], df['lab_2p_change'])
    ax.vlines(df[x_col], df['ppc_lower'], df['ppc_upper'], color='tab:orange', zorder=0, alpha=0.6)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)

    if ax == axes[3]:
        ax.set_ylabel('Change in Labour share of Lab/Con vote')
    if ax == axes[7]:
        ax.set_xlabel('Leave vote share in the 2016 referendum')
    ax.set_title(reg.capitalize(), y=0.8)


# In[28]:

output.loc[lambda df: df['region'] == 'london'].sort_values('leave_vote_share')


# In[29]:

f, ax = plt.subplots(figsize=[12, 8])

x_col = 'waiting_list_ratio'

ax.scatter(output[x_col], output['lab_2p_change'], label='observation')
ax.vlines(output[x_col], output['ppc_lower'], output['ppc_upper'], 
          color='tab:orange', zorder=0, alpha=0.6, label='posterior predictive quantiles 10%-90%')
ax.axhline(0, color='r', linestyle='--', alpha=0.5)

extremes = list(output.sort_values('log_likelihood_mean')[:8].index)
for i, r in output.loc[extremes].iterrows():
    if all(np.isfinite(output.loc[i, [x_col, 'lab_2p_change']].astype(float))):
        plt.text(output.loc[i, x_col], output.loc[i, 'lab_2p_change'], i.capitalize(), fontsize=8)


ax.legend(frameon=False)
ax.set_xlabel('People waiting for housing in 2017 / people waiting in 2010')
ax.set_ylabel('Change in Labour share of Lab/Con vote from 2014 to 2018')
ax.set_title("Changes in local housing waiting lists don't seem to predict changes\n"
             "in Conservative/Labour vote share at the national level...", fontsize=14)



# In[30]:

f, axes = plt.subplots(3, 3, figsize=[12, 8], sharex=True, sharey=True)
f.suptitle("...But there are regional patterns!", fontsize=14)

axes = axes.ravel()
x_col = 'waiting_list_ratio'

for ax, (reg, df) in zip(axes, output.groupby('region')):
    ax.scatter(df[x_col], df['lab_2p_change'])
    ax.vlines(df[x_col], df['ppc_lower'], df['ppc_upper'], color='tab:orange', zorder=0, alpha=0.6)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)

    ax.semilogx()

    if ax == axes[3]:
        ax.set_ylabel('Change in Labour share of Lab/Con vote')
    if ax == axes[7]:
        ax.set_xlabel('People waiting for housing in 2017 / people waiting in 2010 (log scale)')
    ax.set_title(reg.capitalize(), y=0.8)


# In[31]:

f, ax = plt.subplots(figsize=[12, 8])

x_col = 'median_income_2017'

ax.scatter(output[x_col], output['lab_2p_change'], label='observation')
ax.vlines(output[x_col], output['ppc_lower'], output['ppc_upper'], 
          color='tab:orange', zorder=0, alpha=0.6, label='posterior predictive quantiles 10%-90%')
ax.axhline(0, color='r', linestyle='--', alpha=0.5)

extremes = list(output.sort_values('log_likelihood_mean')[:8].index)
for i, r in output.loc[extremes].iterrows():
    if all(np.isfinite(output.loc[i, [x_col, 'lab_2p_change']].astype(float))):
        plt.text(output.loc[i, x_col], output.loc[i, 'lab_2p_change'], i.capitalize(), fontsize=8)


ax.legend(frameon=False)
ax.set_xlabel('Median annual income in pounds')
ax.set_ylabel('Change in Labour share of Lab/Con vote from 2014 to 2018')
ax.set_title("Labour tended to get bigger swings from the Tories in richer areas", fontsize=14)



# In[32]:

f, axes = plt.subplots(3, 3, figsize=[12, 8], sharex=True, sharey=True)
f.suptitle("...the size of the effect varied by region", fontsize=14)

axes = axes.ravel()
x_col = 'median_income_2017'

for ax, (reg, df) in zip(axes, output.groupby('region')):
    ax.scatter(df[x_col], df['lab_2p_change'])
    ax.vlines(df[x_col], df['ppc_lower'], df['ppc_upper'], color='tab:orange', zorder=0, alpha=0.6)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    if ax == axes[3]:
        ax.set_ylabel('Change in Labour share of Lab/Con vote')
    if ax == axes[7]:
        ax.set_xlabel('Median annual income in pounds')
    ax.set_title(reg.capitalize(), y=0.8)

