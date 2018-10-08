import pandas as pd


def proportionise(df: pd.DataFrame):
    return df.div(df.sum(axis=1), axis=0)


def get_vote_share_change(vote_shares: pd.DataFrame,
                          start=2014, end=2018):
    """Turn dataframe of vote shares by year into dataframe of vote share changes"""
    both = (vote_shares
            .loc[start:end]
            .unstack(level=0)
            .dropna()
            .stack()
            .swaplevel()
            .sort_index())
    return (both.loc[2018] - both.loc[2014]).add_suffix("_change")


def stanify_series(s):
    return pd.Series(s.factorize()[0] + 1, index=s.index)
