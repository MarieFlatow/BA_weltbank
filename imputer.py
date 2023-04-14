import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

def impute_backfill(df):
    df = df.fillna(method='bfill', limit=3, axis=1)
    return df


def impute_overall_means(df):
    # fill NaNs with overall mean of that indicator
    values = pd.DataFrame(df.stack()).groupby('Indicator Name')[0].mean()
    df = pd.DataFrame(df.stack(dropna=False))

    df[0] = df[0].fillna(df.groupby('Indicator Name')[0].transform('mean'))
    df = df.unstack()
    df.columns = df.columns.droplevel(0)
    df = df.sort_index(level=['Country Name', 'Indicator Name'])

    return df


def impute_yearly_means(df):
    # fill NaNs with overall mean of that indicator

    for i in df.columns:
        df[i] = df[i].fillna(df.groupby('Indicator Name')[i].transform('mean'))

    return df


def impute_yearly_means_per_region(df):
    country_data = pd.read_csv('../Data/WDICountry.csv')
    country_data = country_data.loc[:, ['Table Name', 'Region']]
    df = pd.merge(df.reset_index(), country_data, how='left', left_on='Country Name', right_on='Table Name').drop(
        'Table Name', axis=1)
    df = df.set_index(['Country Name', 'Indicator Name', 'Region'])

    for i in df.columns:
        df[i] = df[i].fillna(df.groupby(['Indicator Name', 'Region'])[i].transform('mean'))

    df = df.reset_index().set_index(['Country Name', 'Indicator Name']).drop('Region', axis=1)
    return df


def interpolate3(df):
    df = df.interpolate(limit=3, axis=1)
    return df


def interpolate_all(df):
    df = df.interpolate(axis=1)
    return df


def iterative_imputer1(df):
    col = df.columns
    idx = df.index

    iter_imp = IterativeImputer(random_state=999)
    df = iter_imp.fit_transform(df)
    df = pd.DataFrame(df, columns=col, index=idx)
    return df


def iterative_imputer2(df):
    df = df.unstack().T
    col = df.columns
    idx = df.index

    iter_imp = IterativeImputer(random_state=999)
    df = iter_imp.fit_transform(df)

    df = pd.DataFrame(df, columns=col, index=idx)
    df = df.unstack().T
    df = df.sort_index(level=['Country Name', 'Indicator Name'])

    return df


def iterative_imputer3(df):
    df = df.reset_index()
    df = df.set_index(['Indicator Name', 'Country Name'])
    df = df.unstack().T

    col = df.columns
    idx = df.index

    iter_imp = IterativeImputer(random_state=999, verbose=True)
    df = iter_imp.fit_transform(df)

    df = pd.DataFrame(df, columns=col, index=idx)
    df = df.unstack().T
    df = df.reset_index()
    df = df.set_index(['Country Name', 'Indicator Name'])
    df = df.sort_index(level=['Country Name', 'Indicator Name'])

    return df


def mice_imputer(df, verbose=2):
    n_imputations = 12
    dfs = []
    col = df.columns
    idx = df.index

    for i in range(n_imputations):
        print(f'Imputation round {i+1}/{n_imputations}')
        iter_imp = IterativeImputer(random_state=i, sample_posterior=True, verbose=verbose)
        df_temp = iter_imp.fit_transform(df)
        dfs.append(df_temp)

    df = np.mean(np.array(dfs), axis=0)
    df = pd.DataFrame(df, columns=col, index=idx)
    return df


def mice_imputer2(df, detailed=False, verbose=2):
    n_imputations = 5
    dfs = []

    df = df.reset_index()
    df = df.set_index(['Indicator Name', 'Country Name'])
    df = df.unstack().T

    col = df.columns
    idx = df.index

    for i in range(n_imputations):
        print(f'Imputation round {i + 1}/{n_imputations}')
        iter_imp = IterativeImputer(random_state=i + 200, sample_posterior=True, verbose=verbose)
        df_temp = iter_imp.fit_transform(df)
        dfs.append(df_temp)

    df = np.mean(np.array(dfs), axis=0)
    df = pd.DataFrame(df, columns=col, index=idx)
    df = df.unstack().T
    df = df.reset_index()
    df = df.set_index(['Country Name', 'Indicator Name'])
    df = df.sort_index(level=['Country Name', 'Indicator Name'])
    if detailed:
        return dfs, df
    else:
        return df


def knn_imputer1(df, n=16):
    col = df.columns
    idx = df.index

    knn_imp = KNNImputer(n_neighbors=n)
    df = knn_imp.fit_transform(df)
    df = pd.DataFrame(df, columns=col, index=idx)
    return df


def knn_imputer2(df, n=3):
    df = df.reset_index()
    df = df.set_index(['Indicator Name', 'Country Name'])
    df = df.unstack().T

    col = df.columns
    idx = df.index

    knn_imp = KNNImputer(n_neighbors=n)
    df = knn_imp.fit_transform(df)
    df = pd.DataFrame(df, columns=col, index=idx)

    df = df.unstack().T
    df = df.reset_index()
    df = df.set_index(['Country Name', 'Indicator Name'])
    df = df.sort_index(level=['Country Name', 'Indicator Name'])

    return df