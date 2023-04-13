import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

def reset_base(scaled=True, return_scaler=False):
    base = pd.read_csv('additional_data/base.csv')
    base.set_index(['Country Name', 'Indicator Name'], inplace=True)
    base = base.sort_index(level=['Country Name', 'Indicator Name'])
    col = base.columns
    idx = base.index

    # scaling data
    if scaled:
        base = base.reset_index()
        base = base.set_index(['Indicator Name', 'Country Name'])
        base = base.unstack().T

        col = base.columns
        idx = base.index

        scaler = StandardScaler().fit(base)
        base = scaler.transform(base)

        base = pd.DataFrame(base, columns=col, index=idx)
        base = base.unstack().T
        base = base.reset_index()
        base = base.set_index(['Country Name', 'Indicator Name'])
        base = base.sort_index(level=['Country Name', 'Indicator Name'])


    if return_scaler:
        return base, scaler
    else:
        return base


base = reset_base()


def get_cords(frac, rnd_state=999):
    n = int(base.isna().sum().sum() * frac)
    print(f'Testdaten mit {frac * 100}% fehlenden Werten (absolut: {n})')
    # random state to ensure reproducibility
    rnds = np.random.RandomState(rnd_state)

    # coordinates for data entries to be removed randomly
    # 5000 entries are selected
    cords = pd.DataFrame([[rnds.randint(0, len(base), size=n * 4)[i],
                           rnds.randint(0, len(base.columns), size=n * 4)[i]]
                          for i in range(n * 4)])

    # all coordinates pointing to NaN entries are removed and
    # first 1000 remaining entries are selected
    cords['value'] = [base.iloc[cords[0][i], cords[1][i]] for i in cords.index]
    cords = cords.dropna()[:n].reset_index(drop=True)

    return cords


def reset_train(cords, scaled=True):
    if scaled:
        train = reset_base(scaled=True)
    else:
        train = reset_base(scaled=False)


    for i in cords.index:
        train.iloc[cords[0][i], cords[1][i]] = None
    return train


def evaluate(df, t, cords):

    # getting imputed values for simulated NaNs and true value
    res = pd.DataFrame({'y_true': [base.iloc[cords[0][i], cords[1][i]] for i in cords.index],
                        'y_pred': [df.iloc[cords[0][i], cords[1][i]] for i in cords.index]
                        })
    res = res.dropna()

    # calculate evaluation metrics
    r2 = r2_score(res['y_true'], res['y_pred'])
    rmse = math.sqrt(mean_squared_error(res['y_true'], res['y_pred']))
    still_missing = df.isna().sum().sum()

    print(f'r2: {r2}, rmse: {rmse}, t: {t}')
    print('')

    return [r2, rmse, still_missing, t]

def rescale(df, scaler):
    df = df.reset_index()
    df = df.set_index(['Indicator Name', 'Country Name'])
    df = df.unstack().T

    col = df.columns
    idx = df.index

    df = scaler.inverse_transform(df)

    df = pd.DataFrame(df, columns=col, index=idx)
    df = df.unstack().T
    df = df.reset_index()
    df = df.set_index(['Country Name', 'Indicator Name'])
    df = df.sort_index(level=['Country Name', 'Indicator Name'])

    return df

def load_imputed_sets(folder):
    backfill = pd.read_csv(f'additional_data/imputed_sets/{folder}/backfill.csv')
    yearly_mean = pd.read_csv(f'additional_data/imputed_sets/{folder}/Yearly Mean.csv')
    overall_mean = pd.read_csv(f'additional_data/imputed_sets/{folder}/Overall Mean.csv')
    regional_mean = pd.read_csv(f'additional_data/imputed_sets/{folder}/Yearly Mean per Region.csv')
    interpolation_all = pd.read_csv(f'additional_data/imputed_sets/{folder}/Interpolate all.csv')
    interpolation_3 = pd.read_csv(f'additional_data/imputed_sets/{folder}/Interpolate 3.csv')
    ice1 = pd.read_csv(f'additional_data/imputed_sets/{folder}/ICE 1.csv')
    ice2 = pd.read_csv(f'additional_data/imputed_sets/{folder}/ICE 2.csv')
    ice3 = pd.read_csv(f'additional_data/imputed_sets/{folder}/ICE 3.csv')
    mice1 = pd.read_csv(f'additional_data/imputed_sets/{folder}/MICE 1.csv')
    mice2 = pd.read_csv(f'additional_data/imputed_sets/{folder}/MICE 2.csv')
    knn1 = pd.read_csv(f'additional_data/imputed_sets/{folder}/KNN 1.csv')
    knn2 = pd.read_csv(f'additional_data/imputed_sets/{folder}/KNN 2.csv')

    sets = { 'Overall Mean': overall_mean, 'Yearly Mean': yearly_mean, 
            'Yearly Mean per Region': regional_mean, 'Backfill': backfill,
            'Interpolate': interpolation_all,
            'MICE 1': mice1, 'MICE 2': mice2,
            'KNN 1': knn1, 'KNN 2': knn2
            }
    return sets

