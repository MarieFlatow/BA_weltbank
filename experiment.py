import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

def reset_base():
    base = pd.read_csv('additional_data/base.csv')
    base.set_index(['Country Name', 'Indicator Name'], inplace=True)
    base = base.sort_index(level=['Country Name', 'Indicator Name'])
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


def reset_train(cords):
    train = reset_base()
    for i in cords.index:
        train.iloc[cords[0][i], cords[1][i]] = None
    return train


def evaluate(df, t, cords):
    # scaling original data and imputed data
    train = reset_train(cords)
    scaler = StandardScaler().fit(train)  # fitting on train?
    norm_base = pd.DataFrame(scaler.transform(base))
    df = pd.DataFrame(scaler.transform(df))

    # getting imputed values for simulated NaNs and true value
    res = pd.DataFrame({'y_true': [norm_base.iloc[cords[0][i], cords[1][i]] for i in cords.index],
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
