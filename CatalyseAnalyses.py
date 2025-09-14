import numpy as np
import pandas as pd

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sklearn
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VARMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import itertools
import warnings
from statsmodels.tsa.stattools import ccf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from random import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
#from sklearn.inspection import plot_partial_dependence
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from docx import Document
import os

def normalise(df):
    df1 = df.drop(["date"], axis=1)
    df1.fillna(0, inplace=True)
    df_normalized = (df1 - df1.min()) / (df1.max() - df1.min())
    df_normalized["week"] = list(range(0,df_normalized.shape[0]))
    return df_normalized

def plotSeries(df, opf=None):
    ax = df.drop(["week"], axis=1).plot()
    ax.set_xticks(df.index[::26])  # Show every tick, adjust [::2], [::3] for spacing
    for tick in df.index[::26]:  # adjust slice to control frequency
        ax.axvline(x=tick, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if opf:
        plt.savefig(os.path.join(opf, "time_series.png"), dpi=300)
    else:
        plt.show()

def spearmanrr(x, y, window=104, n_boot=10, ci=95):
    #print("hallo")
    """
    Returns rolling Spearman correlation + lower/upper confidence bands.
    """
    rolling_rho = []
    lower = []
    upper = []

    for i in range(window - 1, len(x)):
        #print(i)
        x_win = x.iloc[i - window + 1 : i + 1].values
        y_win = y.iloc[i - window + 1 : i + 1].values
        
        # Compute rolling Spearman
        rho = spearmanr(x_win, y_win).correlation
        rolling_rho.append(rho)
        
        # Bootstrap
        boot_rho = []
        for _ in range(n_boot):
            idx = np.random.choice(range(window), size=window, replace=True)
            boot_rho.append(spearmanr(x_win[idx], y_win[idx]).correlation)
        lower.append(np.percentile(boot_rho, (100 - ci) / 2))
        upper.append(np.percentile(boot_rho, 100 - (100 - ci) / 2))
    
    index = x.index[window - 1 :]
    return pd.DataFrame({
        'rho': rolling_rho,
        'lower': lower,
        'upper': upper
    }, index=index)

def findCorrs(df, opf=None):
    
    sc = spearmanr(df.incidence, df.gr)
    #print(sc)

    sdf = spearmanrr(df.incidence, df.gr, window=52, n_boot=10, ci=95)
    plt.figure(figsize=(12, 8))

    plt.plot(sdf.index, sdf['rho'], color='blue', label='ρ')
    plt.fill_between(sdf.index, sdf['lower'], sdf['upper'], color='blue', alpha=0.2)

    plt.title("Rolling Spearman correlation")
    plt.grid(True)
    plt.tight_layout()
    if opf:
        plt.savefig(os.path.join(opf, "rolling_spearman.png"), dpi=300)
    else:
        plt.show()

    xcor_ig = ccf(df.incidence, df.gr)
    # Plot the cross-correlation
    lags = np.arange(len(xcor_ig))
    plt.figure(figsize=(12, 8))
    #plt.stem(lags, xcor_ig, use_line_collection=True)
    plt.scatter(lags, xcor_ig, marker=".")
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Box-Jenkins Cross-Correlation Function with Lags')
    plt.xlabel('Lag')
    plt.ylabel('Cross-Correlation')
    if opf:
        plt.savefig(os.path.join(opf, "cross_correlation.png"), dpi=300)
    else:
        plt.show()
    #print(lags[np.argmax(xcor_ig)])

    return sc


def create_lagged_features(data, lags=4):
    df_lag = data.copy()
    for col in df.columns:
        for lag in range(1, lags + 1):
            df_lag[f'{col}_lag{lag}'] = df[col].shift(lag)
    df_lag.dropna(inplace=True)
    return df_lag

def create_forecast_frame(df, h=4):
    df_lag = df.copy()
    for i in range(1, h + 1):
        df_lag[f'{target}_t+i={i}'] = df[target].shift(-i)
    df_lag = df_lag.dropna()
    return df_lag

def create_horizon_targets(df, target_col, max_horizon):
    df = df.copy()
    for h in range(0, max_horizon + 1):
        df[f'{target_col}_t+{h}'] = df[target_col].shift(-h)
    return df.dropna()

def tune_model(model, X, y):
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(model, param_grid={'n_estimators': [50, 100]}, cv=tscv)
    grid.fit(X, y)
    return grid.best_estimator_

def tune_enet(X_train, y_train):
    """
    Tune Elastic Net using GridSearchCV with a pipeline that includes scaling.
    """
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('enet', ElasticNet(max_iter=10000, tol=1e-2, random_state=42))
    ])
    
    param_grid = {
        'enet__alpha': [0.001, 0.01, 0.1, 1, 10],
        'enet__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
    }
    
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    return grid.best_estimator_

def rolling_origin_week_based(df, target='target', week_col='week', max_horizon=4, opf=None):
    results = []
    df = create_horizon_targets(df, target, max_horizon)
    exog_cols = ["gr"]

    # Define rolling train/test week bounds
    splits = [
        (0, 51, 52, 103),    # Train: 0–51, Test: 52–103
        (0, 103, 104, 155),  # Train: 0–103, Test: 104–147
        (0, 155, 156, 207),
        (0, 207, 208, 259),
        (0, 259, 260, 311),
        (0, 311, 312, 363),
        (0, 363, 364, 397)
    ]

    splits = []
    ly = 104
    while ly < df.shape[0]:
        splits.append((0, ly-52, ly-51, ly))
        ly += 52
    splits.append((0, ly-52, ly-51, df.shape[0]-1))

    for train_start, train_end, test_start, test_end in splits:
        print(train_end)
        train = df[(df[week_col] >= train_start) & (df[week_col] <= train_end)]
        test = df[(df[week_col] >= test_start) & (df[week_col] <= test_end)]

        for h in range(0, max_horizon + 1):
            #print((train_start, train_end, h))
            y_train = train[f'{target}_t+{h}']
            y_test = test[f'{target}_t+{h}']
            X_train = train[exog_cols]
            X_test = test[exog_cols]

            # Seasonal Naive: same week last year (week - 52)
            #print("running snaive")
            snaive_weeks = test[week_col] - 52
            snaive = df[df[week_col].isin(snaive_weeks)][target].values
            if len(snaive) != len(y_test):
                snaive = np.full_like(y_test, fill_value=np.nan)
            snaive_pred = snaive

            # SARIMAX
            #print("running sarimax")
            sarimax = SARIMAX(endog=y_train, exog=X_train, order=(1,0,0), seasonal_order=(1,0,0,52))
            #sarimax = tune_sarimax(y_train, X_train)
            sarimax_fit = sarimax.fit(disp=False)
            sarimax_pred = sarimax_fit.forecast(steps=len(X_test), exog=X_test)

            # Random Forest
            #print("running rfr")
            rf = tune_model(RandomForestRegressor(random_state=42), X_train, y_train)
            rf_pred = rf.predict(X_test)

            # XGBoost
            #print("running xgboost")
            xgb = tune_model(XGBRegressor(random_state=42, verbosity=0), X_train, y_train)
            xgb_pred = xgb.predict(X_test)
            
            # Elastic Net
            #print("running elastic net")
            enet = tune_enet(X_train, y_train)
            enet_pred = enet.predict(X_test)

            # Store results
            for name, pred in zip(['SNaive', 'SARIMAX', 'RF', 'XGB', 'ENet'],
                                  [snaive_pred, sarimax_pred, rf_pred, xgb_pred, enet_pred]):
                valid = ~np.isnan(pred)
                results.append({
                    'train_weeks': f'{train_start}-{train_end}',
                    'test_weeks': f'{test_start}-{test_end}',
                    'horizon': h,
                    'model': name,
                    'mae': mean_absolute_error(y_test[valid], pred[valid]),
                    'mse': mean_squared_error(y_test[valid], pred[valid], squared=True),
                    'rmse': mean_squared_error(y_test[valid], pred[valid], squared=False),
                    'nrmse': mean_squared_error(y_test[valid], pred[valid], squared=False)/np.mean(y_test[valid]),
                    'mase': mean_absolute_error(y_test[valid], pred[valid])/mean_absolute_error(y_test[valid], snaive_pred[valid])
                    
                })
    rdf = pd.DataFrame(results)
    if opf:
        rdf.to_csv(os.path.join(opf, "results.csv"))
    else:
        return rdf



