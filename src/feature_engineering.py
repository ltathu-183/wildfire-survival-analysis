import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

FEATURE_COLUMNS = [
    'log_fire_gravity',
    'dist_min_ci_0_5h',
    'is_high_gravity_danger',
    'fire_volatility',
    'alignment_abs',
    'gravity_momentum'
]


def master_oracle_engineering(data: pd.DataFrame) -> pd.DataFrame:
    df_e = data.copy()

    df_e['fire_gravity'] = np.log1p(df_e['area_first_ha']) / (df_e['dist_min_ci_0_5h'] + 1e-5)
    df_e['log_fire_gravity'] = np.log1p(df_e['fire_gravity'] * 1000)

    rad = np.deg2rad(df_e['alignment_abs'])
    path_complexity = df_e['dist_std_ci_0_5h'] * (1 + rad)
    df_e['fire_volatility'] = np.log1p(path_complexity)

    q3_gravity = df_e['log_fire_gravity'].quantile(0.75)
    df_e['is_high_gravity_danger'] = (df_e['log_fire_gravity'] >= q3_gravity).astype(int)

    df_e['gravity_momentum'] = df_e['log_fire_gravity'] * df_e['along_track_speed']
    df_e['is_afternoon'] = df_e['event_start_hour'].apply(lambda x: 1 if 12 <= x <= 17 else 0)

    return df_e


def prepare_training_data(df: pd.DataFrame, features: list = FEATURE_COLUMNS):
    df_final = master_oracle_engineering(df)
    X = df_final[features].astype(float)
    y_struct = np.array([
        (bool(e), t) for e, t in zip(df_final['event'], df_final['time_to_hit_hours'])
    ], dtype=[('event', 'bool'), ('time', 'float')])
    return df_final, X, y_struct


def scale_features(X: pd.DataFrame, scaler: RobustScaler = None):
    if scaler is None:
        scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, scaler


def check_econometric_vif(X_df: pd.DataFrame):
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_data = pd.DataFrame()
    vif_data["feature"] = X_df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_df.values, i)
        for i in range(len(X_df.columns))
    ]

    print("📊 Econometric Check: Variance Inflation Factor (VIF)")
    print(vif_data)
    return vif_data
