import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler


def run_econometric_baseline_sksurv(X_scaled, y_struct):
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    print("\n📜 Econometric Baseline: Cox Proportional Hazards (sksurv version)")
    cph_sksurv = CoxPHSurvivalAnalysis()
    cph_sksurv.fit(X_scaled, y_struct)

    hr_data = {
        "feature": list(X_scaled.columns),
        "coef": list(cph_sksurv.coef_),
        "Hazard_Ratio (exp_coef)": list(np.exp(cph_sksurv.coef_)),
    }

    print(hr_data)

    plt.figure(figsize=(10, 6))
    plt.barh(hr_data["feature"], hr_data["coef"], color='skyblue')
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Cox Coefficients (Log-Hazard Scales)")
    plt.xlabel("Coefficient Value")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

    return cph_sksurv


def calculate_safe_brier(y_tr, y_va, surv_funcs, time_point):
    from sksurv.metrics import brier_score as sk_brier

    max_va_t = y_va['time'].max()
    safe_t = min(time_point, max_va_t - 0.01)
    preds_at_t = np.array([
        f(safe_t) if f.domain[1] >= safe_t else f(f.domain[1])
        for f in surv_funcs
    ])

    try:
        _, score = sk_brier(y_tr, y_va, preds_at_t, safe_t)
        return score[0]
    except Exception:
        return 0.1


def train_cv_rsf(X_scaled, y_struct, n_splits=5, random_state=2026, **kwargs):
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored as c_index_func

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_results = []

    print("🔬 Đang huấn luyện Oracle RSF (5-Fold CV)...")

    for fold, (t_idx, v_idx) in enumerate(kf.split(X_scaled)):
        X_tr, X_va = X_scaled.iloc[t_idx], X_scaled.iloc[v_idx]
        y_tr, y_va = y_struct[t_idx], y_struct[v_idx]

        rsf = RandomSurvivalForest(
            n_estimators=1500,
            max_depth=4,
            min_samples_leaf=12,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
            **kwargs,
        ).fit(X_tr, y_tr)

        c_idx = c_index_func(y_va['event'], y_va['time'], rsf.predict(X_va))[0]
        surv_funcs = rsf.predict_survival_function(X_va)
        b24 = calculate_safe_brier(y_tr, y_va, surv_funcs, 24)
        b48 = calculate_safe_brier(y_tr, y_va, surv_funcs, 48)
        b72 = calculate_safe_brier(y_tr, y_va, surv_funcs, 72)

        w_brier = (0.3 * b24) + (0.4 * b48) + (0.3 * b72)
        hybrid_score = 0.3 * c_idx + 0.7 * (1 - w_brier)

        cv_results.append(hybrid_score)
        print(f"Fold {fold+1}: Hybrid = {hybrid_score:.4f} (C-Idx: {c_idx:.4f}, W-Brier: {w_brier:.4f})")

    print(f"\n✅ FINAL MASTER SCORE: {np.mean(cv_results):.4f}")
    return cv_results


def train_final_rsf(X_scaled, y_struct, **kwargs):
    from sksurv.ensemble import RandomSurvivalForest

    final_model = RandomSurvivalForest(
        n_estimators=1500,
        max_depth=4,
        min_samples_leaf=12,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        **kwargs,
    ).fit(X_scaled, y_struct)

    return final_model


def generate_submission(test_df, model, scaler_obj, features, id_col='event_id'):
    df_t = master_oracle_engineering(test_df)
    X_test_scaled = scaler_obj.transform(df_t[features])
    surv_funcs = model.predict_survival_function(X_test_scaled)
    risk_scores = model.predict(X_test_scaled)

    max_train_t = model.unique_times_[-1]
    horizons = [12, 24, 48, 72]
    results = {}

    if id_col in test_df.columns:
        results[id_col] = test_df[id_col]
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy cột {id_col}, đang dùng Index làm ID.")
        results[id_col] = test_df.index

    for h in horizons:
        safe_h = min(h, max_train_t)
        results[f'prob_{h}h'] = [1 - f(safe_h) for f in surv_funcs]

    sub_df = pd.DataFrame(results)
    sub_df.insert(1, 'Risk_Score', risk_scores)
    return sub_df


def master_oracle_engineering(data):
    # Inline import to avoid circular dependency if imported from notebooks
    from src.feature_engineering import master_oracle_engineering as _engineer
    return _engineer(data)
