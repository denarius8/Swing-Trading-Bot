"""ML model training and prediction for SPX opening window direction."""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import config
from data_fetcher import fetch_spx_data
from indicators import add_all_features, get_feature_columns

warnings.filterwarnings("ignore")


def prepare_data(df):
    """Create target variable and clean data for modeling."""
    df = add_all_features(df)

    # Target: will the next day close higher than open?
    # This approximates the first ~2 hour move direction since
    # the opening range often sets the tone for the session.
    # 1 = bullish day (close > open), 0 = bearish day (close <= open)
    df["target"] = (df["Close"].shift(-1) > df["Open"].shift(-1)).astype(int)

    # Replace inf values and drop rows with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # Drop the last row (no target available)
    df = df.iloc[:-1]

    return df


def train_model(force_refresh_data=False):
    """Train the ensemble model and save artifacts."""
    print("\n" + "=" * 60)
    print("  SPX PREDICTIVE MODEL - TRAINING")
    print("=" * 60)

    # Fetch and prepare data
    raw_df = fetch_spx_data(force_refresh=force_refresh_data)
    df = prepare_data(raw_df)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["target"].values

    print(f"\n[MODEL] Features: {len(feature_cols)}")
    print(f"[MODEL] Samples: {len(X)}")
    print(f"[MODEL] Class balance: {y.mean():.1%} bullish / {1 - y.mean():.1%} bearish")

    # Time-series aware train/test split (no shuffling)
    split_idx = int(len(X) * (1 - config.TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"[MODEL] Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"[MODEL] Train period: {df.index[0].date()} to {df.index[split_idx - 1].date()}")
    print(f"[MODEL] Test period:  {df.index[split_idx].date()} to {df.index[-1].date()}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ensemble of Random Forest + Gradient Boosting
    rf = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        min_samples_split=config.MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.MIN_SAMPLES_LEAF,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )

    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_split=config.MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.MIN_SAMPLES_LEAF,
        random_state=config.RANDOM_STATE,
        subsample=0.8
    )

    model = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft"
    )

    print("\n[MODEL] Training ensemble (Random Forest + Gradient Boosting)...")
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[RESULTS] Test Accuracy: {accuracy:.1%}")

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_idx, val_idx in tscv.split(X_train_scaled):
        model_cv = RandomForestClassifier(
            n_estimators=200, max_depth=config.MAX_DEPTH,
            min_samples_split=config.MIN_SAMPLES_SPLIT,
            random_state=config.RANDOM_STATE, n_jobs=-1
        )
        model_cv.fit(X_train_scaled[train_idx], y_train[train_idx])
        cv_scores.append(model_cv.score(X_train_scaled[val_idx], y_train[val_idx]))
    print(f"[RESULTS] CV Accuracy (5-fold TS): {np.mean(cv_scores):.1%} (+/- {np.std(cv_scores):.1%})")

    print(f"\n[RESULTS] Classification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=["Bearish", "Bullish"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"[RESULTS] Confusion Matrix:")
    print(f"             Predicted")
    print(f"             Bear  Bull")
    print(f"  Actual Bear  {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"  Actual Bull  {cm[1][0]:4d}  {cm[1][1]:4d}")

    # Feature importance (from Random Forest)
    rf_model = model.named_estimators_["rf"]
    importances = rf_model.feature_importances_
    top_n = 15
    top_idx = np.argsort(importances)[-top_n:][::-1]
    print(f"\n[RESULTS] Top {top_n} Features:")
    for i, idx in enumerate(top_idx):
        print(f"  {i + 1:2d}. {feature_cols[idx]:<25s} {importances[idx]:.4f}")

    # Save model artifacts
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(feature_cols, config.FEATURE_PATH)
    print(f"\n[MODEL] Saved to {config.MODEL_PATH}")

    return model, scaler, feature_cols


def load_model():
    """Load trained model artifacts."""
    if not os.path.exists(config.MODEL_PATH):
        print("[MODEL] No trained model found. Training now...")
        return train_model()

    model = joblib.load(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    feature_cols = joblib.load(config.FEATURE_PATH)
    return model, scaler, feature_cols


def predict_next_day():
    """Generate prediction for the next trading day using latest data."""
    model, scaler, feature_cols = load_model()

    # Always fetch fresh data for predictions to ensure we use the latest close
    raw_df = fetch_spx_data(force_refresh=True)
    df = add_all_features(raw_df)
    df = df.dropna()

    # Use the most recent row as input
    latest = df.iloc[-1]
    X_new = latest[feature_cols].values.reshape(1, -1)
    X_new_scaled = scaler.transform(X_new)

    proba = model.predict_proba(X_new_scaled)[0]
    bear_prob = proba[0]
    bull_prob = proba[1]

    return {
        "date": df.index[-1],
        "close": latest["Close"],
        "bull_probability": bull_prob,
        "bear_probability": bear_prob,
        "features": {col: latest[col] for col in feature_cols},
        "raw_df": raw_df,
        "df": df
    }
