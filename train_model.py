"""
Depression Prediction Model Training Script
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ─── Data Loading ────────────────────────────────────────────────────────────

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    return train, test

# ─── Preprocessing ────────────────────────────────────────────────────────────

SLEEP_MAP = {
    'less than 5 hours': 4, '1-2 hours': 1.5, '2-3 hours': 2.5,
    '3-4 hours': 3.5, '4-5 hours': 4.5, '4-6 hours': 5, '5-6 hours': 5.5,
    '6-7 hours': 6.5, '6-8 hours': 7, '7-8 hours': 7.5, '8 hours': 8,
    '8-9 hours': 8.5, '9-11 hours': 10, '10-11 hours': 10.5,
    'more than 8 hours': 9, '9-5 hours': 7, '9-5': 7, '9-6 hours': 7.5,
    '1-3 hours': 2, '1-6 hours': 3.5, '3-6 hours': 4.5, '10-6 hours': 8,
    '35-36 hours': 8, '40-45 hours': 8, '45-48 hours': 8, '55-66 hours': 8,
    '45': 8, '49 hours': 8, 'than 5 hours': 4,
}

DIET_VALID = {'healthy', 'unhealthy', 'moderate'}

def clean_sleep(val):
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    return SLEEP_MAP.get(v, np.nan)

def clean_diet(val):
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    return v if v in DIET_VALID else np.nan

def preprocess(df, is_train=True, encoders=None, scaler=None, imputer=None):
    df = df.copy()

    # Drop non-feature cols
    drop_cols = ['id', 'Name', 'City', 'Profession', 'Degree']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Clean noisy columns
    df['Sleep Duration'] = df['Sleep Duration'].apply(clean_sleep)
    df['Dietary Habits'] = df['Dietary Habits'].apply(clean_diet)

    # Fill conditional missings: students have no Work Pressure / Job Satisfaction
    df['Work Pressure']    = df['Work Pressure'].fillna(0)
    df['Job Satisfaction'] = df['Job Satisfaction'].fillna(0)
    # Workers have no Academic Pressure / CGPA / Study Satisfaction
    df['Academic Pressure']  = df['Academic Pressure'].fillna(0)
    df['CGPA']               = df['CGPA'].fillna(0)
    df['Study Satisfaction'] = df['Study Satisfaction'].fillna(0)

    # Binary encodings
    binary_map = {
        'Gender':                             {'Male': 1, 'Female': 0},
        'Have you ever had suicidal thoughts ?': {'Yes': 1, 'No': 0},
        'Family History of Mental Illness':   {'Yes': 1, 'No': 0},
        'Working Professional or Student':    {'Working Professional': 1, 'Student': 0},
    }
    for col, mapping in binary_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Encode Dietary Habits
    if encoders is None:
        encoders = {}
    diet_col = 'Dietary Habits'
    df[diet_col] = df[diet_col].fillna('moderate')
    if is_train:
        le = LabelEncoder()
        df[diet_col] = le.fit_transform(df[diet_col].astype(str))
        encoders[diet_col] = le
    else:
        le = encoders[diet_col]
        df[diet_col] = df[diet_col].astype(str).apply(
            lambda x: x if x in le.classes_ else 'moderate')
        df[diet_col] = le.transform(df[diet_col])

    feature_cols = [c for c in df.columns if c != 'Depression']

    # Impute remaining NaNs
    if is_train:
        imputer = SimpleImputer(strategy='median')
        df[feature_cols] = imputer.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = imputer.transform(df[feature_cols])

    # Scale
    if is_train:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, encoders, scaler, imputer

# ─── Bias Evaluation ─────────────────────────────────────────────────────────

def evaluate_bias(y_true, y_pred, df_raw, group_col):
    results = {}
    unique_vals = df_raw[group_col].dropna().unique()
    for val in unique_vals:
        mask = df_raw[group_col] == val
        if mask.sum() < 50:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        results[str(val)] = {
            'accuracy':  round(accuracy_score(yt, yp), 4),
            'precision': round(precision_score(yt, yp, zero_division=0), 4),
            'recall':    round(recall_score(yt, yp, zero_division=0), 4),
            'f1':        round(f1_score(yt, yp, zero_division=0), 4),
            'n':         int(mask.sum()),
        }
    return results

# ─── Main Training ────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    train_df, test_df = load_data(
        '/mnt/user-data/uploads/train.csv',
        '/mnt/user-data/uploads/test.csv'
    )

    print("Preprocessing …")
    proc_train, encoders, scaler, imputer = preprocess(train_df, is_train=True)

    X = proc_train.drop(columns=['Depression'])
    y = proc_train['Depression']

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Train: {X_tr.shape}, Val: {X_val.shape}")
    print(f"Class distribution – 0: {(y_tr==0).sum()}, 1: {(y_tr==1).sum()}")

    # Deep Learning via MLP
    print("\nTraining MLP (Deep Learning) model …")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=512,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=True,
    )
    mlp.fit(X_tr, y_tr)

    y_pred = mlp.predict(X_val)
    y_prob = mlp.predict_proba(X_val)[:, 1]

    print("\n=== Validation Metrics ===")
    metrics = {
        'accuracy':  round(accuracy_score(y_val, y_pred), 4),
        'precision': round(precision_score(y_val, y_pred), 4),
        'recall':    round(recall_score(y_val, y_pred), 4),
        'f1':        round(f1_score(y_val, y_pred), 4),
    }
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Classification Report ===")
    print(classification_report(y_val, y_pred))

    # Bias evaluation
    print("\n=== Bias Evaluation ===")
    val_idx = X_val.index
    raw_val = train_df.loc[val_idx]
    bias_results = {}
    for grp in ['Gender', 'Working Professional or Student', 'Age']:
        if grp == 'Age':
            raw_val = raw_val.copy()
            raw_val['Age Group'] = pd.cut(raw_val['Age'], bins=[0,25,35,50,100],
                                          labels=['<25','25-35','35-50','50+'])
            res = evaluate_bias(y_val.values, y_pred, raw_val, 'Age Group')
            bias_results['Age Group'] = res
        else:
            res = evaluate_bias(y_val.values, y_pred, raw_val, grp)
            bias_results[grp] = res

    for grp, res in bias_results.items():
        print(f"\n  {grp}:")
        for subgrp, m in res.items():
            print(f"    {subgrp}: {m}")

    # Save artifacts
    print("\nSaving model artifacts …")
    joblib.dump(mlp, '/home/claude/depression_app/model.pkl')
    joblib.dump(encoders, '/home/claude/depression_app/encoders.pkl')
    joblib.dump(scaler, '/home/claude/depression_app/scaler.pkl')
    joblib.dump(imputer, '/home/claude/depression_app/imputer.pkl')

    feature_names = list(X.columns)
    with open('/home/claude/depression_app/feature_names.json', 'w') as f:
        json.dump(feature_names, f)

    with open('/home/claude/depression_app/metrics.json', 'w') as f:
        json.dump({'overall': metrics, 'bias': bias_results}, f, indent=2)

    print("\nAll artifacts saved. Training complete!")
    return metrics, bias_results

if __name__ == '__main__':
    main()
