import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
# pip install optuna-integration[mlflow]
from optuna.integration.mlflow import MLflowCallback

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import time
import os

# Limit CPU overhead for parallel backends
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1: Load Dataset
# ─────────────────────────────────────────────
csv_filename = "neo.csv"

if not os.path.exists(csv_filename):
    raise FileNotFoundError(f"Could not find '{csv_filename}' in the current working directory.")

df = pd.read_csv(csv_filename)

# ─────────────────────────────────────────────
# STEP 2: Data Cleaning & Feature Engineering
# ─────────────────────────────────────────────
df.columns = df.columns.str.lower().str.replace(" ", "_")
df = df.drop_duplicates()

# Drop identifier columns and columns with zero variance (orbiting_body='Earth', sentry_object=False)
columns_to_drop = ['id', 'name', 'orbiting_body', 'sentry_object']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Remove outliers using IQR on remaining numeric feature columns
numeric_features = ['est_diameter_min', 'est_diameter_max', 'relative_velocity', 'miss_distance', 'absolute_magnitude']
Q1 = df[numeric_features].quantile(0.25)
Q3 = df[numeric_features].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = ((df[numeric_features] < (Q1 - 1.5 * IQR)) | (df[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)
df = df[~outlier_mask]

# OPTIONAL COMPUTE PROTECTION: SVC scales quadratically O(N^2). 
# Downsampling ensures the hyperparameter optimization finishes in a reasonable time.
if len(df) > 20000:
    df = df.sample(n=20000, random_state=42).reset_index(drop=True)

print(f"Dataset shape after cleaning and sampling: {df.shape}")

# ─────────────────────────────────────────────
# STEP 3: Segregate Features and Target
# ─────────────────────────────────────────────
X = df[numeric_features]
y = df['hazardous'].astype(int)  # Convert boolean target to 1/0 binary integers

# ─────────────────────────────────────────────
# STEP 4: Define Column Groups
# ─────────────────────────────────────────────
numeric_cols = X.columns.tolist()  # All remaining features are strictly numeric

# ─────────────────────────────────────────────
# STEP 5: Train / Test Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ─────────────────────────────────────────────
# STEP 6: Preprocessor (Scaling)
# ─────────────────────────────────────────────
def build_preprocessor(scaler):
    """Build a ColumnTransformer with given scaler for numeric features."""
    return ColumnTransformer(
        transformers=[
            ('num', scaler, numeric_cols)
        ],
        remainder='drop'
    )

# ─────────────────────────────────────────────
# STEP 7: Objective Functions (Optimizing ROC AUC)
# ─────────────────────────────────────────────

def objective_knn(trial):
    scaler = StandardScaler() if trial.suggest_categorical('scaler_type', ['standard', 'minmax']) == 'standard' else MinMaxScaler()
    preprocessor = build_preprocessor(scaler)
    model = KNeighborsClassifier(
        n_neighbors = trial.suggest_int('n_neighbors', 3, 21, step=2),
        weights     = trial.suggest_categorical('weights', ['uniform', 'distance']),
        p           = trial.suggest_int('p', 1, 3)
    )
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=kf).mean()

def objective_dt(trial):
    scaler = StandardScaler() if trial.suggest_categorical('scaler_type', ['standard', 'minmax']) == 'standard' else MinMaxScaler()
    preprocessor = build_preprocessor(scaler)
    model = DecisionTreeClassifier(
        criterion         = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        max_depth         = trial.suggest_int('max_depth', 2, 30),
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf  = trial.suggest_int('min_samples_leaf', 1, 20),
        max_features      = trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
        random_state      = 42
    )
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=kf).mean()

def objective_svc(trial):
    scaler = StandardScaler() if trial.suggest_categorical('scaler_type', ['standard', 'minmax']) == 'standard' else MinMaxScaler()
    preprocessor = build_preprocessor(scaler)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
    params = {
        'C':      trial.suggest_float('C', 1e-3, 1e2, log=True),
        'kernel': kernel,
        'probability': True  # Required for predicting robust probabilities and ROC AUC scores
    }
    if kernel in ['rbf', 'poly', 'sigmoid']:
        params['gamma'] = trial.suggest_float('gamma', 1e-4, 1e-1, log=True)
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
    
    model = SVC(**params, random_state=42)
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    kf = KFold(n_splits=3, shuffle=True, random_state=42)  # 3 Folds used for SVR/SVC parity speed
    return cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=kf).mean()

def objective_ridge(trial):
    scaler = StandardScaler() if trial.suggest_categorical('scaler_type', ['standard', 'minmax']) == 'standard' else MinMaxScaler()
    preprocessor = build_preprocessor(scaler)
    model = RidgeClassifier(
        alpha  = trial.suggest_float('alpha', 1e-3, 1e3, log=True),
        solver = trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr'])
    )
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=kf).mean()

def objective_rf(trial):
    scaler = StandardScaler() if trial.suggest_categorical('scaler_type', ['standard', 'minmax']) == 'standard' else MinMaxScaler()
    preprocessor = build_preprocessor(scaler)
    model = RandomForestClassifier(
        n_estimators      = trial.suggest_int('n_estimators', 100, 500, step=50),
        criterion         = trial.suggest_categorical('criterion', ['gini', 'entropy']),
        max_depth         = trial.suggest_int('max_depth', 5, 40),
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf  = trial.suggest_int('min_samples_leaf', 1, 20),
        max_features      = trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        bootstrap         = trial.suggest_categorical('bootstrap', [True, False]),
        random_state      = 42,
        n_jobs            = -1
    )
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=kf).mean()

def objective_gb(trial):
    scaler = StandardScaler() if trial.suggest_categorical('scaler_type', ['standard', 'minmax']) == 'standard' else MinMaxScaler()
    preprocessor = build_preprocessor(scaler)
    model = GradientBoostingClassifier(
        n_estimators      = trial.suggest_int('n_estimators', 100, 500, step=50),
        learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        max_depth         = trial.suggest_int('max_depth', 2, 10),
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf  = trial.suggest_int('min_samples_leaf', 1, 20),
        max_features      = trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        subsample         = trial.suggest_float('subsample', 0.5, 1.0),
        random_state      = 42
    )
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=kf).mean()

# ─────────────────────────────────────────────
# STEP 8: Map model names to objectives
# ─────────────────────────────────────────────
objectives = {
    "KNN":              objective_knn,
    "DecisionTree":     objective_dt,
    "SVC":              objective_svc,
    "Ridge":            objective_ridge,
    "RandomForest":     objective_rf,
    "GradientBoosting": objective_gb
}

# ─────────────────────────────────────────────
# STEP 9: MLflow Experiment
# ─────────────────────────────────────────────
mlflow.set_experiment("NEO_HAZARDOUS_CLASSIFICATION")

results    = {}
model_dict = {name: idx for idx, name in enumerate(objectives.keys())}
scaler_dict = {'standard': 0, 'minmax': 1}

# ─────────────────────────────────────────────
# STEP 10: Loop & Optimize Each Model
# ─────────────────────────────────────────────
for model_name, obj_fn in objectives.items():
    print(f"\n--- Optimizing {model_name} ---")

    with mlflow.start_run(run_name=f"{model_name}_Parent"):
        
        mlflow_cb = MLflowCallback(
            tracking_uri  = None,
            metric_name   = "cv_roc_auc",
            mlflow_kwargs = {"nested": True}
        )

        study = optuna.create_study(direction="maximize")
        
        start_fit = time.time()
        study.optimize(obj_fn, n_trials=20, callbacks=[mlflow_cb])
        fit_time = time.time() - start_fit

        print(f"Best CV ROC AUC for {model_name}: {study.best_value:.4f}")
        
        best_params = study.best_params.copy()
        scaler_type = best_params.pop("scaler_type")

        results[model_name] = {
            "best_params":    study.best_params,
            "best_cv_roc_auc": study.best_value
        }

        # ── Rebuild best pipeline ──────────────────
        best_scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        best_preprocessor = build_preprocessor(best_scaler)

        if model_name == "KNN":
            best_model = KNeighborsClassifier(**best_params)
        elif model_name == "DecisionTree":
            best_model = DecisionTreeClassifier(**best_params, random_state=42)
        elif model_name == "SVC":
            best_model = SVC(**best_params, probability=True, random_state=42)
        elif model_name == "Ridge":
            best_model = RidgeClassifier(**best_params)
        elif model_name == "RandomForest":
            best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        elif model_name == "GradientBoosting":
            best_model = GradientBoostingClassifier(**best_params, random_state=42)

        best_pipeline = Pipeline([
            ('preprocessor', best_preprocessor),
            ('model',        best_model)
        ])

        # ── Train & Evaluate Final Classification Pipeline ────────
        best_pipeline.fit(X_train, y_train)

        start_test = time.time()
        y_pred     = best_pipeline.predict(X_test)
        
        # RidgeClassifier does not natively support predict_proba()
        if hasattr(best_pipeline['model'], "predict_proba"):
            y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        else:
            y_proba = best_pipeline.decision_function(X_test)
            
        test_time  = time.time() - start_test

        train_acc     = best_pipeline.score(X_train, y_train)
        test_acc      = accuracy_score(y_test, y_pred)
        test_f1       = f1_score(y_test, y_pred)
        test_roc_auc  = roc_auc_score(y_test, y_proba)

        print(f"{model_name} | Accuracy: {test_acc:.4f} | F1: {test_f1:.4f} | ROC AUC: {test_roc_auc:.4f}")

        # ── Save model to calculate structural footprint ───
        model_path = f"{model_name}_final_model.pkl"
        joblib.dump(best_pipeline, model_path)
        model_size = os.path.getsize(model_path)

        # ── Log final evaluation run to MLflow ───
        mlflow.log_params(study.best_params)
        mlflow.log_metric("model_id",       model_dict[model_name])
        mlflow.log_metric("scaler_id",      scaler_dict[scaler_type])
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy",  test_acc)
        mlflow.log_metric("test_f1",        test_f1)
        mlflow.log_metric("test_roc_auc",   test_roc_auc)
        mlflow.log_metric("train_time",     fit_time)
        mlflow.log_metric("test_time",      test_time)
        mlflow.log_metric("model_size",     model_size)
        mlflow.sklearn.log_model(best_pipeline, artifact_path=f"{model_name}_model")
        
        if os.path.exists(model_path):
            os.remove(model_path)

        results[model_name].update({
            "test_accuracy":    test_acc,
            "test_f1":          test_f1,
            "test_roc_auc":     test_roc_auc,
            "fit_time":         fit_time,
            "test_time":        test_time,
            "model_size_bytes": model_size
        })

# ─────────────────────────────────────────────
# STEP 11: Summary
# ─────────────────────────────────────────────
print("\n" + "="*95)
print("FINAL SUMMARY — NEAR EARTH OBJECTS (NEO) HAZARD CLASSIFICATION")
print("="*95)
for model_name, res in results.items():
    print(
        f"{model_name:<16} | CV AUC={res['best_cv_roc_auc']:.4f} | "
        f"Test Acc={res['test_accuracy']:.4f} | Test F1={res['test_f1']:.4f} | "
        f"Test AUC={res['test_roc_auc']:.4f} | Fit Time={res['fit_time']:.1f}s | "
        f"Size={res['model_size_bytes']} B"
    )