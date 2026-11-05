import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

FEATURES_CSV = "data/labels/features.csv"
MODEL_DIR = "models"

MODEL_FILE = os.path.join(MODEL_DIR, "lane_direction_classifier.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, "feature_scaler.joblib")
ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.joblib")
MODEL_CARD_FILE = os.path.join(MODEL_DIR, "model_card.json")

def train_and_select_model():
    """
    Train and evaluate multiple machine learning classifiers on extracted CV features.
    Selects best model via 5-fold cross-validation, tunes hyperparameters, and saves artifacts.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Features file not found at {FEATURES_CSV}. Run src/features.py first.")
        
    df = pd.read_csv(FEATURES_CSV)
    df = df.dropna().reset_index(drop=True)
    
    feature_cols = [c for c in df.columns if c not in ["image_filename", "direction"]]
    X = df[feature_cols].values
    y_raw = df["direction"].values
    
    # Label Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    print(f"Loaded feature matrix shape: {X.shape}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Stratified Train/Test Split (75% Train, 25% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    
    # Save test set split indices/data for evaluate.py
    np.savez(
        os.path.join(MODEL_DIR, "test_split.npz"),
        X_test=X_test,
        y_test=y_test,
        feature_names=np.array(feature_cols)
    )
    
    # Feature Scaling (Fit on Train ONLY)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define candidate classifiers
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVC_RBF": SVC(kernel="rbf", probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = {}
    print("\n--- 5-Fold Stratified Cross-Validation Model Comparison ---")
    print(f"{'Model Name':<20} | {'Mean CV Acc':<12} | {'Std CV Acc':<12} | {'Mean CV F1 (Macro)':<18}")
    print("-" * 70)
    
    best_model_name = None
    best_cv_score = -1.0
    
    for name, model in candidates.items():
        scores = cross_validate(
            model, X_train_scaled, y_train,
            cv=cv, scoring=["accuracy", "f1_macro"]
        )
        mean_acc = float(np.mean(scores["test_accuracy"]))
        std_acc = float(np.std(scores["test_accuracy"]))
        mean_f1 = float(np.mean(scores["test_f1_macro"]))
        
        cv_results[name] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_f1_macro": mean_f1
        }
        
        print(f"{name:<20} | {mean_acc:<12.4f} | {std_acc:<12.4f} | {mean_f1:<18.4f}")
        
        if mean_acc > best_cv_score:
            best_cv_score = mean_acc
            best_model_name = name
            
    print(f"\nWinning Candidate Model: {best_model_name} (CV Acc: {best_cv_score:.4f})")
    
    # Perform Hyperparameter Tuning on Winning Model
    print(f"\nPerforming GridSearchCV hyperparameter tuning on {best_model_name}...")
    
    if best_model_name == "RandomForest":
        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5]
        }
        base_estimator = RandomForestClassifier(random_state=42)
    elif best_model_name == "SVC_RBF":
        param_grid = {
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", "auto", 0.01, 0.1]
        }
        base_estimator = SVC(kernel="rbf", probability=True, random_state=42)
    elif best_model_name == "GradientBoosting":
        param_grid = {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5]
        }
        base_estimator = GradientBoostingClassifier(random_state=42)
    else:  # LogisticRegression
        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs", "liblinear"]
        }
        base_estimator = LogisticRegression(max_iter=1000, random_state=42)
        
    grid_search = GridSearchCV(
        base_estimator, param_grid, cv=cv, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_tuned_cv_score = float(grid_search.best_score_)
    
    print(f"Optimal Hyperparameters: {best_params}")
    print(f"Tuned 5-Fold CV Accuracy: {best_tuned_cv_score:.4f}")
    
    # Refit tuned best model on full training set
    best_estimator.fit(X_train_scaled, y_train)
    
    # Save artifacts
    joblib.dump(best_estimator, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(label_encoder, ENCODER_FILE)
    
    model_card = {
        "chosen_model": best_model_name,
        "best_hyperparameters": best_params,
        "cv_results": cv_results,
        "tuned_cv_accuracy": best_tuned_cv_score,
        "num_train_samples": int(len(X_train)),
        "num_test_samples": int(len(X_test)),
        "feature_names": feature_cols,
        "classes": list(label_encoder.classes_)
    }
    
    with open(MODEL_CARD_FILE, "w") as f:
        json.dump(model_card, f, indent=4)
        
    print(f"\nSaved model artifacts to {MODEL_DIR}/:")
    print(f"  - Model: {MODEL_FILE}")
    print(f"  - Scaler: {SCALER_FILE}")
    print(f"  - Label Encoder: {ENCODER_FILE}")
    print(f"  - Model Card: {MODEL_CARD_FILE}")
    
    return best_estimator, scaler, label_encoder, model_card

if __name__ == "__main__":
    train_and_select_model()
