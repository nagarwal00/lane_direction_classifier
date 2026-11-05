import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

MODEL_DIR = "models"
RESULTS_DIR = "results"

MODEL_FILE = os.path.join(MODEL_DIR, "lane_direction_classifier.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, "feature_scaler.joblib")
ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.joblib")
TEST_SPLIT_FILE = os.path.join(MODEL_DIR, "test_split.npz")

CONFUSION_MATRIX_PNG = os.path.join(RESULTS_DIR, "confusion_matrix.png")
REPORT_TXT = os.path.join(RESULTS_DIR, "classification_report.txt")
FEATURE_IMPORTANCE_PNG = os.path.join(RESULTS_DIR, "feature_importance.png")

def evaluate_model():
    """
    Evaluates trained classifier on held-out test set.
    Generates classification report, confusion matrix heatmap, and feature importance chart.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found at {MODEL_FILE}. Run src/train.py first.")
        
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    encoder = joblib.load(ENCODER_FILE)
    
    test_data = np.load(TEST_SPLIT_FILE)
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    feature_names = test_data["feature_names"]
    
    class_names = encoder.classes_
    
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    
    # Model Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Compute Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report_str = classification_report(
        y_test, y_pred, target_names=class_names, digits=4
    )
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n==================================================")
    print("           HELD-OUT TEST EVALUATION RESULTS       ")
    print("==================================================")
    print(f"Overall Test Accuracy : {acc * 100.0:.2f}%")
    print(f"Macro F1-Score        : {macro_f1:.4f}")
    print("\nClassification Report:\n")
    print(report_str)
    
    # Save Classification Report Text File
    with open(REPORT_TXT, "w") as f:
        f.write(f"Lane Curvature Direction Classifier — Final Evaluation\n")
        f.write(f"=====================================================\n\n")
        f.write(f"Overall Accuracy: {acc * 100.0:.2f}%\n")
        f.write(f"Macro F1-Score:   {macro_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n")
        
    # Plot & Save Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix — Lane Direction Classifier", fontsize=13, fontweight="bold", pad=15)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, [c.upper() for c in class_names], fontsize=11)
    plt.yticks(tick_marks, [c.upper() for c in class_names], fontsize=11)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold"
            )
            
    plt.ylabel("True Ground-Truth Label", fontsize=11)
    plt.xlabel("Predicted Model Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG, dpi=300)
    plt.close()
    
    # Feature Importance Plot (if available)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        sorted_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        plt.figure(figsize=(10, 5))
        plt.barh(range(len(sorted_importances)), sorted_importances[::-1], color="#2b5c8f", align="center")
        plt.yticks(range(len(sorted_importances)), sorted_names[::-1], fontsize=11)
        plt.xlabel("Relative Feature Importance Score", fontsize=11)
        plt.title("Feature Importance Breakdown", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_PNG, dpi=300)
        plt.close()
        
        print(f"\nTop 3 Driving Features:")
        for idx in range(min(3, len(sorted_names))):
            print(f"  {idx+1}. {sorted_names[idx]:<25}: {sorted_importances[idx]:.4f}")
            
    print(f"\nArtifacts saved successfully:")
    print(f"  - Report: {REPORT_TXT}")
    print(f"  - Confusion Matrix Plot: {CONFUSION_MATRIX_PNG}")
    print(f"  - Feature Importance Plot: {FEATURE_IMPORTANCE_PNG}")
    
    return acc, macro_f1

if __name__ == "__main__":
    evaluate_model()
