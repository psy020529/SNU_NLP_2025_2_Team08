"""Enhanced detector with better features and model."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from project.src.feature_extraction import extract_enhanced_features

def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    parser = argparse.ArgumentParser(description="Train enhanced detector")
    parser.add_argument("--baseline-dir", type=str, required=True)
    parser.add_argument("--manipulated-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--use-tfidf", action="store_true", help="Combine with TF-IDF features")
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    baseline_files = list(Path(args.baseline_dir).glob("*.jsonl"))
    baseline_data = []
    for f in baseline_files:
        baseline_data.extend(load_jsonl(f))
    
    manipulated_files = list(Path(args.manipulated_dir).glob("*.jsonl"))
    manipulated_data = []
    for f in manipulated_files:
        manipulated_data.extend(load_jsonl(f))
    
    print(f"Loaded {len(baseline_data)} baseline and {len(manipulated_data)} manipulated responses")
    
    # Extract responses and labels
    X_baseline_text = [d["response"] for d in baseline_data]
    X_manipulated_text = [d["response"] for d in manipulated_data]
    
    X_text = X_baseline_text + X_manipulated_text
    y = [0] * len(X_baseline_text) + [1] * len(X_manipulated_text)
    
    # Extract enhanced features
    print("\nExtracting enhanced linguistic features...")
    feature_list = []
    for text in X_text:
        features = extract_enhanced_features(text)
        feature_list.append(features)
    
    # Convert to DataFrame
    X_features = pd.DataFrame(feature_list)
    print(f"Extracted {X_features.shape[1]} linguistic features")
    
    # Optionally add TF-IDF
    if args.use_tfidf:
        print("Adding TF-IDF features...")
        vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        X_tfidf = vectorizer.fit_transform(X_text).toarray()
        X_tfidf_df = pd.DataFrame(X_tfidf, columns=[f'tfidf_{i}' for i in range(X_tfidf.shape[1])])
        X_combined = pd.concat([X_features, X_tfidf_df], axis=1)
    else:
        X_combined = X_features
    
    print(f"Total features: {X_combined.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest (more robust than LogReg for small data)
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=["Baseline", "Manipulated"]))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC: {auc:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_combined.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save results
    metrics = {
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "feature_importance": feature_importance.to_dict('records')[:20]
    }
    
    with open(output_path / "metrics_enhanced.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save model
    with open(output_path / "model_enhanced.pkl", 'wb') as f:
        pickle.dump({'scaler': scaler, 'classifier': clf, 'feature_names': X_combined.columns.tolist()}, f)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Baseline", "Manipulated"])
    ax.set_yticklabels(["Baseline", "Manipulated"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Enhanced Detector)")
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", 
                   color="white" if cm[i, j] > cm.max()/2 else "black")
    
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix_enhanced.png", dpi=150)
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Enhanced Detector)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "roc_curve_enhanced.png", dpi=150)
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = feature_importance.head(15)
    ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Feature Importances")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path / "feature_importance.png", dpi=150)
    
    print(f"\n✓ All results saved to {output_path}")

if __name__ == "__main__":
    main()
