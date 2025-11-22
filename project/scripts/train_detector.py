"""Train a binary classifier to detect manipulated (PRO/CON) vs baseline (NEUTRAL) responses."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    parser = argparse.ArgumentParser(description="Train detector")
    parser.add_argument("--baseline-dir", type=str, required=True,
                       help="Directory with baseline (NEUTRAL) responses")
    parser.add_argument("--manipulated-dir", type=str, required=True,
                       help="Directory with manipulated (PRO/CON) responses")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--test-size", type=float, default=0.3,
                       help="Test set ratio")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load baseline (label = 0)
    baseline_files = list(Path(args.baseline_dir).glob("*.jsonl"))
    baseline_data = []
    for f in baseline_files:
        baseline_data.extend(load_jsonl(f))
    
    # Load manipulated (label = 1)
    manipulated_files = list(Path(args.manipulated_dir).glob("*.jsonl"))
    manipulated_data = []
    for f in manipulated_files:
        manipulated_data.extend(load_jsonl(f))
    
    print(f"Loaded {len(baseline_data)} baseline and {len(manipulated_data)} manipulated responses")
    
    # Prepare data
    X_baseline = [d["response"] for d in baseline_data]
    y_baseline = [0] * len(X_baseline)
    
    X_manipulated = [d["response"] for d in manipulated_data]
    y_manipulated = [1] * len(X_manipulated)
    
    X = X_baseline + X_manipulated
    y = y_baseline + y_manipulated
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train classifier
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train_vec, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test_vec)
    y_pred_proba = clf.predict_proba(X_test_vec)[:, 1]
    
    # Metrics
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=["Baseline", "Manipulated"]))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # ROC AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC: {auc:.3f}")
    
    # Save metrics
    metrics = {
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    with open(output_path / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save model
    with open(output_path / "model.pkl", 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'classifier': clf}, f)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Baseline", "Manipulated"])
    ax.set_yticklabels(["Baseline", "Manipulated"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png", dpi=150)
    print(f"\n✓ Saved confusion matrix to {output_path / 'confusion_matrix.png'}")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "roc_curve.png", dpi=150)
    print(f"✓ Saved ROC curve to {output_path / 'roc_curve.png'}")
    
    print(f"\n✓ All results saved to {output_path}")

if __name__ == "__main__":
    main()
