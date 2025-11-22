"""Train ensemble detector with multiple classifiers."""
import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pickle
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from project.src.feature_extraction import extract_enhanced_features

def load_responses(baseline_dir, manipulated_dir):
    """Load responses from directories."""
    texts, labels = [], []
    
    # Load baseline (label=0)
    baseline_path = Path(baseline_dir)
    for jsonl_file in baseline_path.glob("*.jsonl"):
        if "v2" in jsonl_file.name:  # Use best performing version
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    texts.append(data['response'])
                    labels.append(0)
    
    # Load manipulated (label=1)
    manipulated_path = Path(manipulated_dir)
    for jsonl_file in manipulated_path.glob("*.jsonl"):
        if "v2" in jsonl_file.name:  # Use best performing version
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    texts.append(data['response'])
                    labels.append(1)
    
    return texts, np.array(labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=str, required=True)
    parser.add_argument("--manipulated-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tune-hyperparams", action="store_true",
                       help="Perform hyperparameter tuning (slower)")
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    texts, labels = load_responses(args.baseline_dir, args.manipulated_dir)
    print(f"Loaded {len(labels)} responses: {sum(labels==0)} baseline, {sum(labels==1)} manipulated\n")
    
    # Extract linguistic features
    print("Extracting enhanced linguistic features...")
    feature_dicts = [extract_enhanced_features(text) for text in texts]
    linguistic_features = np.array([[d[k] for k in sorted(d.keys())] for d in feature_dicts])
    feature_names = sorted(feature_dicts[0].keys())
    print(f"Extracted {len(feature_names)} linguistic features")
    
    # Extract TF-IDF features
    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    tfidf_features = tfidf.fit_transform(texts).toarray()
    print(f"Extracted {tfidf_features.shape[1]} TF-IDF features")
    
    # Combine features
    X = np.hstack([linguistic_features, tfidf_features])
    print(f"Total features: {X.shape[1]}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define classifiers
    if args.tune_hyperparams:
        print("Tuning Random Forest hyperparameters...")
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            rf_params, cv=5, scoring='roc_auc', n_jobs=-1
        )
        rf_grid.fit(X_train_scaled, y_train)
        rf = rf_grid.best_estimator_
        print(f"Best RF params: {rf_grid.best_params_}")
    else:
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, 
            class_weight='balanced', min_samples_split=5
        )
    
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, random_state=42, learning_rate=0.1
    )
    
    lr = LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced', C=1.0
    )
    
    # Create ensemble
    print("\nTraining Ensemble (RF + GB + LR)...")
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('lr', lr)
        ],
        voting='soft'  # Use probability voting
    )
    
    ensemble.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = ensemble.predict(X_test_scaled)
    y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    
    print("\n" + "="*60)
    print("ENSEMBLE CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=['Baseline', 'Manipulated']))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}\n")
    
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {auc:.3f}\n")
    
    # Save metrics
    metrics = {
        'auc': float(auc),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    with open(output_path / 'metrics_ensemble.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Ensemble Detector - Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Baseline', 'Manipulated'])
    plt.yticks(tick_marks, ['Baseline', 'Manipulated'])
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix_ensemble.png', dpi=150)
    print(f"✓ Saved confusion matrix")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Ensemble (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Ensemble Detector - ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'roc_curve_ensemble.png', dpi=150)
    print(f"✓ Saved ROC curve")
    
    # Save models
    with open(output_path / 'ensemble_model.pkl', 'wb') as f:
        pickle.dump({
            'ensemble': ensemble,
            'scaler': scaler,
            'tfidf': tfidf,
            'feature_names': feature_names
        }, f)
    print(f"✓ Saved ensemble model")
    
    print(f"\n✓ All results saved to {output_path}")

if __name__ == "__main__":
    main()
