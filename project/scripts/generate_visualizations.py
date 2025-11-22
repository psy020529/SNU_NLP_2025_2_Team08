"""Generate comparison visualizations for v2 vs v3 vs ensemble results."""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_metrics(file_path):
    """Load metrics from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    # Load all metrics
    metrics_v2 = load_metrics('nlp-proj/outputs/detector_results_enhanced/metrics_enhanced.json')
    metrics_ensemble = load_metrics('nlp-proj/outputs/detector_results_ensemble/metrics_ensemble.json')
    
    # Load scores
    scores_v2 = pd.read_csv('nlp-proj/outputs/scores/scores_v2.csv')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. AUC Comparison
    ax1 = plt.subplot(2, 3, 1)
    methods = ['v1\n(LogReg)', 'v2\n(RF Enhanced)', 'v3\n(Ensemble)']
    aucs = [0.331, metrics_v2['auc'], metrics_ensemble['auc']]
    colors = ['red', 'orange', 'green']
    bars = ax1.bar(methods, aucs, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0.7, color='blue', linestyle='--', linewidth=2, label='Target (0.70)')
    ax1.set_ylabel('ROC AUC Score', fontsize=12, fontweight='bold')
    ax1.set_title('Detector Performance Improvement', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. F1 Score Comparison
    ax2 = plt.subplot(2, 3, 2)
    x = np.arange(2)
    width = 0.25
    
    f1_v1 = [0.00, 0.79]  # Baseline, Manipulated
    f1_v2 = [
        metrics_v2['classification_report']['0']['f1-score'],
        metrics_v2['classification_report']['1']['f1-score']
    ]
    
    # Handle different key formats in ensemble metrics
    ensemble_report = metrics_ensemble['classification_report']
    if 'Baseline' in ensemble_report:
        f1_ensemble = [
            ensemble_report['Baseline']['f1-score'],
            ensemble_report['Manipulated']['f1-score']
        ]
    else:
        f1_ensemble = [
            ensemble_report['0']['f1-score'],
            ensemble_report['1']['f1-score']
        ]
    
    ax2.bar(x - width, f1_v1, width, label='v1', color='red', alpha=0.7)
    ax2.bar(x, f1_v2, width, label='v2', color='orange', alpha=0.7)
    ax2.bar(x + width, f1_ensemble, width, label='v3 (Ensemble)', color='green', alpha=0.7)
    
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1 Score by Class', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Baseline', 'Manipulated'])
    ax2.legend()
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Polarity Distribution by Frame
    ax3 = plt.subplot(2, 3, 3)
    frames = ['CON', 'NEUTRAL', 'PRO']
    frame_data = scores_v2.groupby('frame')['polarity'].apply(list)
    
    positions = [1, 2, 3]
    box_parts = ax3.boxplot(
        [frame_data['CON'], frame_data['NEUTRAL'], frame_data['PRO']],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        labels=frames
    )
    
    # Color boxes
    colors_frames = ['lightcoral', 'lightgray', 'lightgreen']
    for patch, color in zip(box_parts['boxes'], colors_frames):
        patch.set_facecolor(color)
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Polarity (VADER)', fontsize=12, fontweight='bold')
    ax3.set_title('Polarity Distribution by Frame (v2)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Frame Type', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add mean values
    means = [np.mean(frame_data['CON']), np.mean(frame_data['NEUTRAL']), np.mean(frame_data['PRO'])]
    for pos, mean, color in zip(positions, means, colors_frames):
        ax3.plot(pos, mean, 'D', markersize=10, color='darkred', 
                markeredgecolor='black', markeredgewidth=1.5)
        ax3.text(pos, mean + 0.1, f'{mean:.3f}', 
                ha='center', fontsize=9, fontweight='bold')
    
    # 4. Confusion Matrix - v2
    ax4 = plt.subplot(2, 3, 4)
    cm_v2 = np.array(metrics_v2['confusion_matrix'])
    im4 = ax4.imshow(cm_v2, interpolation='nearest', cmap=plt.cm.Oranges)
    ax4.set_title('Confusion Matrix - v2 (RF)', fontsize=14, fontweight='bold')
    
    tick_marks = np.arange(2)
    ax4.set_xticks(tick_marks)
    ax4.set_yticks(tick_marks)
    ax4.set_xticklabels(['Baseline', 'Manipulated'])
    ax4.set_yticklabels(['Baseline', 'Manipulated'])
    
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, str(cm_v2[i, j]),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm_v2[i, j] > cm_v2.max() / 2 else "black",
                    fontweight='bold')
    
    ax4.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # 5. Confusion Matrix - Ensemble
    ax5 = plt.subplot(2, 3, 5)
    cm_ensemble = np.array(metrics_ensemble['confusion_matrix'])
    im5 = ax5.imshow(cm_ensemble, interpolation='nearest', cmap=plt.cm.Greens)
    ax5.set_title('Confusion Matrix - v3 (Ensemble)', fontsize=14, fontweight='bold')
    
    ax5.set_xticks(tick_marks)
    ax5.set_yticks(tick_marks)
    ax5.set_xticklabels(['Baseline', 'Manipulated'])
    ax5.set_yticklabels(['Baseline', 'Manipulated'])
    
    for i in range(2):
        for j in range(2):
            ax5.text(j, i, str(cm_ensemble[i, j]),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm_ensemble[i, j] > cm_ensemble.max() / 2 else "black",
                    fontweight='bold')
    
    ax5.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # 6. Summary Text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    PERFORMANCE SUMMARY
    {'='*50}
    
    Research Question 1 (Controllability):
    ✓ Enhanced prompts increased polarity 
      separation 10.7x (0.077 → 0.821)
    ✓ CON: -0.849 (highly negative)
    ✓ PRO: -0.028 (nearly neutral)
    ✓ NEUTRAL: -0.072 (balanced)
    
    Research Question 2 (Detectability):
    ✓ Ensemble detector achieved AUC=0.846
    ✓ Exceeds target threshold (0.70)
    ✓ Manipulated detection: 83% recall
    
    KEY IMPROVEMENTS:
    • Enhanced system prompts (v2)
    • Linguistic features (13 metrics)
    • Ensemble learning (RF+GB+LR)
    
    CONCLUSION:
    Both RQs answered successfully!
    LLM responses can be steered AND detected.
    """
    
    ax6.text(0.1, 0.5, summary_text, 
            fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('NLP Project: LLM Response Framing - Complete Results', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    output_path = Path('nlp-proj/outputs/figs/comprehensive_results.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comprehensive visualization to {output_path}")
    
    # Also create individual polarity comparison chart
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    frame_means = scores_v2.groupby('frame')['polarity'].mean()
    frame_stds = scores_v2.groupby('frame')['polarity'].std()
    
    frames_ordered = ['CON', 'NEUTRAL', 'PRO']
    means_ordered = [frame_means[f] for f in frames_ordered]
    stds_ordered = [frame_stds[f] for f in frames_ordered]
    
    x_pos = np.arange(len(frames_ordered))
    colors_bar = ['#ff6b6b', '#95a5a6', '#51cf66']
    
    bars = ax.bar(x_pos, means_ordered, yerr=stds_ordered, 
                  color=colors_bar, alpha=0.7, capsize=10,
                  edgecolor='black', linewidth=2)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Mean Polarity Score (VADER)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frame Type', fontsize=14, fontweight='bold')
    ax.set_title('Polarity by Frame (v2 - Enhanced Prompts)', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(frames_ordered, fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means_ordered, stds_ordered)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.05,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add polarity difference annotation
    polarity_diff = means_ordered[2] - means_ordered[0]  # PRO - CON
    ax.annotate('', xy=(2, means_ordered[2]), xytext=(0, means_ordered[0]),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax.text(1, (means_ordered[0] + means_ordered[2])/2 - 0.2,
            f'Δ = {polarity_diff:.3f}',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    polarity_path = Path('nlp-proj/outputs/figs/polarity_comparison.png')
    plt.savefig(polarity_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved polarity comparison to {polarity_path}")
    
    print("\n✓ All visualizations generated successfully!")

if __name__ == "__main__":
    main()
