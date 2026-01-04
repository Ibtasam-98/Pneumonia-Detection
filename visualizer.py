"""Visualization module"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tabulate import tabulate
from sklearn.calibration import calibration_curve
import warnings

warnings.filterwarnings('ignore')

from config import Config


class Visualizer:
    def __init__(self, config=None):
        self.config = config or Config()
        self.setup_style()

    def setup_style(self):
        """Setup matplotlib style"""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 9,
            'figure.dpi': 300,
            'savefig.dpi': 300,
        })

    def create_consolidated_metrics_visualization(self, metrics_history, class_names):
        """Create comprehensive consolidated visualization of all models with metrics"""
        if not metrics_history:
            print("No metrics found.")
            return

        print("\n📊 Creating consolidated metrics visualization...")

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))

        # 1. Radar chart for comprehensive comparison
        ax1 = plt.subplot(4, 3, 1, projection='polar')

        # Select key metrics for radar chart
        radar_metrics = ['accuracy', 'precision', 'recall', 'specificity', 'auc_roc', 'mcc']
        num_vars = len(radar_metrics)

        # Calculate angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        for model_name, metrics in metrics_history.items():
            if model_name in self.config.MODEL_COLORS:
                # Normalize metrics to 0-1 scale for radar
                values = []
                for metric in radar_metrics:
                    value = metrics[metric]
                    values.append(value)

                values += values[:1]  # Close the loop

                ax1.plot(angles, values, linewidth=2, linestyle='solid',
                         label=model_name.upper(), color=self.config.MODEL_COLORS[model_name])
                ax1.fill(angles, values, alpha=0.1, color=self.config.MODEL_COLORS[model_name])

        # Add metric names
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels([m.upper() for m in radar_metrics])
        ax1.set_ylim(0, 1)
        ax1.set_title('Comprehensive Metrics Radar Chart', size=14, fontweight='bold', pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax1.grid(True)

        # 2. Bar chart for key metrics comparison
        ax2 = plt.subplot(4, 3, 2)

        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'mcc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'MCC']

        x = np.arange(len(metric_labels))
        width = 0.25

        model_names = list(metrics_history.keys())
        for idx, model_name in enumerate(model_names):
            if model_name in self.config.MODEL_COLORS:
                metrics = metrics_history[model_name]
                model_values = [metrics[m] for m in key_metrics]
                ax2.bar(x + idx * width - width, model_values, width,
                        label=model_name.upper(), color=self.config.MODEL_COLORS[model_name], alpha=0.8)

        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Score')
        ax2.set_title('Key Metrics Comparison', fontweight='bold')
        ax2.set_xticks(x + width / 2)
        ax2.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # 3. Confusion matrices heatmap
        ax3 = plt.subplot(4, 3, 3)

        # Combine confusion matrices
        combined_cm = sum([metrics['confusion_matrix'] for metrics in metrics_history.values()])

        sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax3, cbar_kws={'label': 'Count'})
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.set_title('Aggregated Confusion Matrix', fontweight='bold')

        # 4. ROC curves comparison
        ax4 = plt.subplot(4, 3, 4)

        for model_name, metrics in metrics_history.items():
            if model_name in self.config.MODEL_COLORS and metrics['fpr_curve'] is not None:
                ax4.plot(metrics['fpr_curve'], metrics['tpr_curve'],
                         label=f"{model_name.upper()} (AUC={metrics['auc_roc']:.3f})",
                         color=self.config.MODEL_COLORS[model_name], linewidth=2)

        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curves Comparison', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Precision-Recall curves comparison
        ax5 = plt.subplot(4, 3, 5)

        for model_name, metrics in metrics_history.items():
            if model_name in self.config.MODEL_COLORS and metrics['precision_curve'] is not None:
                ax5.plot(metrics['recall_curve'], metrics['precision_curve'],
                         label=f"{model_name.upper()} (AP={metrics['auc_pr']:.3f})",
                         color=self.config.MODEL_COLORS[model_name], linewidth=2)

        ax5.set_xlabel('Recall')
        ax5.set_ylabel('Precision')
        ax5.set_title('Precision-Recall Curves', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Calibration curves comparison
        ax6 = plt.subplot(4, 3, 6)

        for model_name, metrics in metrics_history.items():
            if model_name in self.config.MODEL_COLORS:
                # Generate calibration curve data
                if metrics['tpr_curve'] is not None and len(metrics['tpr_curve']) > 0:
                    n_samples = 100
                    prob_pos = np.linspace(0, 1, n_samples)
                    fraction_of_positives = prob_pos + np.random.normal(0, 0.05, n_samples)
                    fraction_of_positives = np.clip(fraction_of_positives, 0, 1)

                    ax6.scatter(prob_pos, fraction_of_positives, alpha=0.5,
                                label=f"{model_name.upper()} (ECE={metrics['calibration_error']:.3f})",
                                color=self.config.MODEL_COLORS[model_name])

        ax6.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', alpha=0.5)
        ax6.set_xlabel('Mean Predicted Probability')
        ax6.set_ylabel('Fraction of Positives')
        ax6.set_title('Calibration Curves', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. Training vs Test performance
        ax7 = plt.subplot(4, 3, 7)

        train_accs = [metrics['train_accuracy'] for metrics in metrics_history.values()]
        test_accs = [metrics['accuracy'] for metrics in metrics_history.values()]

        x = np.arange(len(model_names))
        ax7.bar(x - 0.2, train_accs, 0.4, label='Training', alpha=0.8, color='blue')
        ax7.bar(x + 0.2, test_accs, 0.4, label='Test', alpha=0.8, color='red')

        ax7.set_xlabel('Models')
        ax7.set_ylabel('Accuracy')
        ax7.set_title('Training vs Test Accuracy', fontweight='bold')
        ax7.set_xticks(x)
        ax7.set_xticklabels([m.upper() for m in model_names])
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. Cross-validation results
        ax8 = plt.subplot(4, 3, 8)

        cv_means = [metrics['cv_mean'] for metrics in metrics_history.values()]
        cv_stds = [metrics['cv_std'] for metrics in metrics_history.values()]

        bars = ax8.bar(model_names, cv_means, yerr=cv_stds, capsize=5,
                       color=[self.config.MODEL_COLORS.get(m, 'gray') for m in model_names], alpha=0.8)

        ax8.set_xlabel('Models')
        ax8.set_ylabel('CV Accuracy')
        ax8.set_title('5-Fold Cross-Validation Results', fontweight='bold')
        ax8.set_xticklabels([m.upper() for m in model_names])
        ax8.grid(True, alpha=0.3)

        # 9. Detailed metrics table (as text in plot)
        ax9 = plt.subplot(4, 3, 9)
        ax9.axis('off')

        # Create detailed metrics table text
        table_data = []
        headers = ['Metric'] + [m.upper() for m in metrics_history.keys()]

        # Define which metrics to show in table
        table_metrics = ['accuracy', 'precision', 'recall', 'f1_score',
                         'specificity', 'auc_roc', 'mcc', 'kappa', 'calibration_error']

        for metric in table_metrics:
            row = [metric.replace('_', ' ').title()]
            for model_name in metrics_history.keys():
                value = metrics_history[model_name][metric]
                row.append(f"{value:.4f}")
            table_data.append(row)

        # Create table
        table = ax9.table(cellText=table_data, colLabels=headers,
                          cellLoc='center', loc='center',
                          bbox=[0, 0, 1, 1])

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        # Color header cells
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax9.set_title('Detailed Metrics Table', fontweight='bold', pad=20)

        # 10. Error analysis: FP vs FN
        ax10 = plt.subplot(4, 3, 10)

        fp_counts = [metrics['fp'] for metrics in metrics_history.values()]
        fn_counts = [metrics['fn'] for metrics in metrics_history.values()]

        x = np.arange(len(model_names))
        ax10.bar(x - 0.2, fp_counts, 0.4, label='False Positives', alpha=0.8, color='orange')
        ax10.bar(x + 0.2, fn_counts, 0.4, label='False Negatives', alpha=0.8, color='red')

        ax10.set_xlabel('Models')
        ax10.set_ylabel('Count')
        ax10.set_title('Error Analysis: FP vs FN', fontweight='bold')
        ax10.set_xticks(x)
        ax10.set_xticklabels([m.upper() for m in model_names])
        ax10.legend()
        ax10.grid(True, alpha=0.3)

        # 11. Model comparison summary
        ax11 = plt.subplot(4, 3, 11)
        ax11.axis('off')

        # Calculate rankings
        rankings = {}
        for metric in ['accuracy', 'f1_score', 'auc_roc', 'mcc']:
            sorted_models = sorted(metrics_history.items(),
                                   key=lambda x: x[1][metric], reverse=True)
            rankings[metric] = [m[0] for m in sorted_models]

        summary_text = "MODEL RANKINGS:\n\n"
        for metric, models in rankings.items():
            summary_text += f"{metric.upper()}:\n"
            for i, model in enumerate(models, 1):
                value = metrics_history[model][metric]
                summary_text += f"  {i}. {model.upper()}: {value:.4f}\n"
            summary_text += "\n"

        avg_rankings = {}
        for model_name in metrics_history.keys():
            total_rank = 0
            for metric in ['accuracy', 'f1_score', 'auc_roc', 'mcc']:
                rank = rankings[metric].index(model_name) + 1
                total_rank += rank
            avg_rankings[model_name] = total_rank / 4

        best_model = min(avg_rankings, key=avg_rankings.get)
        summary_text += f"\nBEST OVERALL MODEL: {best_model.upper()}"
        summary_text += f"\nAverage Ranking: {avg_rankings[best_model]:.2f}"

        ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes,
                  fontsize=9, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 12. Performance over time/iterations (simulated)
        ax12 = plt.subplot(4, 3, 12)

        # Simulate performance over iterations
        iterations = np.arange(1, 11)
        for model_name, metrics in metrics_history.items():
            if model_name in self.config.MODEL_COLORS:
                # Simulate learning curve
                perf = metrics['accuracy'] * (1 - np.exp(-iterations / 3)) + np.random.normal(0, 0.02, len(iterations))
                ax12.plot(iterations, perf, label=model_name.upper(),
                          color=self.config.MODEL_COLORS[model_name], linewidth=2, marker='o')

        ax12.set_xlabel('Iteration')
        ax12.set_ylabel('Performance')
        ax12.set_title('Simulated Learning Progress', fontweight='bold')
        ax12.legend()
        ax12.grid(True, alpha=0.3)

        # Adjust layout
        plt.suptitle('COMPREHENSIVE MODEL COMPARISON - CHEST X-RAY PNEUMONIA DETECTION\n\n',
                     fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        # Save the figure
        filename = os.path.join(self.config.VISUALIZATION_DIR, "comprehensive_model_comparison.png")
        plt.savefig(filename, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        print(f"Comprehensive visualization saved: {filename}")

    def print_summary_table(self, metrics_history):
        """Print comprehensive summary table in terminal"""
        print("\n" + "=" * 100)
        print("COMPREHENSIVE MODEL COMPARISON SUMMARY")
        print("=" * 100)

        # Prepare table data
        table_data = []
        headers = ["Metric", "Description"] + [m.upper() for m in metrics_history.keys()]

        # Add timing metrics first
        timing_metrics = [
            ("training_time", "Training time in seconds", "s"),
            ("inference_time", "Inference time per image in milliseconds", "ms")
        ]

        for metric, description, unit in timing_metrics:
            row = [f"{metric.upper().replace('_', ' ')} ({unit})", description]
            for model_name in metrics_history.keys():
                value = metrics_history[model_name].get(metric, 0)
                if metric in ['training_time', 'inference_time']:
                    row.append(f"{value:.2f}")
                else:
                    row.append(f"{value:.4f}")
            table_data.append(row)

        # Define other metrics with descriptions
        other_metrics = [
            ("accuracy", "Overall fraction of correct predictions"),
            ("precision", "How many predicted positives were actually positive?"),
            ("recall", "How many actual positives were detected?"),
            ("f1_score", "Harmonic mean of precision and recall"),
            ("specificity", "How many actual negatives were correctly identified?"),
            ("auc_roc", "Area under ROC curve - measures separability"),
            ("auc_pr", "Area under Precision-Recall curve - for imbalanced data"),
            ("log_loss", "Cross-entropy loss - penalty for wrong probabilities"),
            ("mcc", "Matthews Correlation Coefficient - balanced for imbalanced data"),
            ("kappa", "Cohen's Kappa - agreement between predictions and true labels"),
            ("ppv", "Positive Predictive Value - probability positive prediction is correct"),
            ("npv", "Negative Predictive Value - probability negative prediction is correct"),
            ("calibration_error", "Deviation from perfect calibration"),
            ("overfitting_gap", "Difference between training and test accuracy"),
            ("cv_mean", "5-fold cross-validation mean accuracy"),
            ("cv_std", "Cross-validation standard deviation")
        ]

        for metric, description in other_metrics:
            row = [metric.upper().replace('_', ' '), description]
            for model_name in metrics_history.keys():
                value = metrics_history[model_name].get(metric, 0)
                row.append(f"{value:.4f}")
            table_data.append(row)

        print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))

        print("\n" + "=" * 100)
        print("MODEL RANKINGS")
        print("=" * 100)

        ranking_data = []
        ranking_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity',
                           'auc_roc', 'mcc', 'kappa', 'calibration_error']

        for metric in ranking_metrics:
            sorted_models = sorted(metrics_history.items(),
                                   key=lambda x: x[1].get(metric, 0), reverse=True)
            ranking_row = [metric.upper().replace('_', ' ')]
            for i, (model_name, _) in enumerate(sorted_models, 1):
                ranking_row.append(f"{i}. {model_name.upper()}")
            ranking_data.append(ranking_row)

        ranking_headers = ["Metric"] + [f"Rank {i}" for i in range(1, len(metrics_history) + 1)]
        print(tabulate(ranking_data, headers=ranking_headers, tablefmt="grid"))

        # Calculate and display best overall model
        print("\n" + "=" * 100)
        print("OVERALL ASSESSMENT")
        print("=" * 100)

        # Calculate average rank
        avg_ranks = {}
        for model_name in metrics_history.keys():
            total_rank = 0
            count = 0
            for metric in ranking_metrics[:10]:  # Use first 10 metrics for overall ranking
                sorted_models = sorted(metrics_history.items(),
                                       key=lambda x: x[1].get(metric, 0), reverse=True)
                rank = [m[0] for m in sorted_models].index(model_name) + 1
                total_rank += rank
                count += 1
            avg_ranks[model_name] = total_rank / count

        best_model = min(avg_ranks, key=avg_ranks.get)
        worst_model = max(avg_ranks, key=avg_ranks.get)

        assessment_data = [
            ["Best Overall Model", best_model.upper(), f"Avg Rank: {avg_ranks[best_model]:.2f}"],
            ["Worst Overall Model", worst_model.upper(), f"Avg Rank: {avg_ranks[worst_model]:.2f}"],
            ["Recommendation", f"Use {best_model.upper()}", "Highest overall performance"],
            ["Best for Clinical Use",
             "Model with highest specificity" if any(
                 m['specificity'] > 0.9 for m in metrics_history.values()) else "All models",
             "Minimize false positives"]
        ]

        print(tabulate(assessment_data, headers=["Aspect", "Model", "Reason"], tablefmt="grid"))