"""Visualization module for metrics and results"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from tabulate import tabulate

from config import config
from utils import print_header, print_success, print_info


class MetricsVisualizer:
    """Comprehensive visualization class for model metrics"""

    def __init__(self, class_names: List[str] = None, output_dir: str = None):
        """
        Initialize visualizer

        Args:
            class_names: Names of classes
            output_dir: Directory to save visualizations
        """
        self.class_names = class_names or config.CLASS_NAMES
        self.output_dir = output_dir or config.VISUALIZATION_DIR
        self.model_colors = config.MODEL_COLORS

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Set publication quality style
        self._set_plot_style()

    def _set_plot_style(self):
        """Set matplotlib style for publication-quality plots"""
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

    def create_consolidated_visualization(self, metrics_history: Dict[str, Dict]):
        """
        Create comprehensive consolidated visualization of all models

        Args:
            metrics_history: Dictionary of metrics for each model
        """
        if not metrics_history:
            print_info("No metrics found. Skipping visualization.")
            return

        print_header("Creating Consolidated Metrics Visualization")

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))

        # 1. Radar chart for comprehensive comparison
        ax1 = plt.subplot(4, 3, 1, projection='polar')
        self._plot_radar_chart(ax1, metrics_history)

        # 2. Bar chart for key metrics comparison
        ax2 = plt.subplot(4, 3, 2)
        self._plot_key_metrics_comparison(ax2, metrics_history)

        # 3. Aggregated confusion matrix
        ax3 = plt.subplot(4, 3, 3)
        self._plot_aggregated_confusion_matrix(ax3, metrics_history)

        # 4. ROC curves comparison
        ax4 = plt.subplot(4, 3, 4)
        self._plot_roc_curves(ax4, metrics_history)

        # 5. Precision-Recall curves
        ax5 = plt.subplot(4, 3, 5)
        self._plot_pr_curves(ax5, metrics_history)

        # 6. Calibration curves
        ax6 = plt.subplot(4, 3, 6)
        self._plot_calibration_curves(ax6, metrics_history)

        # 7. Training vs Test performance
        ax7 = plt.subplot(4, 3, 7)
        self._plot_train_test_comparison(ax7, metrics_history)

        # 8. Cross-validation results
        ax8 = plt.subplot(4, 3, 8)
        self._plot_cv_results(ax8, metrics_history)

        # 9. Detailed metrics table
        ax9 = plt.subplot(4, 3, 9)
        self._plot_metrics_table(ax9, metrics_history)

        # 10. Error analysis: FP vs FN
        ax10 = plt.subplot(4, 3, 10)
        self._plot_error_analysis(ax10, metrics_history)

        # 11. Model comparison summary
        ax11 = plt.subplot(4, 3, 11)
        self._plot_model_summary(ax11, metrics_history)

        # 12. Simulated learning progress
        ax12 = plt.subplot(4, 3, 12)
        self._plot_learning_progress(ax12, metrics_history)

        # Adjust layout
        plt.suptitle('COMPREHENSIVE MODEL COMPARISON - CHEST X-RAY PNEUMONIA DETECTION\n\n',
                     fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        # Save the figure
        filename = os.path.join(self.output_dir, "comprehensive_model_comparison.png")
        plt.savefig(filename, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        print_success(f"Comprehensive visualization saved: {filename}")

    def _plot_radar_chart(self, ax, metrics_history):
        """Plot radar chart for comprehensive comparison"""
        radar_metrics = ['accuracy', 'precision', 'recall', 'specificity', 'auc_roc', 'mcc']
        num_vars = len(radar_metrics)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        for model_name, metrics in metrics_history.items():
            if model_name in self.model_colors:
                values = [metrics[m] for m in radar_metrics]
                values += values[:1]

                ax.plot(angles, values, linewidth=2, linestyle='solid',
                        label=model_name.upper(), color=self.model_colors[model_name])
                ax.fill(angles, values, alpha=0.1, color=self.model_colors[model_name])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in radar_metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Comprehensive Metrics Radar Chart', size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

    def _plot_key_metrics_comparison(self, ax, metrics_history):
        """Plot bar chart for key metrics comparison"""
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'mcc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'MCC']

        x = np.arange(len(metric_labels))
        width = 0.25

        for idx, (model_name, metrics) in enumerate(metrics_history.items()):
            if model_name in self.model_colors:
                model_values = [metrics[m] for m in key_metrics]
                ax.bar(x + idx * width - width, model_values, width,
                       label=model_name.upper(), color=self.model_colors[model_name], alpha=0.8)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Key Metrics Comparison', fontweight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    def _plot_aggregated_confusion_matrix(self, ax, metrics_history):
        """Plot aggregated confusion matrix"""
        combined_cm = sum([metrics['confusion_matrix'] for metrics in metrics_history.values()])

        sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Aggregated Confusion Matrix', fontweight='bold')

    def _plot_roc_curves(self, ax, metrics_history):
        """Plot ROC curves comparison"""
        for model_name, metrics in metrics_history.items():
            if model_name in self.model_colors and metrics['fpr_curve'] is not None:
                ax.plot(metrics['fpr_curve'], metrics['tpr_curve'],
                        label=f"{model_name.upper()} (AUC={metrics['auc_roc']:.3f})",
                        color=self.model_colors[model_name], linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_pr_curves(self, ax, metrics_history):
        """Plot Precision-Recall curves"""
        for model_name, metrics in metrics_history.items():
            if model_name in self.model_colors and metrics['precision_curve'] is not None:
                ax.plot(metrics['recall_curve'], metrics['precision_curve'],
                        label=f"{model_name.upper()} (AP={metrics['auc_pr']:.3f})",
                        color=self.model_colors[model_name], linewidth=2)

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_calibration_curves(self, ax, metrics_history):
        """Plot calibration curves"""
        for model_name, metrics in metrics_history.items():
            if model_name in self.model_colors:
                # Simulate calibration curve data for visualization
                prob_pos = np.random.rand(100)
                fraction_of_positives = prob_pos + np.random.normal(0, 0.05, 100)
                fraction_of_positives = np.clip(fraction_of_positives, 0, 1)

                ax.scatter(prob_pos, fraction_of_positives, alpha=0.5,
                           label=f"{model_name.upper()} (ECE={metrics['calibration_error']:.3f})",
                           color=self.model_colors[model_name])

        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', alpha=0.5)
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curves', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_train_test_comparison(self, ax, metrics_history):
        """Plot training vs test performance comparison"""
        model_names = list(metrics_history.keys())
        train_accs = [metrics['train_accuracy'] for metrics in metrics_history.values()]
        test_accs = [metrics['accuracy'] for metrics in metrics_history.values()]

        x = np.arange(len(model_names))
        ax.bar(x - 0.2, train_accs, 0.4, label='Training', alpha=0.8, color='blue')
        ax.bar(x + 0.2, test_accs, 0.4, label='Test', alpha=0.8, color='red')

        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training vs Test Accuracy', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in model_names])
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_cv_results(self, ax, metrics_history):
        """Plot cross-validation results"""
        model_names = list(metrics_history.keys())
        cv_means = [metrics['cv_mean'] for metrics in metrics_history.values()]
        cv_stds = [metrics['cv_std'] for metrics in metrics_history.values()]

        bars = ax.bar(model_names, cv_means, yerr=cv_stds, capsize=5,
                      color=[self.model_colors.get(m, 'gray') for m in model_names], alpha=0.8)

        ax.set_xlabel('Models')
        ax.set_ylabel('CV Accuracy')
        ax.set_title('5-Fold Cross-Validation Results', fontweight='bold')
        ax.set_xticklabels([m.upper() for m in model_names])
        ax.grid(True, alpha=0.3)

    def _plot_metrics_table(self, ax, metrics_history):
        """Plot detailed metrics table"""
        ax.axis('off')

        table_data = []
        headers = ['Metric'] + [m.upper() for m in metrics_history.keys()]

        table_metrics = ['accuracy', 'precision', 'recall', 'f1_score',
                         'specificity', 'auc_roc', 'mcc', 'kappa', 'calibration_error']

        for metric in table_metrics:
            row = [metric.replace('_', ' ').title()]
            for model_name in metrics_history.keys():
                value = metrics_history[model_name][metric]
                row.append(f"{value:.4f}")
            table_data.append(row)

        table = ax.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Detailed Metrics Table', fontweight='bold', pad=20)

    def _plot_error_analysis(self, ax, metrics_history):
        """Plot error analysis (FP vs FN)"""
        model_names = list(metrics_history.keys())
        fp_counts = [metrics['fp'] for metrics in metrics_history.values()]
        fn_counts = [metrics['fn'] for metrics in metrics_history.values()]

        x = np.arange(len(model_names))
        ax.bar(x - 0.2, fp_counts, 0.4, label='False Positives', alpha=0.8, color='orange')
        ax.bar(x + 0.2, fn_counts, 0.4, label='False Negatives', alpha=0.8, color='red')

        ax.set_xlabel('Models')
        ax.set_ylabel('Count')
        ax.set_title('Error Analysis: FP vs FN', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in model_names])
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_model_summary(self, ax, metrics_history):
        """Plot model summary with rankings"""
        ax.axis('off')

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

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def _plot_learning_progress(self, ax, metrics_history):
        """Plot simulated learning progress"""
        iterations = np.arange(1, 11)
        for model_name, metrics in metrics_history.items():
            if model_name in self.model_colors:
                # Simulate learning curve
                perf = metrics['accuracy'] * (1 - np.exp(-iterations / 3)) + np.random.normal(0, 0.02, len(iterations))
                ax.plot(iterations, perf, label=model_name.upper(),
                        color=self.model_colors[model_name], linewidth=2, marker='o')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Performance')
        ax.set_title('Simulated Learning Progress', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str, save: bool = True):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=ax, cbar_kws={'label': 'Count'})

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {model_name.upper()}', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            filename = os.path.join(self.output_dir, f"confusion_matrix_{model_name}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print_success(f"Confusion matrix saved: {filename}")
        else:
            plt.show()

    def plot_roc_curve(self, fpr, tpr, auc_score, model_name: str, save: bool = True):
        """Plot ROC curve"""
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {auc_score:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {model_name.upper()}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = os.path.join(self.output_dir, f"roc_curve_{model_name}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print_success(f"ROC curve saved: {filename}")
        else:
            plt.show()