"""Model evaluation module"""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, precision_recall_curve,
                             matthews_corrcoef, cohen_kappa_score, log_loss,
                             roc_auc_score, average_precision_score, classification_report)
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import time

from config import config
from utils import print_header, print_success, print_info, print_warning


class ModelEvaluator:
    """Comprehensive model evaluation class"""

    def __init__(self, class_names: List[str] = None):
        """
        Initialize evaluator

        Args:
            class_names: Names of classes
        """
        self.class_names = class_names or config.CLASS_NAMES
        self.metrics_history = {}

    def calculate_all_metrics(self, model, X_train, X_test, y_train, y_test,
                              model_name: str, training_time: float = 0,
                              inference_time: float = 0) -> Dict:
        """
        Calculate all comprehensive metrics for a model

        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            model_name: Name of the model
            training_time: Training time in seconds
            inference_time: Inference time in ms per image

        Returns:
            Dictionary of all metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # Probabilities (if available)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            y_proba_train = model.predict_proba(X_train)[:, 1]
        else:
            y_proba = None
            y_proba_train = None

        # 1. Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # 2. Confusion matrix metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # 3. Advanced metrics
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        # 4. AUC metrics
        if y_proba is not None:
            auc_roc = roc_auc_score(y_test, y_proba)
            auc_pr = average_precision_score(y_test, y_proba)
            logloss = log_loss(y_test, y_proba)

            # ROC curve data
            fpr_curve, tpr_curve, _ = roc_curve(y_test, y_proba)
            # PR curve data
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        else:
            auc_roc = 0
            auc_pr = 0
            logloss = 0
            fpr_curve, tpr_curve = None, None
            precision_curve, recall_curve = None, None

        # 5. Calibration metrics
        if y_proba is not None:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_proba, n_bins=10, strategy='uniform'
            )
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        else:
            calibration_error = 0

        # 6. Training metrics (for overfitting analysis)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        overfitting_gap = train_accuracy - accuracy

        # 7. Cross-validation metrics
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
        except:
            cv_mean = 0
            cv_std = 0
            print_warning("Cross-validation failed for this model")

        # Store all metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'log_loss': logloss,
            'mcc': mcc,
            'kappa': kappa,
            'ppv': ppv,
            'npv': npv,
            'fpr': fpr,
            'fnr': fnr,
            'calibration_error': calibration_error,
            'train_accuracy': train_accuracy,
            'overfitting_gap': overfitting_gap,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'fpr_curve': fpr_curve,
            'tpr_curve': tpr_curve,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'confusion_matrix': cm,
            'training_time': training_time,
            'inference_time': inference_time
        }

        self.metrics_history[model_name] = metrics
        return metrics

    def print_confusion_matrix(self, cm: np.ndarray, model_name: str):
        """Print confusion matrix in terminal"""
        print(f"\n{model_name.upper()} - Confusion Matrix:")
        print("-" * 40)

        cm_table = [
            [f"TP: {cm[1, 1]}", f"FP: {cm[0, 1]}"],
            [f"FN: {cm[1, 0]}", f"TN: {cm[0, 0]}"]
        ]

        print(tabulate(cm_table, headers=["Predicted Positive", "Predicted Negative"],
                       showindex=["Actual Positive", "Actual Negative"], tablefmt="grid"))

        tn, fp, fn, tp = cm.ravel()

        metrics_table = [
            ["Sensitivity (Recall)", f"{tp / (tp + fn):.4f}" if (tp + fn) > 0 else "N/A"],
            ["Specificity", f"{tn / (tn + fp):.4f}" if (tn + fp) > 0 else "N/A"],
            ["PPV (Precision)", f"{tp / (tp + fp):.4f}" if (tp + fp) > 0 else "N/A"],
            ["NPV", f"{tn / (tn + fn):.4f}" if (tn + fn) > 0 else "N/A"],
            ["FPR", f"{fp / (fp + tn):.4f}" if (fp + tn) > 0 else "N/A"],
            ["FNR", f"{fn / (fn + tp):.4f}" if (fn + tp) > 0 else "N/A"]
        ]

        print("\nConfusion Matrix Derived Metrics:")
        print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

    def print_classification_report(self, y_true, y_pred, model_name: str):
        """Print detailed classification report"""
        print(f"\n{model_name.upper()} - Classification Report:")
        print("-" * 60)

        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)

        table_data = []
        for class_name in self.class_names:
            class_report = report[class_name]
            table_data.append([
                class_name,
                f"{class_report['precision']:.4f}",
                f"{class_report['recall']:.4f}",
                f"{class_report['f1-score']:.4f}",
                f"{int(class_report['support'])}"
            ])

        table_data.append([
            "Weighted Avg",
            f"{report['weighted avg']['precision']:.4f}",
            f"{report['weighted avg']['recall']:.4f}",
            f"{report['weighted avg']['f1-score']:.4f}",
            f"{report['weighted avg']['support']}"
        ])

        print(tabulate(table_data, headers=["Class", "Precision", "Recall", "F1-Score", "Support"],
                       tablefmt="grid"))

        accuracy = report['accuracy']
        print(f"\nOverall Accuracy: {accuracy:.4f}")

    def print_learning_curves(self, model, X, y, model_name: str):
        """Print learning curve analysis"""
        print(f"\n{model_name.upper()} - Learning Curve Analysis:")
        print("-" * 60)

        try:
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=5,
                train_sizes=np.linspace(0.1, 1.0, 5),
                scoring='accuracy', n_jobs=-1, random_state=config.RANDOM_STATE
            )

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            table_data = []
            for i, size in enumerate(train_sizes):
                table_data.append([
                    f"{int(size)}",
                    f"{train_scores_mean[i]:.4f}",
                    f"±{train_scores_std[i]:.4f}",
                    f"{test_scores_mean[i]:.4f}",
                    f"±{test_scores_std[i]:.4f}"
                ])

            print(tabulate(table_data,
                           headers=["Training Size", "Train Score", "±Std", "Val Score", "±Std"],
                           tablefmt="grid"))

            final_gap = train_scores_mean[-1] - test_scores_mean[-1]
            print(f"\nFinal Overfitting Gap: {final_gap:.4f}")
            if final_gap > 0.1:
                print("⚠️  High overfitting detected!")
            elif final_gap > 0.05:
                print("⚠️  Moderate overfitting detected.")
            else:
                print("✓ Low overfitting - good generalization.")
        except Exception as e:
            print_warning(f"Could not compute learning curves: {e}")

    def print_bootstrapping_metrics(self, model, X, y, model_name: str, n_bootstraps: int = None):
        """Print bootstrapping metrics"""
        n_bootstraps = n_bootstraps or config.BOOTSTRAP_ITERATIONS

        print(f"\n{model_name.upper()} - Bootstrapping Metrics (n={n_bootstraps}):")
        print("-" * 60)

        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

        for _ in tqdm(range(n_bootstraps), desc=f"Bootstrapping {model_name}"):
            X_resampled, y_resampled = resample(X, y, random_state=np.random.randint(1000))

            X_train_bs, X_test_bs, y_train_bs, y_test_bs = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=config.RANDOM_STATE
            )

            try:
                model_bs = model.__class__(**model.get_params())
                model_bs.fit(X_train_bs, y_train_bs)

                y_pred_bs = model_bs.predict(X_test_bs)

                metrics['accuracy'].append(accuracy_score(y_test_bs, y_pred_bs))
                metrics['precision'].append(precision_score(y_test_bs, y_pred_bs, average='weighted'))
                metrics['recall'].append(recall_score(y_test_bs, y_pred_bs, average='weighted'))
                metrics['f1'].append(f1_score(y_test_bs, y_pred_bs, average='weighted'))
            except:
                continue

        table_data = []
        for metric_name, values in metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                ci_lower = np.percentile(values, 2.5)
                ci_upper = np.percentile(values, 97.5)

                table_data.append([
                    metric_name.title(),
                    f"{mean_val:.4f}",
                    f"{std_val:.4f}",
                    f"[{ci_lower:.4f}, {ci_upper:.4f}]"
                ])

        print(tabulate(table_data,
                       headers=["Metric", "Mean", "Std Dev", "95% CI"],
                       tablefmt="grid"))

    def print_all_metrics_terminal(self, metrics: Dict):
        """Print all comprehensive metrics in terminal"""
        model_name = metrics['model_name']

        print(f"\n{'=' * 80}")
        print(f"{model_name.upper()} - COMPREHENSIVE METRICS ANALYSIS")
        print(f"{'=' * 80}")

        # Timing information
        print("\n⏱️  TIMING INFORMATION:")
        print("-" * 40)

        timing_table = [
            ["Training Time", f"{metrics['training_time']:.2f} seconds"],
            ["Inference Time", f"{metrics['inference_time']:.2f} ms/image"]
        ]
        print(tabulate(timing_table, headers=["Metric", "Value"], tablefmt="grid"))

        # 1. Basic Performance Metrics
        print("\n📊 1. BASIC PERFORMANCE METRICS:")
        print("-" * 60)

        basic_metrics = [
            ["Accuracy", f"{metrics['accuracy']:.4f}", "Overall fraction of correct predictions"],
            ["Precision", f"{metrics['precision']:.4f}", "How many predicted positives were actually positive?"],
            ["Recall (Sensitivity)", f"{metrics['recall']:.4f}", "How many actual positives were detected?"],
            ["F1-Score", f"{metrics['f1_score']:.4f}", "Harmonic mean of precision and recall"],
            ["Specificity", f"{metrics['specificity']:.4f}", "How many actual negatives were correctly identified?"]
        ]

        print(tabulate(basic_metrics, headers=["Metric", "Value", "Description"], tablefmt="grid"))

        # 2. Advanced Statistical Metrics
        print("\n📈 2. ADVANCED STATISTICAL METRICS:")
        print("-" * 60)

        advanced_metrics = [
            ["AUC-ROC", f"{metrics['auc_roc']:.4f}", "Area under ROC curve - separability measure"],
            ["AUC-PR", f"{metrics['auc_pr']:.4f}", "Area under PR curve - imbalanced data"],
            ["Log Loss", f"{metrics['log_loss']:.4f}", "Cross-entropy loss with probabilities"],
            ["MCC", f"{metrics['mcc']:.4f}", "Matthews Correlation Coefficient"],
            ["Cohen's Kappa", f"{metrics['kappa']:.4f}", "Agreement between predictions and true labels"]
        ]

        print(tabulate(advanced_metrics, headers=["Metric", "Value", "Description"], tablefmt="grid"))

        # 3. Clinical/Diagnostic Metrics
        print("\n🏥 3. CLINICAL/DIAGNOSTIC METRICS:")
        print("-" * 60)

        clinical_metrics = [
            ["PPV (Positive Predictive Value)", f"{metrics['ppv']:.4f}",
             "Probability that positive prediction is correct"],
            ["NPV (Negative Predictive Value)", f"{metrics['npv']:.4f}",
             "Probability that negative prediction is correct"],
            ["FPR (False Positive Rate)", f"{metrics['fpr']:.4f}", "Type I error rate"],
            ["FNR (False Negative Rate)", f"{metrics['fnr']:.4f}", "Type II error rate"],
            ["Calibration Error", f"{metrics['calibration_error']:.4f}", "Deviation from perfect calibration"]
        ]

        print(tabulate(clinical_metrics, headers=["Metric", "Value", "Description"], tablefmt="grid"))

        # 4. Model Robustness Metrics
        print("\n🔧 4. MODEL ROBUSTNESS METRICS:")
        print("-" * 60)

        robustness_metrics = [
            ["Training Accuracy", f"{metrics['train_accuracy']:.4f}", "Accuracy on training set"],
            ["Overfitting Gap", f"{metrics['overfitting_gap']:.4f}", "Train vs Test accuracy difference"],
            ["CV Mean Accuracy", f"{metrics['cv_mean']:.4f}", "5-fold cross-validation mean"],
            ["CV Std Dev", f"{metrics['cv_std']:.4f}", "Cross-validation standard deviation"]
        ]

        print(tabulate(robustness_metrics, headers=["Metric", "Value", "Description"], tablefmt="grid"))

        # 5. Confusion Matrix Details
        print("\n🎯 5. CONFUSION MATRIX DETAILS:")
        print("-" * 60)

        confusion_details = [
            ["True Positives (TP)", metrics['tp'], "Correct pneumonia predictions"],
            ["True Negatives (TN)", metrics['tn'], "Correct normal predictions"],
            ["False Positives (FP)", metrics['fp'], "Normal incorrectly predicted as pneumonia"],
            ["False Negatives (FN)", metrics['fn'], "Pneumonia incorrectly predicted as normal"]
        ]

        print(tabulate(confusion_details, headers=["Category", "Count", "Description"], tablefmt="grid"))

        # Print confusion matrix
        self.print_confusion_matrix(metrics['confusion_matrix'], model_name)

        print(f"\n{'=' * 80}")

    def print_summary_table(self):
        """Print comprehensive summary table of all models"""
        if not self.metrics_history:
            print_warning("No metrics found. Run calculate_all_metrics first.")
            return

        print("\n" + "=" * 100)
        print("COMPREHENSIVE MODEL COMPARISON SUMMARY")
        print("=" * 100)

        table_data = []
        headers = ["Metric", "Description"] + [m.upper() for m in self.metrics_history.keys()]

        # Timing metrics
        timing_metrics = [
            ("training_time", "Training time in seconds", "s"),
            ("inference_time", "Inference time per image in milliseconds", "ms")
        ]

        for metric, description, unit in timing_metrics:
            row = [f"{metric.upper().replace('_', ' ')} ({unit})", description]
            for model_name in self.metrics_history.keys():
                value = self.metrics_history[model_name].get(metric, 0)
                if metric in ['training_time', 'inference_time']:
                    row.append(f"{value:.2f}")
                else:
                    row.append(f"{value:.4f}")
            table_data.append(row)

        # Other metrics
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
            for model_name in self.metrics_history.keys():
                value = self.metrics_history[model_name].get(metric, 0)
                row.append(f"{value:.4f}")
            table_data.append(row)

        print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))

        # Model rankings
        self.print_model_rankings()

    def print_model_rankings(self):
        """Print model rankings"""
        print("\n" + "=" * 100)
        print("MODEL RANKINGS")
        print("=" * 100)

        ranking_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity',
                           'auc_roc', 'mcc', 'kappa', 'calibration_error']

        ranking_data = []
        for metric in ranking_metrics:
            sorted_models = sorted(self.metrics_history.items(),
                                   key=lambda x: x[1].get(metric, 0), reverse=True)
            ranking_row = [metric.upper().replace('_', ' ')]
            for i, (model_name, _) in enumerate(sorted_models, 1):
                ranking_row.append(f"{i}. {model_name.upper()}")
            ranking_data.append(ranking_row)

        ranking_headers = ["Metric"] + [f"Rank {i}" for i in range(1, len(self.metrics_history) + 1)]
        print(tabulate(ranking_data, headers=ranking_headers, tablefmt="grid"))

        # Overall assessment
        self.print_overall_assessment()

    def print_overall_assessment(self):
        """Print overall model assessment"""
        print("\n" + "=" * 100)
        print("OVERALL ASSESSMENT")
        print("=" * 100)

        # Calculate average rank
        avg_ranks = {}
        ranking_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity',
                           'auc_roc', 'mcc', 'kappa']

        for model_name in self.metrics_history.keys():
            total_rank = 0
            count = 0
            for metric in ranking_metrics:
                sorted_models = sorted(self.metrics_history.items(),
                                       key=lambda x: x[1].get(metric, 0), reverse=True)
                rank = [m[0] for m in sorted_models].index(model_name) + 1
                total_rank += rank
                count += 1
            avg_ranks[model_name] = total_rank / count if count > 0 else 0

        best_model = min(avg_ranks, key=avg_ranks.get) if avg_ranks else None
        worst_model = max(avg_ranks, key=avg_ranks.get) if avg_ranks else None

        assessment_data = [
            ["Best Overall Model", best_model.upper() if best_model else "N/A",
             f"Avg Rank: {avg_ranks[best_model]:.2f}" if best_model else "N/A"],
            ["Worst Overall Model", worst_model.upper() if worst_model else "N/A",
             f"Avg Rank: {avg_ranks[worst_model]:.2f}" if worst_model else "N/A"],
            ["Recommendation", f"Use {best_model.upper()}" if best_model else "N/A",
             "Highest overall performance"],
            ["Best for Clinical Use",
             "Model with highest specificity" if any(
                 m['specificity'] > 0.9 for m in self.metrics_history.values()) else "All models",
             "Minimize false positives"]
        ]

        print(tabulate(assessment_data, headers=["Aspect", "Model", "Reason"], tablefmt="grid"))