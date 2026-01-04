"""Model evaluation module"""
import time
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    precision_recall_curve, matthews_corrcoef, cohen_kappa_score,
    log_loss, roc_auc_score, average_precision_score
)
from sklearn.model_selection import learning_curve, cross_val_score, StratifiedKFold
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

from config import Config


class ModelEvaluator:
    def __init__(self, config=None):
        self.config = config or Config()
        self.metrics_history = {}
        self.training_times = {}
        self.inference_times = {}

    def measure_inference_time(self, model, X_test, num_runs=100):
        """Measure inference time for a model"""
        if not hasattr(model, 'predict'):
            return 0

        # Warm up
        model.predict(X_test[:1])

        # Measure inference time
        start_time = time.perf_counter()
        for _ in range(num_runs):
            model.predict(X_test[:1])
        total_time = time.perf_counter() - start_time

        # Average inference time per image in milliseconds
        avg_inference_time_ms = (total_time / num_runs) * 1000
        return avg_inference_time_ms

    def measure_all_inference_times(self, X_test, models):
        """Measure inference times for all models"""
        print("\n📊 Measuring inference times...")

        for model_name, model in models.items():
            inference_time = self.measure_inference_time(model, X_test)
            self.inference_times[model_name] = inference_time
            print(f"  {model_name.upper()}: {inference_time:.2f} ms/image")

    def print_timing_summary(self):
        """Print timing summary table"""
        if not self.training_times or not self.inference_times:
            print("No timing data available.")
            return

        print("\n" + "=" * 60)
        print("MODEL TRAINING AND INFERENCE TIMING SUMMARY")
        print("=" * 60)

        table_data = []
        for model_name in self.inference_times.keys():
            train_time = self.training_times.get(model_name, 0)
            inf_time = self.inference_times.get(model_name, 0)
            table_data.append([
                model_name.upper(),
                f"{train_time:.2f} s",
                f"{inf_time:.2f} ms"
            ])

        print(tabulate(table_data,
                       headers=["Model", "Training Time", "Inference Time/Image"],
                       tablefmt="grid"))

    def print_confusion_matrix(self, cm, model_name, class_names):
        """Print confusion matrix with metrics"""
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

    def print_classification_report(self, y_true, y_pred, model_name, class_names):
        """Print detailed classification report in terminal"""
        print(f"\n{model_name.upper()} - Classification Report:")
        print("-" * 60)

        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

        # Prepare table data
        table_data = []
        for class_name in class_names:
            class_report = report[class_name]
            table_data.append([
                class_name,
                f"{class_report['precision']:.4f}",
                f"{class_report['recall']:.4f}",
                f"{class_report['f1-score']:.4f}",
                f"{int(class_report['support'])}"
            ])

        # Add averages
        table_data.append([
            "Weighted Avg",
            f"{report['weighted avg']['precision']:.4f}",
            f"{report['weighted avg']['recall']:.4f}",
            f"{report['weighted avg']['f1-score']:.4f}",
            f"{report['weighted avg']['support']}"
        ])

        print(tabulate(table_data, headers=["Class", "Precision", "Recall", "F1-Score", "Support"], tablefmt="grid"))

        # Overall accuracy
        accuracy = report['accuracy']
        print(f"\nOverall Accuracy: {accuracy:.4f}")

    def print_learning_curves(self, model, X, y, model_name):
        """Print learning curve analysis in terminal"""
        print(f"\n{model_name.upper()} - Learning Curve Analysis:")
        print("-" * 60)

        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5,
            train_sizes=np.linspace(0.1, 1.0, 5),
            scoring='accuracy', n_jobs=-1, random_state=self.config.RANDOM_STATE
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

        # Calculate gap (overfitting indicator)
        final_gap = train_scores_mean[-1] - test_scores_mean[-1]
        print(f"\nFinal Overfitting Gap: {final_gap:.4f}")
        if final_gap > 0.1:
            print("⚠️  High overfitting detected!")
        elif final_gap > 0.05:
            print("⚠️  Moderate overfitting detected.")
        else:
            print("✓ Low overfitting - good generalization.")

    def print_cross_validation_scores(self, model, X, y, model_name):
        """Print cross-validation scores in terminal"""
        print(f"\n{model_name.upper()} - Cross-Validation Analysis:")
        print("-" * 60)

        cv_methods = [
            ("Stratified 5-Fold", StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.RANDOM_STATE)),
        ]

        for method_name, cv in cv_methods:
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

            table_data = []
            for fold, score in enumerate(scores, 1):
                table_data.append([f"Fold {fold}", f"{score:.4f}"])

            table_data.append(["", ""])
            table_data.append(["Mean", f"{np.mean(scores):.4f}"])
            table_data.append(["Std Dev", f"{np.std(scores):.4f}"])
            table_data.append(["Min", f"{np.min(scores):.4f}"])
            table_data.append(["Max", f"{np.max(scores):.4f}"])

            print(f"\n{method_name}:")
            print(tabulate(table_data, headers=["Fold", "Accuracy"], tablefmt="grid"))

    def print_bootstrapping_metrics(self, model, X, y, model_name, n_bootstraps=100):
        """Print bootstrapping metrics in terminal"""
        print(f"\n{model_name.upper()} - Bootstrapping Metrics (n={n_bootstraps}):")
        print("-" * 60)

        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

        for _ in tqdm(range(n_bootstraps), desc=f"Bootstrapping {model_name}"):
            X_resampled, y_resampled = resample(X, y, random_state=np.random.randint(1000))

            # Train-test split
            X_train_bs, X_test_bs, y_train_bs, y_test_bs = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=self.config.RANDOM_STATE
            )

            # Train model on bootstrap sample
            model_bs = model.__class__(**model.get_params())
            model_bs.fit(X_train_bs, y_train_bs)

            # Predict and calculate metrics
            y_pred_bs = model_bs.predict(X_test_bs)

            metrics['accuracy'].append(accuracy_score(y_test_bs, y_pred_bs))
            metrics['precision'].append(precision_score(y_test_bs, y_pred_bs, average='weighted'))
            metrics['recall'].append(recall_score(y_test_bs, y_pred_bs, average='weighted'))
            metrics['f1'].append(f1_score(y_test_bs, y_pred_bs, average='weighted'))

        # Calculate statistics
        table_data = []
        for metric_name, values in metrics.items():
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

    def calculate_all_metrics(self, model, X_train, X_test, y_train, y_test, model_name):
        """Calculate all comprehensive metrics for a model"""
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
        fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
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
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

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
            'fpr': fpr_rate,
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
            'training_time': self.training_times.get(model_name, 0),
            'inference_time': self.inference_times.get(model_name, 0)
        }

        self.metrics_history[model_name] = metrics
        return metrics

    def print_all_metrics_terminal(self, metrics, class_names):
        """Print all comprehensive metrics in terminal using tabulate"""
        model_name = metrics['model_name']

        print(f"\n{'=' * 80}")
        print(f"{model_name.upper()} - COMPREHENSIVE METRICS ANALYSIS")
        print(f"{'=' * 80}")

        # Print timing information first
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
        self.print_confusion_matrix(metrics['confusion_matrix'], model_name, class_names)

        print(f"\n{'=' * 80}")

    def evaluate_all_models(self, X_train, X_test, y_train, y_test, models, class_names):
        """Evaluate all models comprehensively"""
        print("\n" + "=" * 100)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 100)

        # Measure inference times first
        self.measure_all_inference_times(X_test, models)

        results = {}

        for model_name, model in models.items():
            print(f"\n{'=' * 80}")
            print(f"EVALUATING {model_name.upper()}")
            print(f"{'=' * 80}")

            # Calculate all metrics
            metrics = self.calculate_all_metrics(model, X_train, X_test, y_train, y_test, model_name)

            # Print all metrics in terminal
            self.print_all_metrics_terminal(metrics, class_names)

            # Print confusion matrix
            self.print_confusion_matrix(metrics['confusion_matrix'], model_name, class_names)

            # Print classification report
            y_pred = model.predict(X_test)
            self.print_classification_report(y_test, y_pred, model_name, class_names)

            # Print learning curves
            self.print_learning_curves(model, X_train, y_train, model_name)

            # Print cross-validation scores
            self.print_cross_validation_scores(model, X_train, y_train, model_name)

            # Print bootstrapping metrics
            self.print_bootstrapping_metrics(model, X_train, y_train, model_name,
                                           n_bootstraps=self.config.N_BOOTSTRAPS)

            results[model_name] = metrics

        # Print timing summary
        self.print_timing_summary()

        return results

    def get_metrics_history(self):
        """Get metrics history"""
        return self.metrics_history

    def get_timing_data(self):
        """Get timing data"""
        return self.training_times, self.inference_times