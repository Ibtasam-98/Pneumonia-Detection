# chest_xray_ml_predictor.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, roc_curve, auc, precision_recall_curve, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm
import argparse
from datetime import datetime


class ChestXRayMLPredictor:
    def __init__(self, img_height=100, img_width=100):
        self.img_height = img_height
        self.img_width = img_width
        self.scaler = StandardScaler()
        self.pca = None
        self.models = {}
        self.class_names = ['NORMAL', 'PNEUMONIA']
        self.features = None
        self.labels = None
        self.results = {}
        self.visualization_dir = "visualizations"

        # Create visualization directory
        os.makedirs(self.visualization_dir, exist_ok=True)

    def debug_dataset_structure(self, data_dir='dataset/chest_xray'):
        """Debug function to check dataset structure"""
        print("\n🔍 Debugging dataset structure...")

        if not os.path.exists(data_dir):
            print(f"❌ Main dataset directory not found: {data_dir}")
            return False

        splits = ['train', 'test', 'val']

        for split in splits:
            split_path = os.path.join(data_dir, split)
            print(f"\nChecking {split_path}:")

            if os.path.exists(split_path):
                print(f"  ✅ Found")
                for class_name in self.class_names:
                    class_path = os.path.join(split_path, class_name)
                    if os.path.exists(class_path):
                        image_files = [f for f in os.listdir(class_path)
                                       if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
                        print(f"    {class_name}: {len(image_files)} images")
                    else:
                        print(f"    ❌ {class_name}: directory not found")
            else:
                print(f"  ❌ Not found")

        return True

    def load_and_preprocess_images(self, data_dir='dataset/chest_xray'):
        """Load and preprocess images from directory"""
        print("Loading and preprocessing images...")

        features = []
        labels = []

        # Define the splits to process
        splits = ['train', 'test', 'val']

        for split in splits:
            split_path = os.path.join(data_dir, split)
            if not os.path.exists(split_path):
                print(f"Warning: {split_path} not found, skipping...")
                continue

            for class_name in self.class_names:
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    print(f"Warning: {class_path} not found, skipping...")
                    continue

                print(f"Processing {split}/{class_name}...")

                # Get all image files
                image_files = [f for f in os.listdir(class_path)
                               if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

                for image_file in tqdm(image_files, desc=f"{split}/{class_name}"):
                    img_path = os.path.join(class_path, image_file)

                    try:
                        # Load and preprocess image
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"Warning: Could not load image {img_path}")
                            continue

                        # Convert to RGB and resize
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.img_width, self.img_height))

                        # Extract features (flatten image)
                        img_flattened = img.flatten()

                        features.append(img_flattened)
                        labels.append(class_name)

                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue

        if len(features) == 0:
            print("❌ No images were loaded. Please check your dataset structure.")
            return None, None

        self.features = np.array(features)
        self.labels = np.array(labels)

        print(f"✅ Loaded {len(features)} images")
        print(f"Class distribution:")

        # Fixed: Properly count class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        class_distribution = dict(zip(unique, counts))

        for class_name, count in class_distribution.items():
            percentage = (count / len(self.labels)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")

        return self.features, self.labels

    def extract_advanced_features(self, images):
        """Extract advanced features from images"""
        print("Extracting advanced features...")

        advanced_features = []

        for img_flat in tqdm(images, desc="Feature extraction"):
            # Reshape back to image
            img = img_flat.reshape(self.img_height, self.img_width, 3)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            feature_vector = []

            # 1. Histogram features
            hist = cv2.calcHist([img_gray], [0], None, [16], [0, 256])
            feature_vector.extend(hist.flatten())

            # 2. Statistical features
            feature_vector.append(np.mean(img_gray))
            feature_vector.append(np.std(img_gray))
            feature_vector.append(np.median(img_gray))

            # 3. Texture features using GLCM-like properties
            sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            feature_vector.append(np.mean(sobelx))
            feature_vector.append(np.mean(sobely))
            feature_vector.append(np.std(sobelx))
            feature_vector.append(np.std(sobely))

            # 4. Edge density
            edges = cv2.Canny(img_gray, 50, 150)
            feature_vector.append(np.sum(edges > 0) / edges.size)

            advanced_features.append(feature_vector)

        return np.array(advanced_features)

    def preprocess_data(self, use_advanced_features=False, use_pca=False, n_components=50):
        """Preprocess the data for ML models"""
        print("Preprocessing data...")

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.labels)

        if use_advanced_features:
            X_processed = self.extract_advanced_features(self.features)
        else:
            X_processed = self.features

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Apply PCA if requested
        if use_pca:
            print(f"Applying PCA with {n_components} components...")
            self.pca = PCA(n_components=n_components, random_state=42)
            X_train_final = self.pca.fit_transform(X_train_scaled)
            X_test_final = self.pca.transform(X_test_scaled)
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            print(f"Explained variance ratio: {explained_variance:.4f}")
        else:
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled

        print(f"Training set shape: {X_train_final.shape}")
        print(f"Test set shape: {X_test_final.shape}")

        return X_train_final, X_test_final, y_train, y_test

    def train_svm(self, X_train, y_train, cv_tuning=True):
        """Train SVM with RBF kernel"""
        print("\n" + "=" * 50)
        print("TRAINING SVM WITH RBF KERNEL")
        print("=" * 50)

        if cv_tuning:
            # Hyperparameter tuning with GridSearchCV
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf']
            }

            svm = SVC(random_state=42, probability=True)
            grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            best_svm = grid_search.best_estimator_
            print(f"Best SVM parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        else:
            # Use default parameters
            best_svm = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42, probability=True)
            best_svm.fit(X_train, y_train)

        self.models['svm'] = best_svm
        return best_svm

    def train_knn(self, X_train, y_train, cv_tuning=True):
        """Train KNN classifier"""
        print("\n" + "=" * 50)
        print("TRAINING K-NEAREST NEIGHBORS")
        print("=" * 50)

        if cv_tuning:
            # Hyperparameter tuning with GridSearchCV
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }

            knn = KNeighborsClassifier()
            grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            best_knn = grid_search.best_estimator_
            print(f"Best KNN parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        else:
            # Use default parameters
            best_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
            best_knn.fit(X_train, y_train)

        self.models['knn'] = best_knn
        return best_knn

    def train_random_forest(self, X_train, y_train, cv_tuning=True):
        """Train Random Forest classifier"""
        print("\n" + "=" * 50)
        print("TRAINING RANDOM FOREST")
        print("=" * 50)

        if cv_tuning:
            # Hyperparameter tuning with GridSearchCV
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            best_rf = grid_search.best_estimator_
            print(f"Best Random Forest parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        else:
            # Use default parameters
            best_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            best_rf.fit(X_train, y_train)

        self.models['random_forest'] = best_rf
        return best_rf

    def create_research_visualizations(self, X_train, y_train, X_test, y_test):
        """Create publication-ready visualizations for research paper"""
        print("\n📊 Creating research paper visualizations...")

        # Set publication quality style
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'figure.figsize': (10, 8)
        })

        # 1. Create combined learning curves visualization
        self._create_combined_learning_curves(X_train, y_train)

        # 2. Create combined calibration curves visualization
        self._create_combined_calibration_curves(X_test, y_test)

        print("✅ Research paper visualizations created successfully!")

    def _create_combined_learning_curves(self, X, y):
        """Create publication-ready combined learning curves"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Define colors and markers for different models
        model_styles = {
            'svm': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'linewidth': 2},
            'knn': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--', 'linewidth': 2},
            'random_forest': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.', 'linewidth': 2}
        }

        for model_name, model in self.models.items():
            if model_name in model_styles:
                style = model_styles[model_name]

                train_sizes, train_scores, test_scores = learning_curve(
                    model, X, y, cv=5,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring='accuracy', n_jobs=-1, random_state=42
                )

                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)

                # Plot training scores
                ax.plot(train_sizes, train_scores_mean,
                        label=f'{model_name.upper()} (Training)',
                        **style)

                # Plot validation scores with different linestyle
                val_style = style.copy()
                val_style['linestyle'] = ':'
                ax.plot(train_sizes, test_scores_mean,
                        label=f'{model_name.upper()} (Validation)',
                        **val_style)

                # Add shaded areas for standard deviation
                ax.fill_between(train_sizes,
                                train_scores_mean - train_scores_std,
                                train_scores_mean + train_scores_std,
                                alpha=0.1, color=style['color'])
                ax.fill_between(train_sizes,
                                test_scores_mean - test_scores_std,
                                test_scores_mean + test_scores_std,
                                alpha=0.1, color=style['color'])

        # Customize the plot
        ax.set_xlabel('Number of Training Examples', fontweight='bold')
        ax.set_ylabel('Accuracy Score', fontweight='bold')
        ax.set_title('Learning Curves Comparison for Pneumonia Detection Models',
                     fontsize=16, fontweight='bold', pad=20)

        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0.5, 1.05)

        # Add minor grid
        ax.grid(True, which='minor', alpha=0.2, linestyle=':')

        # Improve layout
        plt.tight_layout()

        # Save with high quality
        filename = os.path.join(self.visualization_dir, "research_learning_curves.png")
        plt.savefig(filename, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none',
                    transparent=False)
        plt.close()

        print(f"📈 Learning curves saved: {filename}")

    def _create_combined_calibration_curves(self, X_test, y_test):
        """Create publication-ready combined calibration curves"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Define colors and markers for different models
        model_styles = {
            'svm': {'color': '#1f77b4', 'marker': 'o', 'markersize': 6},
            'knn': {'color': '#ff7f0e', 'marker': 's', 'markersize': 6},
            'random_forest': {'color': '#2ca02c', 'marker': '^', 'markersize': 6}
        }

        calibration_results = {}

        for model_name, model in self.models.items():
            if model_name in model_styles:
                style = model_styles[model_name]

                if hasattr(model, 'predict_proba'):
                    prob_pos = model.predict_proba(X_test)[:, 1]
                else:
                    prob_pos = model.decision_function(X_test)
                    prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test, prob_pos, n_bins=10, strategy='uniform'
                )

                # Calculate calibration error
                calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                calibration_results[model_name] = calibration_error

                # Plot calibration curve
                ax.plot(mean_predicted_value, fraction_of_positives,
                        label=f'{model_name.upper()} (ECE: {calibration_error:.3f})',
                        **style, linewidth=2)

        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated",
                linewidth=2, alpha=0.8)

        # Customize the plot
        ax.set_xlabel('Mean Predicted Probability', fontweight='bold')
        ax.set_ylabel('Fraction of Positives', fontweight='bold')
        ax.set_title('Calibration Curves Comparison for Pneumonia Detection Models',
                     fontsize=16, fontweight='bold', pad=20)

        ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add diagonal reference
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)

        # Add minor grid
        ax.grid(True, which='minor', alpha=0.2, linestyle=':')

        # Add reliability diagram elements
        ax.set_aspect('equal', adjustable='box')

        # Improve layout
        plt.tight_layout()

        # Save with high quality
        filename = os.path.join(self.visualization_dir, "research_calibration_curves.png")
        plt.savefig(filename, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none',
                    transparent=False)
        plt.close()

        print(f"📊 Calibration curves saved: {filename}")
        return calibration_results

    def plot_roc_curves(self, X_test, y_test):
        """Plot ROC curves for all models"""
        # Set publication quality style
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300
        })

        plt.figure(figsize=(10, 8))

        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = model.decision_function(X_test)

            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2, label=f'{model_name.upper()} (AUC = {roc_auc:.4f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('ROC Curves - All Models', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        filename = os.path.join(self.visualization_dir, "roc_curves_all_models.png")
        plt.savefig(filename, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none',
                    transparent=False)
        plt.close()
        print(f"📊 ROC curves saved: {filename}")

    def plot_precision_recall_curves(self, X_test, y_test):
        """Plot Precision-Recall curves for all models"""
        # Set publication quality style
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300
        })

        plt.figure(figsize=(10, 8))

        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = model.decision_function(X_test)

            precision, recall, _ = precision_recall_curve(y_test, y_score)
            avg_precision = auc(recall, precision)

            plt.plot(recall, precision, lw=2, label=f'{model_name.upper()} (AP = {avg_precision:.4f})')

        plt.xlabel('Recall', fontweight='bold')
        plt.ylabel('Precision', fontweight='bold')
        plt.title('Precision-Recall Curves - All Models', fontsize=16, fontweight='bold')
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)

        filename = os.path.join(self.visualization_dir, "precision_recall_curves_all_models.png")
        plt.savefig(filename, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none',
                    transparent=False)
        plt.close()
        print(f"📊 Precision-Recall curves saved: {filename}")

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model"""
        print(f"\nEvaluating {model_name}...")

        # Make predictions
        y_pred = model.predict(X_test)

        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = None

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)

        # Calculate AUC-ROC
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = auc(fpr, tpr)
        else:
            auc_score = 0.0

        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Calculate PPV and NPV
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        print(f"{model_name.upper()} Results:")
        print(f"  Accuracy:          {accuracy:.4f}")
        print(f"  AUC-ROC:           {auc_score:.4f}")
        print(f"  Sensitivity:       {recall:.4f}")
        print(f"  Specificity:       {specificity:.4f}")
        print(f"  F1-Score:          {f1:.4f}")
        print(f"  FP/FN:             {fp}/{fn}")
        print(f"  PPV:               {ppv:.4f}")
        print(f"  NPV:               {npv:.4f}")
        print(f"  MCC:               {mcc:.4f}")

        # Classification report
        print(f"\nClassification Report for {model_name.upper()}:")
        report = classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)
        print(classification_report(y_test, y_pred, target_names=self.class_names))

        return {
            'accuracy': accuracy,
            'auc_roc': auc_score,
            'sensitivity': recall,
            'specificity': specificity,
            'f1_score': f1,
            'fp_fn': f"{fp}/{fn}",
            'ppv': ppv,
            'npv': npv,
            'mcc': mcc,
            'classification_report': report
        }

    def compare_models(self, X_train, y_train, X_test, y_test):
        """Compare performance of all trained models"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)

        results = {}

        for model_name, model in self.models.items():
            results[model_name] = self.evaluate_model(model, X_test, y_test, model_name)

        # Create comparison table
        comparison_df = pd.DataFrame(results).T
        metrics_to_show = ['accuracy', 'auc_roc', 'sensitivity', 'specificity', 'f1_score', 'ppv', 'npv', 'mcc']
        comparison_df = comparison_df[metrics_to_show].round(4)

        print("\n" + "=" * 80)
        print("COMPREHENSIVE COMPARISON SUMMARY")
        print("=" * 80)
        print(comparison_df)

        # Generate research paper visualizations
        print("\n📊 Creating research paper visualizations...")
        self.create_research_visualizations(X_train, y_train, X_test, y_test)

        # Plot ROC curves
        self.plot_roc_curves(X_test, y_test)

        # Plot Precision-Recall curves
        self.plot_precision_recall_curves(X_test, y_test)

        # Find best model
        best_model_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
        print(f"\n🏆 BEST MODEL: {best_model_name.upper()}")

        # Save detailed results
        self.save_detailed_results(results)

        return results, best_model_name

    def save_detailed_results(self, results):
        """Save detailed results to CSV and text file"""
        # Save to CSV
        results_df = pd.DataFrame(results).T
        csv_filename = os.path.join(self.visualization_dir, "detailed_results.csv")
        results_df.to_csv(csv_filename)

        # Save to text file
        txt_filename = os.path.join(self.visualization_dir, "detailed_results.txt")
        with open(txt_filename, 'w') as f:
            f.write("CHEST X-RAY PNEUMONIA DETECTION - DETAILED RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total images: {len(self.features)}\n")

            # Fixed class distribution display
            unique, counts = np.unique(self.labels, return_counts=True)
            class_distribution = dict(zip(unique, counts))
            f.write(f"Class distribution: {class_distribution}\n\n")

            for model_name, metrics in results.items():
                f.write(f"{model_name.upper()} RESULTS:\n")
                f.write("-" * 40 + "\n")
                for metric, value in metrics.items():
                    if metric != 'classification_report':
                        f.write(f"{metric:.<20}: {value}\n")
                f.write("\n")

                # Classification report
                f.write("CLASSIFICATION REPORT:\n")
                report = metrics['classification_report']
                for class_name in self.class_names:
                    f.write(f"{class_name}:\n")
                    f.write(f"  Precision: {report[class_name]['precision']:.4f}\n")
                    f.write(f"  Recall:    {report[class_name]['recall']:.4f}\n")
                    f.write(f"  F1-Score:  {report[class_name]['f1-score']:.4f}\n")
                    f.write(f"  Support:   {report[class_name]['support']}\n")
                f.write("\n" + "=" * 60 + "\n\n")

        print(f"📊 Detailed results saved to:")
        print(f"   - {csv_filename}")
        print(f"   - {txt_filename}")

    def predict_single_image(self, image_path, model_name='svm'):
        """Predict a single image using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img_flattened = img.flatten().reshape(1, -1)

        # Preprocess using the same scaler and PCA
        img_scaled = self.scaler.transform(img_flattened)

        if self.pca:
            img_processed = self.pca.transform(img_scaled)
        else:
            img_processed = img_scaled

        # Make prediction
        model = self.models[model_name]
        prediction = model.predict(img_processed)[0]
        prediction_proba = model.predict_proba(img_processed)[0] if hasattr(model, 'predict_proba') else [0, 0]

        result = self.class_names[prediction]
        confidence = prediction_proba[prediction] if hasattr(model, 'predict_proba') else 1.0

        print(f"\n{'=' * 50}")
        print("PREDICTION RESULT")
        print(f"{'=' * 50}")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Model: {model_name.upper()}")
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.4f}")
        if hasattr(model, 'predict_proba'):
            print(f"Probabilities: NORMAL={prediction_proba[0]:.4f}, PNEUMONIA={prediction_proba[1]:.4f}")
        print(f"{'=' * 50}")

        return result, confidence

    def save_models(self, filename='chest_xray_models.pkl'):
        """Save all models and preprocessors"""
        save_data = {
            'models': self.models,
            'scaler': self.scaler,
            'pca': self.pca,
            'img_height': self.img_height,
            'img_width': self.img_width,
            'class_names': self.class_names
        }
        joblib.dump(save_data, filename)
        print(f"✅ Models saved to {filename}")

    def load_models(self, filename='chest_xray_models.pkl'):
        """Load models and preprocessors"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")

        save_data = joblib.load(filename)
        self.models = save_data['models']
        self.scaler = save_data['scaler']
        self.pca = save_data['pca']
        self.img_height = save_data['img_height']
        self.img_width = save_data['img_width']
        self.class_names = save_data['class_names']
        print(f"✅ Models loaded from {filename}")
        print(f"Available models: {list(self.models.keys())}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Chest X-Ray Pneumonia Detection with ML')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'compare'],
                        help='Mode: train, predict, or compare')
    parser.add_argument('--data_dir', type=str, default='dataset/chest_xray',
                        help='Path to dataset directory')
    parser.add_argument('--image_path', type=str,
                        help='Path to image for prediction')
    parser.add_argument('--model_file', type=str, default='chest_xray_models.pkl',
                        help='Path to saved models')
    parser.add_argument('--use_advanced_features', action='store_true',
                        help='Use advanced feature extraction')
    parser.add_argument('--use_pca', action='store_true',
                        help='Use PCA for dimensionality reduction')
    parser.add_argument('--pca_components', type=int, default=50,
                        help='Number of PCA components')

    args = parser.parse_args()

    # Initialize predictor
    predictor = ChestXRayMLPredictor(img_height=100, img_width=100)

    if args.mode == 'train':
        try:
            # Debug dataset structure first
            if not predictor.debug_dataset_structure(args.data_dir):
                print("❌ Dataset structure issue detected. Please check your dataset path.")
                return

            # Load and preprocess data
            features, labels = predictor.load_and_preprocess_images(args.data_dir)

            if features is None or len(features) == 0:
                print("❌ No images were loaded. Please check your dataset.")
                return

            X_train, X_test, y_train, y_test = predictor.preprocess_data(
                use_advanced_features=args.use_advanced_features,
                use_pca=args.use_pca,
                n_components=args.pca_components
            )

            # Train models
            predictor.train_svm(X_train, y_train, cv_tuning=True)
            predictor.train_knn(X_train, y_train, cv_tuning=True)
            predictor.train_random_forest(X_train, y_train, cv_tuning=True)

            # Evaluate and compare models
            results, best_model = predictor.compare_models(X_train, y_train, X_test, y_test)

            # Save models
            predictor.save_models(args.model_file)

        except Exception as e:
            print(f"❌ Error during training: {e}")
            import traceback
            traceback.print_exc()

    elif args.mode == 'predict':
        if not args.image_path:
            print("❌ Error: --image_path is required for prediction mode")
            return

        try:
            predictor.load_models(args.model_file)
            result, confidence = predictor.predict_single_image(args.image_path, model_name='svm')

        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()

    elif args.mode == 'compare':
        try:
            predictor.load_models(args.model_file)

            # Debug dataset structure first
            if not predictor.debug_dataset_structure(args.data_dir):
                print("❌ Dataset structure issue detected. Please check your dataset path.")
                return

            features, labels = predictor.load_and_preprocess_images(args.data_dir)

            if features is None or len(features) == 0:
                print("❌ No images were loaded. Please check your dataset.")
                return

            X_train, X_test, y_train, y_test = predictor.preprocess_data(
                use_advanced_features=args.use_advanced_features,
                use_pca=args.use_pca,
                n_components=args.pca_components
            )
            results, best_model = predictor.compare_models(X_train, y_train, X_test, y_test)

        except Exception as e:
            print(f"❌ Error during comparison: {e}")
            import traceback
            traceback.print_exc()


def quick_start():
    """Quick start function"""
    print("🚀 Quick Start: Chest X-Ray Pneumonia Detection with ML")
    print("=" * 60)

    predictor = ChestXRayMLPredictor(img_height=100, img_width=100)

    # Check if models already exist
    if os.path.exists('chest_xray_models.pkl'):
        print("✅ Pre-trained models found! Loading models...")
        predictor.load_models('chest_xray_models.pkl')

        # Test predictions
        sample_images = [
            "dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg",
            "dataset/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
        ]

        for img_path in sample_images:
            if os.path.exists(img_path):
                print(f"\n🔍 Testing with: {os.path.basename(img_path)}")
                for model_name in predictor.models.keys():
                    try:
                        predictor.predict_single_image(img_path, model_name)
                    except Exception as e:
                        print(f"Error with {model_name}: {e}")
            else:
                print(f"⚠️  Sample image not found: {img_path}")

    else:
        print("🔄 No pre-trained models found. Starting training...")
        try:
            # Debug dataset structure first
            if not predictor.debug_dataset_structure():
                print("❌ Dataset structure issue detected. Please check your dataset path.")
                return

            # Full training pipeline
            print("\n📥 Loading and preprocessing images...")
            features, labels = predictor.load_and_preprocess_images()

            if features is None or len(features) == 0:
                print("❌ No images were loaded. Please check:")
                print("   - Dataset path is correct")
                print("   - Images are in JPEG/PNG format")
                print("   - Directory structure is correct")
                return

            print(f"✅ Successfully loaded {len(features)} images")

            X_train, X_test, y_train, y_test = predictor.preprocess_data(
                use_advanced_features=False,
                use_pca=True,
                n_components=50
            )

            print("\n🤖 Training models...")
            predictor.train_svm(X_train, y_train, cv_tuning=True)
            predictor.train_knn(X_train, y_train, cv_tuning=True)
            predictor.train_random_forest(X_train, y_train, cv_tuning=True)

            print("\n📊 Evaluating models...")
            results, best_model = predictor.compare_models(X_train, y_train, X_test, y_test)
            predictor.save_models()

            print("\n🎉 Training completed! You can now use the models for predictions.")

        except Exception as e:
            print(f"❌ Error during training: {e}")

            import traceback
            traceback.print_exc()
            print("\n💡 Make sure your dataset is in the correct structure:")
            print("   dataset/chest_xray/train/NORMAL/...")
            print("   dataset/chest_xray/train/PNEUMONIA/...")
            print("   dataset/chest_xray/val/...")
            print("   dataset/chest_xray/test/...")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        quick_start()