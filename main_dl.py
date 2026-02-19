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
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_curve, auc,
                             precision_recall_curve, matthews_corrcoef, cohen_kappa_score,
                             log_loss, roc_auc_score, average_precision_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.utils import resample
import pandas as pd
from tqdm import tqdm
import argparse
from datetime import datetime
from tabulate import tabulate
import warnings
import time

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')


class ChestXRayMLPredictor:
    def __init__(self, img_height=224, img_width=224):  # EfficientNet requires 224x224
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
        self.metrics_history = {}
        self.training_times = {}
        self.inference_times = {}

        # Store raw image data for EfficientNet
        self.raw_features = None
        self.raw_labels = None
        self.raw_X_train = None
        self.raw_X_test = None
        self.raw_y_train = None
        self.raw_y_test = None

        # EfficientNet specific attributes
        self.efficientnet_model = None
        self.efficientnet_history = None

        # Create visualization directory
        os.makedirs(self.visualization_dir, exist_ok=True)

    def debug_dataset_structure(self, data_dir='dataset/chest_xray'):
        """Debug function to check dataset structure"""
        print("\n🔍 Debugging dataset structure...")

        if not os.path.exists(data_dir):
            print(f"❌ Main dataset directory not found: {data_dir}")
            return False

        splits = ['train', 'test', 'val']
        table_data = []

        for split in splits:
            split_path = os.path.join(data_dir, split)

            if os.path.exists(split_path):
                split_data = [split]
                total_images = 0

                for class_name in self.class_names:
                    class_path = os.path.join(split_path, class_name)
                    if os.path.exists(class_path):
                        image_files = [f for f in os.listdir(class_path)
                                       if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
                        num_images = len(image_files)
                        split_data.append(f"{num_images}")
                        total_images += num_images
                    else:
                        split_data.append("0")

                split_data.append(str(total_images))
                table_data.append(split_data)
            else:
                table_data.append([split, "Not Found", "Not Found", "0"])

        headers = ["Split", "NORMAL", "PNEUMONIA", "Total"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        return True

    def load_and_preprocess_images(self, data_dir='dataset/chest_xray'):
        """Load and preprocess images from directory"""
        print("Loading and preprocessing images...")

        features = []
        raw_features = []  # Store raw images for EfficientNet
        labels = []

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
                image_files = [f for f in os.listdir(class_path)
                               if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

                for image_file in tqdm(image_files, desc=f"{split}/{class_name}"):
                    img_path = os.path.join(class_path, image_file)

                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            continue

                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # Resize to target size (224x224 for EfficientNet)
                        img = cv2.resize(img, (self.img_width, self.img_height))

                        # Store raw image for EfficientNet
                        raw_features.append(img)

                        # Store flattened image for ML models
                        img_flattened = img.flatten()
                        features.append(img_flattened)
                        labels.append(class_name)

                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue

        if len(features) == 0:
            print("❌ No images were loaded.")
            return None, None

        self.features = np.array(features)
        self.raw_features = np.array(raw_features)
        self.labels = np.array(labels)

        print(f"\n✅ Loaded {len(features)} images")
        print(f"✅ Raw features shape: {self.raw_features.shape}")

        # Print class distribution table
        unique, counts = np.unique(self.labels, return_counts=True)
        table_data = []
        for class_name, count in zip(unique, counts):
            percentage = (count / len(self.labels)) * 100
            table_data.append([class_name, count, f"{percentage:.2f}%"])

        print(tabulate(table_data, headers=["Class", "Count", "Percentage"], tablefmt="grid"))

        return self.features, self.labels

    def extract_advanced_features(self, images):
        """Extract advanced features from images"""
        print("Extracting advanced features...")

        advanced_features = []

        for img_flat in tqdm(images, desc="Feature extraction"):
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

            # 3. Texture features
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
        """Preprocess the data for ML models and prepare raw data for EfficientNet"""
        print("Preprocessing data...")

        le = LabelEncoder()
        y_encoded = le.fit_transform(self.labels)

        if use_advanced_features:
            X_processed = self.extract_advanced_features(self.features)
        else:
            X_processed = self.features

        # Split data for ML models (flattened features)
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Split raw images for EfficientNet using the same indices
        raw_train_indices, raw_test_indices = train_test_split(
            np.arange(len(self.raw_features)), test_size=0.2, random_state=42, stratify=y_encoded
        )

        self.raw_X_train = self.raw_features[raw_train_indices]
        self.raw_X_test = self.raw_features[raw_test_indices]
        self.raw_y_train = y_encoded[raw_train_indices]
        self.raw_y_test = y_encoded[raw_test_indices]

        print(f"Raw training set shape for EfficientNet: {self.raw_X_train.shape}")
        print(f"Raw test set shape for EfficientNet: {self.raw_X_test.shape}")

        # Scale the features for ML models
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

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

        print(f"ML Training set shape: {X_train_final.shape}")
        print(f"ML Test set shape: {X_test_final.shape}")

        return X_train_final, X_test_final, y_train, y_test

    def train_svm(self, X_train, y_train, cv_tuning=True):
        """Train SVM with RBF kernel"""
        start_time = time.time()
        print("\n" + "=" * 50)
        print("TRAINING SVM WITH RBF KERNEL")
        print("=" * 50)

        if cv_tuning:
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.01],
                'kernel': ['rbf']
            }

            svm = SVC(random_state=42, probability=True)
            grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
            grid_search.fit(X_train, y_train)

            best_svm = grid_search.best_estimator_
            print(f"Best SVM parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        else:
            best_svm = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42, probability=True)
            best_svm.fit(X_train, y_train)

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        self.models['svm'] = best_svm
        self.training_times['svm'] = training_time
        return best_svm

    def train_knn(self, X_train, y_train, cv_tuning=True):
        """Train KNN classifier"""
        start_time = time.time()
        print("\n" + "=" * 50)
        print("TRAINING K-NEAREST NEIGHBORS")
        print("=" * 50)

        if cv_tuning:
            param_grid = {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }

            knn = KNeighborsClassifier()
            grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
            grid_search.fit(X_train, y_train)

            best_knn = grid_search.best_estimator_
            print(f"Best KNN parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        else:
            best_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
            best_knn.fit(X_train, y_train)

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        self.models['knn'] = best_knn
        self.training_times['knn'] = training_time
        return best_knn

    def train_random_forest(self, X_train, y_train, cv_tuning=True):
        """Train Random Forest classifier"""
        start_time = time.time()
        print("\n" + "=" * 50)
        print("TRAINING RANDOM FOREST")
        print("=" * 50)

        if cv_tuning:
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5]
            }

            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
            grid_search.fit(X_train, y_train)

            best_rf = grid_search.best_estimator_
            print(f"Best Random Forest parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        else:
            best_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            best_rf.fit(X_train, y_train)

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        self.models['random_forest'] = best_rf
        self.training_times['random_forest'] = training_time
        return best_rf

    def train_efficientnet_single_phase(self, epochs=5, batch_size=32, model_variant='B0'):
        """Train EfficientNetV2 model with single-phase fine-tuning (optimized for speed)"""
        if self.raw_X_train is None or self.raw_X_test is None:
            print("❌ Raw image data not available. Run preprocess_data first.")
            return None, None

        start_time = time.time()
        print("\n" + "=" * 50)
        print(f"TRAINING EFFICIENTNETV2-{model_variant} (SINGLE PHASE)")
        print("=" * 50)

        # Normalize pixel values
        X_train_eff = self.raw_X_train.astype('float32') / 255.0
        X_test_eff = self.raw_X_test.astype('float32') / 255.0

        # Convert labels to categorical
        y_train_cat = keras.utils.to_categorical(self.raw_y_train, 2)
        y_test_cat = keras.utils.to_categorical(self.raw_y_test, 2)

        print(f"Training data shape: {X_train_eff.shape}")
        print(f"Test data shape: {X_test_eff.shape}")

        # Load pre-trained EfficientNetV2
        base_model = EfficientNetV2B0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )

        # Unfreeze the last few layers for fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-20]:  # Freeze earlier layers
            layer.trainable = False

        # Build model
        inputs = keras.Input(shape=(self.img_height, self.img_width, 3))
        x = base_model(inputs, training=True)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(2, activation='softmax')(x)

        model = keras.Model(inputs, outputs)

        # Compile with moderate learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Print model summary (reduced)
        print(f"\nEfficientNetV2-{model_variant} Model")
        print(f"Total parameters: {model.count_params():,}")

        # Callbacks for efficient training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6, verbose=1)
        ]

        # Single phase training
        print(f"\nFine-tuning for {epochs} epochs...")
        history = model.fit(
            X_train_eff, y_train_cat,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test_eff, y_test_cat),
            callbacks=callbacks,
            verbose=1
        )

        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.2f} seconds")

        # Store model and history
        self.efficientnet_model = model
        self.efficientnet_history = history
        self.models['efficientnet'] = model
        self.training_times['efficientnet'] = training_time

        return model, history

    def evaluate_dl_model(self, model, model_name, X_test, y_test):
        """Evaluate EfficientNet model"""
        print(f"\nEvaluating {model_name}...")

        # Get predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Calculate additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1]) if len(np.unique(y_test)) > 1 else 0.5
        auc_pr = average_precision_score(y_test, y_pred_proba[:, 1])
        logloss = log_loss(y_test, y_pred_proba)
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Calibration error
        calibration_error = np.mean(np.abs(y_pred_proba[:, 1] - (y_test == 1).astype(float)))

        # ROC and PR curve data
        fpr_curve, tpr_curve, _ = roc_curve(y_test, y_pred_proba[:, 1])
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])

        # Store metrics
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
            'train_accuracy': accuracy,
            'overfitting_gap': 0.0,
            'cv_mean': 0.0,
            'cv_std': 0.0,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'fpr_curve': fpr_curve,
            'tpr_curve': tpr_curve,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'confusion_matrix': cm,
            'training_time': self.training_times.get(model_name, 0),
            'inference_time': self.inference_times.get(model_name, 0)
        }

        self.metrics_history[model_name] = metrics

        # Print results
        print(f"\n{model_name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC-ROC: {auc_roc:.4f}")

        return metrics

    def measure_dl_inference_time(self, model, X_test_sample, num_runs=30):
        """Measure inference time for EfficientNet"""
        if model is None:
            return 0

        # Warm up
        _ = model.predict(X_test_sample[:1], verbose=0)

        # Measure inference time
        start_time = time.perf_counter()
        for i in range(min(num_runs, len(X_test_sample))):
            _ = model.predict(X_test_sample[i:i + 1], verbose=0)
        total_time = time.perf_counter() - start_time

        avg_inference_time_ms = (total_time / min(num_runs, len(X_test_sample))) * 1000
        return avg_inference_time_ms

    def measure_inference_time(self, model, X_test, num_runs=100):
        """Measure inference time for ML models"""
        if not hasattr(model, 'predict'):
            return 0

        # Warm up
        model.predict(X_test[:1])

        # Measure inference time
        start_time = time.perf_counter()
        for _ in range(num_runs):
            model.predict(X_test[:1])
        total_time = time.perf_counter() - start_time

        avg_inference_time_ms = (total_time / num_runs) * 1000
        return avg_inference_time_ms

    def measure_all_inference_times(self, X_test):
        """Measure inference times for all models"""
        print("\n📊 Measuring inference times...")

        # Prepare test samples for EfficientNet
        X_test_dl = self.raw_X_test.astype('float32') / 255.0

        for model_name, model in self.models.items():
            if model_name == 'efficientnet':
                inference_time = self.measure_dl_inference_time(model, X_test_dl[:30])
            else:
                inference_time = self.measure_inference_time(model, X_test)

            self.inference_times[model_name] = inference_time
            print(f"  {model_name.upper()}: {inference_time:.2f} ms/image")

    def print_timing_summary(self):
        """Print timing summary table"""
        print("\n" + "=" * 60)
        print("MODEL TRAINING AND INFERENCE TIMING SUMMARY")
        print("=" * 60)

        table_data = []
        for model_name in self.models.keys():
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

    def print_confusion_matrix(self, cm, model_name):
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

    def print_classification_report(self, y_true, y_pred, model_name):
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

        print(tabulate(table_data, headers=["Class", "Precision", "Recall", "F1-Score", "Support"], tablefmt="grid"))

        accuracy = report['accuracy']
        print(f"\nOverall Accuracy: {accuracy:.4f}")

    def print_learning_curves(self, model, X, y, model_name):
        """Print learning curve analysis"""
        print(f"\n{model_name.upper()} - Learning Curve Analysis:")
        print("-" * 60)

        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=3,
            train_sizes=np.linspace(0.3, 1.0, 3),
            scoring='accuracy', n_jobs=-1, random_state=42
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

    def print_cross_validation_scores(self, model, X, y, model_name):
        """Print cross-validation scores"""
        print(f"\n{model_name.upper()} - Cross-Validation Analysis:")
        print("-" * 60)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

        table_data = []
        for fold, score in enumerate(scores, 1):
            table_data.append([f"Fold {fold}", f"{score:.4f}"])

        table_data.append(["", ""])
        table_data.append(["Mean", f"{np.mean(scores):.4f}"])
        table_data.append(["Std Dev", f"{np.std(scores):.4f}"])

        print(tabulate(table_data, headers=["Fold", "Accuracy"], tablefmt="grid"))

    def calculate_all_metrics(self, model, X_train, X_test, y_train, y_test, model_name):
        """Calculate all comprehensive metrics for ML models"""
        if model not in self.models.values():
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        if y_proba is not None:
            auc_roc = roc_auc_score(y_test, y_proba)
            auc_pr = average_precision_score(y_test, y_proba)
            logloss = log_loss(y_test, y_proba)

            fpr_curve, tpr_curve, _ = roc_curve(y_test, y_proba)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        else:
            auc_roc = 0
            auc_pr = 0
            logloss = 0
            fpr_curve, tpr_curve = None, None
            precision_curve, recall_curve = None, None

        if y_proba is not None:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_proba, n_bins=5, strategy='uniform'
            )
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        else:
            calibration_error = 0

        train_accuracy = accuracy_score(y_train, y_pred_train)
        overfitting_gap = train_accuracy - accuracy

        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

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
            'training_time': self.training_times.get(model_name, 0),
            'inference_time': self.inference_times.get(model_name, 0)
        }

        self.metrics_history[model_name] = metrics
        return metrics

    def print_all_metrics_terminal(self, metrics):
        """Print all metrics in terminal"""
        model_name = metrics['model_name']

        print(f"\n{'=' * 80}")
        print(f"{model_name.upper()} - METRICS ANALYSIS")
        print(f"{'=' * 80}")

        print("\n⏱️  TIMING:")
        timing_table = [
            ["Training Time", f"{metrics.get('training_time', 0):.2f} s"],
            ["Inference Time", f"{metrics.get('inference_time', 0):.2f} ms"]
        ]
        print(tabulate(timing_table, headers=["Metric", "Value"], tablefmt="grid"))

        print("\n📊 PERFORMANCE:")
        basic_metrics = [
            ["Accuracy", f"{metrics.get('accuracy', 0):.4f}"],
            ["Precision", f"{metrics.get('precision', 0):.4f}"],
            ["Recall", f"{metrics.get('recall', 0):.4f}"],
            ["F1-Score", f"{metrics.get('f1_score', 0):.4f}"],
            ["Specificity", f"{metrics.get('specificity', 0):.4f}"],
            ["AUC-ROC", f"{metrics.get('auc_roc', 0):.4f}"]
        ]
        print(tabulate(basic_metrics, headers=["Metric", "Value"], tablefmt="grid"))

        print("\n📈 ADVANCED:")
        advanced = [
            ["MCC", f"{metrics.get('mcc', 0):.4f}"],
            ["Kappa", f"{metrics.get('kappa', 0):.4f}"],
            ["PPV", f"{metrics.get('ppv', 0):.4f}"],
            ["NPV", f"{metrics.get('npv', 0):.4f}"],
            ["Calibration Error", f"{metrics.get('calibration_error', 0):.4f}"]
        ]
        print(tabulate(advanced, headers=["Metric", "Value"], tablefmt="grid"))

        if 'confusion_matrix' in metrics:
            self.print_confusion_matrix(metrics['confusion_matrix'], model_name)

        print(f"\n{'=' * 80}")

    def create_consolidated_metrics_visualization(self):
        """Create consolidated visualization"""
        if not self.metrics_history:
            return

        print("\n📊 Creating visualization...")

        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(16, 10))

        model_colors = {
            'svm': '#1f77b4',
            'knn': '#ff7f0e',
            'random_forest': '#2ca02c',
            'efficientnet': '#d62728'
        }

        model_names = list(self.metrics_history.keys())

        # 1. Bar chart for key metrics
        ax1 = plt.subplot(2, 3, 1)
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Acc', 'Prec', 'Rec', 'F1']
        x = np.arange(len(metric_labels))
        width = 0.2

        for idx, (model_name, metrics) in enumerate(self.metrics_history.items()):
            if model_name in model_colors:
                values = [metrics.get(m, 0) for m in key_metrics]
                ax1.bar(x + idx * width - width, values, width,
                        label=model_name.upper(), color=model_colors[model_name], alpha=0.8)

        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Key Metrics', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_labels)
        ax1.legend()
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # 2. ROC curves
        ax2 = plt.subplot(2, 3, 2)
        for model_name, metrics in self.metrics_history.items():
            if model_name in model_colors and metrics.get('fpr_curve') is not None:
                ax2.plot(metrics['fpr_curve'], metrics['tpr_curve'],
                         label=f"{model_name.upper()} (AUC={metrics.get('auc_roc', 0):.3f})",
                         color=model_colors[model_name], linewidth=2)
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_xlabel('FPR')
        ax2.set_ylabel('TPR')
        ax2.set_title('ROC Curves', fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

        # 3. Confusion matrix heatmap
        ax3 = plt.subplot(2, 3, 3)
        combined_cm = sum([metrics['confusion_matrix'] for metrics in self.metrics_history.values()])
        sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=ax3, cbar=True)
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.set_title('Combined Confusion Matrix', fontweight='bold')

        # 4. Training vs Test
        ax4 = plt.subplot(2, 3, 4)
        train_accs = [metrics.get('train_accuracy', 0) for metrics in self.metrics_history.values()]
        test_accs = [metrics.get('accuracy', 0) for metrics in self.metrics_history.values()]
        x = np.arange(len(model_names))
        ax4.bar(x - 0.2, train_accs, 0.4, label='Train', alpha=0.8, color='blue')
        ax4.bar(x + 0.2, test_accs, 0.4, label='Test', alpha=0.8, color='red')
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Train vs Test Accuracy', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([m.upper() for m in model_names])
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Inference Time
        ax5 = plt.subplot(2, 3, 5)
        times = [self.inference_times.get(m, 0) for m in model_names]
        colors = [model_colors.get(m, 'gray') for m in model_names]
        ax5.bar(model_names, times, color=colors, alpha=0.8)
        ax5.set_xlabel('Models')
        ax5.set_ylabel('Time (ms)')
        ax5.set_title('Inference Time', fontweight='bold')
        ax5.set_xticklabels([m.upper() for m in model_names])
        ax5.grid(True, alpha=0.3)

        # 6. Training Time
        ax6 = plt.subplot(2, 3, 6)
        times = [self.training_times.get(m, 0) for m in model_names]
        ax6.bar(model_names, times, color=colors, alpha=0.8)
        ax6.set_xlabel('Models')
        ax6.set_ylabel('Time (s)')
        ax6.set_title('Training Time', fontweight='bold')
        ax6.set_xticklabels([m.upper() for m in model_names])
        ax6.grid(True, alpha=0.3)

        plt.suptitle('MODEL COMPARISON - CHEST X-RAY PNEUMONIA DETECTION', fontsize=14, fontweight='bold')
        plt.tight_layout()

        filename = os.path.join(self.visualization_dir, "model_comparison.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Visualization saved: {filename}")
        self.print_summary_table()

    def print_summary_table(self):
        """Print summary table"""
        print("\n" + "=" * 100)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 100)

        table_data = []
        headers = ["Metric"] + [m.upper() for m in self.metrics_history.keys()]

        metrics_to_show = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'mcc']

        for metric in metrics_to_show:
            row = [metric.upper()]
            for model_name in self.metrics_history.keys():
                value = self.metrics_history[model_name].get(metric, 0)
                row.append(f"{value:.4f}")
            table_data.append(row)

        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Best model
        print("\n" + "=" * 100)
        print("BEST MODEL BY METRIC")
        print("=" * 100)

        best_model_data = []
        for metric in metrics_to_show:
            best = max(self.metrics_history.items(), key=lambda x: x[1].get(metric, 0))
            best_model_data.append([metric.upper(), best[0].upper(), f"{best[1].get(metric, 0):.4f}"])

        print(tabulate(best_model_data, headers=["Metric", "Best Model", "Value"], tablefmt="grid"))

    def evaluate_all_models(self, X_train, X_test, y_train, y_test):
        """Evaluate all models"""
        print("\n" + "=" * 100)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 100)

        self.measure_all_inference_times(X_test)

        results = {}
        X_test_dl = self.raw_X_test.astype('float32') / 255.0

        for model_name, model in self.models.items():
            print(f"\n{'=' * 80}")
            print(f"EVALUATING {model_name.upper()}")
            print(f"{'=' * 80}")

            if model_name == 'efficientnet':
                metrics = self.evaluate_dl_model(model, model_name, X_test_dl, self.raw_y_test)
            else:
                metrics = self.calculate_all_metrics(model, X_train, X_test, y_train, y_test, model_name)

            self.print_all_metrics_terminal(metrics)

            if model_name != 'efficientnet':
                y_pred = model.predict(X_test)
                self.print_classification_report(y_test, y_pred, model_name)

            results[model_name] = metrics

        self.print_timing_summary()
        self.create_consolidated_metrics_visualization()

        return results

    def predict_single_image(self, image_path, model_name='svm'):
        """Predict a single image"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height))

        model = self.models[model_name]

        if model_name == 'efficientnet':
            img_processed = img.reshape(1, self.img_height, self.img_width, 3).astype('float32') / 255.0

            start_time = time.perf_counter()
            prediction_proba = model.predict(img_processed, verbose=0)[0]
            inference_time = (time.perf_counter() - start_time) * 1000

            prediction = np.argmax(prediction_proba)
            confidence = prediction_proba[prediction]

        else:
            img_flattened = img.flatten().reshape(1, -1)
            img_scaled = self.scaler.transform(img_flattened)

            if self.pca:
                img_processed = self.pca.transform(img_scaled)
            else:
                img_processed = img_scaled

            start_time = time.perf_counter()
            prediction = model.predict(img_processed)[0]
            inference_time = (time.perf_counter() - start_time) * 1000

            prediction_proba = model.predict_proba(img_processed)[0] if hasattr(model, 'predict_proba') else [0, 0]
            confidence = prediction_proba[prediction] if hasattr(model, 'predict_proba') else 1.0

        result = self.class_names[prediction]

        print(f"\n{'=' * 60}")
        print("PREDICTION RESULT")
        print(f"{'=' * 60}")

        pred_table = [
            ["Image", os.path.basename(image_path)],
            ["Model", model_name.upper()],
            ["Prediction", result],
            ["Confidence", f"{confidence:.4f}"],
            ["Inference Time", f"{inference_time:.2f} ms"]
        ]

        if hasattr(model, 'predict_proba') or model_name == 'efficientnet':
            pred_table.append(
                ["Probabilities", f"NORMAL={prediction_proba[0]:.4f}, PNEUMONIA={prediction_proba[1]:.4f}"])

        print(tabulate(pred_table, tablefmt="grid"))
        print(f"{'=' * 60}")

        return result, confidence, inference_time

    def save_models(self, filename='chest_xray_models.pkl'):
        """Save all models"""
        ml_models = {k: v for k, v in self.models.items() if k != 'efficientnet'}

        save_data = {
            'models': ml_models,
            'scaler': self.scaler,
            'pca': self.pca,
            'img_height': self.img_height,
            'img_width': self.img_width,
            'class_names': self.class_names,
            'metrics_history': self.metrics_history,
            'training_times': self.training_times,
            'inference_times': self.inference_times
        }
        joblib.dump(save_data, filename)

        if self.efficientnet_model is not None:
            self.efficientnet_model.save('efficientnet_model.keras')
            print(f"✅ EfficientNet model saved to efficientnet_model.keras")

        print(f"✅ Models saved to {filename}")

    def load_models(self, filename='chest_xray_models.pkl'):
        """Load models"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")

        save_data = joblib.load(filename)
        self.models = save_data['models']
        self.scaler = save_data['scaler']
        self.pca = save_data['pca']
        self.img_height = save_data['img_height']
        self.img_width = save_data['img_width']
        self.class_names = save_data['class_names']
        self.metrics_history = save_data.get('metrics_history', {})
        self.training_times = save_data.get('training_times', {})
        self.inference_times = save_data.get('inference_times', {})

        efficientnet_path = 'efficientnet_model.keras'
        if os.path.exists(efficientnet_path):
            try:
                self.efficientnet_model = keras.models.load_model(efficientnet_path)
                self.models['efficientnet'] = self.efficientnet_model
                print(f"✅ EfficientNet model loaded")
            except Exception as e:
                print(f"⚠️ Could not load EfficientNet model: {e}")

        print(f"✅ Models loaded from {filename}")
        print("\nAvailable Models:", list(self.models.keys()))


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Chest X-Ray Pneumonia Detection')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict'],
                        help='Mode: train or predict')
    parser.add_argument('--data_dir', type=str, default='dataset/chest_xray',
                        help='Dataset directory')
    parser.add_argument('--image_path', type=str, help='Image path for prediction')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs for EfficientNet')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    args = parser.parse_args()

    predictor = ChestXRayMLPredictor()

    if args.mode == 'train':
        try:
            if not predictor.debug_dataset_structure(args.data_dir):
                return

            features, labels = predictor.load_and_preprocess_images(args.data_dir)
            if features is None:
                return

            X_train, X_test, y_train, y_test = predictor.preprocess_data(use_pca=True)

            # Train ML models
            print("\n" + "=" * 50)
            print("TRAINING ML MODELS")
            print("=" * 50)
            predictor.train_svm(X_train, y_train)
            predictor.train_knn(X_train, y_train)
            predictor.train_random_forest(X_train, y_train)

            # Train EfficientNet (single phase, few epochs)
            print("\n" + "=" * 50)
            print("TRAINING EFFICIENTNET (SINGLE PHASE)")
            print("=" * 50)
            predictor.train_efficientnet_single_phase(epochs=args.epochs, batch_size=args.batch_size)

            # Evaluate all
            predictor.evaluate_all_models(X_train, X_test, y_train, y_test)
            predictor.save_models()

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    elif args.mode == 'predict':
        if not args.image_path:
            print("❌ Error: --image_path required")
            return

        try:
            predictor.load_models()
            for model_name in predictor.models.keys():
                predictor.predict_single_image(args.image_path, model_name)
        except Exception as e:
            print(f"❌ Error: {e}")


def quick_start():
    """Quick start"""
    print("=" * 60)
    print("CHEST X-RAY PNEUMONIA DETECTION")
    print("=" * 60)

    predictor = ChestXRayMLPredictor()

    if os.path.exists('chest_xray_models.pkl') and os.path.exists('efficientnet_model.keras'):
        print("Loading pre-trained models...")
        predictor.load_models()

        # Test with sample images
        sample_images = [
            "dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg",
            "dataset/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
        ]

        for img_path in sample_images:
            if os.path.exists(img_path):
                print(f"\n🔍 Testing: {os.path.basename(img_path)}")
                for model_name in predictor.models.keys():
                    predictor.predict_single_image(img_path, model_name)
    else:
        print("No models found. Starting fast training...")
        try:
            if not predictor.debug_dataset_structure():
                return

            features, labels = predictor.load_and_preprocess_images()
            X_train, X_test, y_train, y_test = predictor.preprocess_data(use_pca=True)

            # Train ML models
            predictor.train_svm(X_train, y_train)
            predictor.train_knn(X_train, y_train)
            predictor.train_random_forest(X_train, y_train)

            # Train EfficientNet with just 3 epochs for speed
            predictor.train_efficientnet_single_phase(epochs=3, batch_size=32)

            # Evaluate
            predictor.evaluate_all_models(X_train, X_test, y_train, y_test)
            predictor.save_models()

            print("\n✅ Training completed!")

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        quick_start()