
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, roc_curve, auc, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


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
        self.le = LabelEncoder()
        self.training_history = {}
        self.visualization_dir = "visualizations"

        # Create visualization directory
        os.makedirs(self.visualization_dir, exist_ok=True)

    def debug_dataset_structure(self, data_dir='dataset/chest_xray'):
        """Debug function to check dataset structure"""
        st.info("Debugging dataset structure...")

        if not os.path.exists(data_dir):
            st.error(f"Main dataset directory not found: {data_dir}")
            return False

        splits = ['train', 'test', 'val']

        for split in splits:
            split_path = os.path.join(data_dir, split)
            st.write(f"Checking {split_path}:")

            if os.path.exists(split_path):
                st.success("Found")
                for class_name in self.class_names:
                    class_path = os.path.join(split_path, class_name)
                    if os.path.exists(class_path):
                        image_files = [f for f in os.listdir(class_path)
                                       if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
                        st.write(f"    {class_name}: {len(image_files)} images")
                    else:
                        st.error(f"    {class_name}: directory not found")
            else:
                st.error("Not found")

        return True

    def load_and_preprocess_images(self, data_dir='dataset/chest_xray'):
        """Load and preprocess images from directory"""
        st.info("Loading and preprocessing images...")

        features = []
        labels = []
        failed_images = []

        # Define the splits to process
        splits = ['train', 'test', 'val']

        progress_bar = st.progress(0)
        status_text = st.empty()

        total_splits = len(splits)
        current_split = 0

        for split in splits:
            split_path = os.path.join(data_dir, split)
            if not os.path.exists(split_path):
                st.warning(f"Warning: {split_path} not found, skipping...")
                continue

            for class_name in self.class_names:
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    st.warning(f"Warning: {class_path} not found, skipping...")
                    continue

                status_text.text(f"Processing {split}/{class_name}...")

                # Get all image files
                image_files = [f for f in os.listdir(class_path)
                               if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

                for i, image_file in enumerate(image_files):
                    img_path = os.path.join(class_path, image_file)

                    try:
                        # Load and preprocess image
                        img = cv2.imread(img_path)
                        if img is None:
                            failed_images.append(img_path)
                            continue

                        # Convert to RGB and resize
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.img_width, self.img_height))

                        # Extract features (flatten image)
                        img_flattened = img.flatten()

                        features.append(img_flattened)
                        labels.append(class_name)

                    except Exception as e:
                        st.error(f"Error processing {img_path}: {e}")
                        failed_images.append(img_path)
                        continue

                # Update progress
                current_split += 1
                progress_bar.progress(current_split / (total_splits * len(self.class_names)))

        if failed_images:
            st.warning(f"Failed to process {len(failed_images)} images")

        if len(features) == 0:
            st.error("No images were loaded. Please check your dataset structure.")
            return None, None

        self.features = np.array(features)
        self.labels = np.array(labels)

        status_text.text(f"Successfully loaded {len(features)} images")

        # Display class distribution
        st.success("Class distribution:")
        unique, counts = np.unique(self.labels, return_counts=True)
        class_dist = {}
        for class_name, count in zip(unique, counts):
            percentage = (count / len(self.labels)) * 100
            class_dist[class_name] = count
            st.write(f"  {class_name}: {count} ({percentage:.2f}%)")

        # Plot class distribution
        fig = px.bar(x=list(class_dist.keys()), y=list(class_dist.values()),
                     title="Class Distribution", labels={'x': 'Class', 'y': 'Count'},
                     color=list(class_dist.keys()))
        st.plotly_chart(fig)

        return self.features, self.labels

    def extract_advanced_features(self, images):
        """Extract advanced features from images"""
        st.info("Extracting advanced features...")

        advanced_features = []
        progress_bar = st.progress(0)

        for i, img_flat in enumerate(images):
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

            # Update progress
            progress_bar.progress((i + 1) / len(images))

        return np.array(advanced_features)

    def preprocess_data(self, use_advanced_features=False, use_pca=False, n_components=50):
        """Preprocess the data for ML models"""
        st.info("Preprocessing data...")

        # Encode labels
        y_encoded = self.le.fit_transform(self.labels)

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
            st.info(f"Applying PCA with {n_components} components...")
            self.pca = PCA(n_components=n_components, random_state=42)
            X_train_final = self.pca.fit_transform(X_train_scaled)
            X_test_final = self.pca.transform(X_test_scaled)
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            st.success(f"Explained variance ratio: {explained_variance:.4f}")

            # Plot PCA explained variance
            fig = px.line(y=np.cumsum(self.pca.explained_variance_ratio_),
                          title="Cumulative Explained Variance by PCA Components",
                          labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance'})
            fig.add_hline(y=0.95, line_dash="dash", line_color="red", annotation_text="95% Variance")
            st.plotly_chart(fig)
        else:
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled

        st.success(f"Final feature dimensions:")
        st.success(f"  Training set: {X_train_final.shape}")
        st.success(f"  Test set: {X_train_final.shape}")

        return X_train_final, X_test_final, y_train, y_test

    def train_svm(self, X_train, y_train, cv_tuning=True):
        """Train SVM with RBF kernel"""
        st.info("Training SVM with RBF Kernel...")

        if cv_tuning:
            # Hyperparameter tuning with GridSearchCV
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf']
            }

            svm = SVC(random_state=42, probability=True)
            grid_search = GridSearchCV(
                svm, param_grid, cv=3, scoring='accuracy',
                n_jobs=-1, verbose=0, refit=True
            )

            with st.spinner("Tuning SVM hyperparameters..."):
                grid_search.fit(X_train, y_train)

            best_svm = grid_search.best_estimator_
            st.success(f"Best SVM parameters: {grid_search.best_params_}")
            st.success(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        else:
            # Use default parameters
            best_svm = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42, probability=True)
            with st.spinner("Training SVM with default parameters..."):
                best_svm.fit(X_train, y_train)
            st.success("SVM trained with default parameters")

        self.models['svm'] = best_svm
        self.training_history['svm'] = {
            'best_params': grid_search.best_params_ if cv_tuning else {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'},
            'cv_score': grid_search.best_score_ if cv_tuning else "Not calculated"
        }
        return best_svm

    def train_knn(self, X_train, y_train, cv_tuning=True):
        """Train KNN classifier"""
        st.info("Training K-Nearest Neighbors...")

        if cv_tuning:
            # Hyperparameter tuning with GridSearchCV
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }

            knn = KNeighborsClassifier()
            grid_search = GridSearchCV(
                knn, param_grid, cv=3, scoring='accuracy',
                n_jobs=-1, verbose=0, refit=True
            )

            with st.spinner("Tuning KNN hyperparameters..."):
                grid_search.fit(X_train, y_train)

            best_knn = grid_search.best_estimator_
            st.success(f"Best KNN parameters: {grid_search.best_params_}")
            st.success(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        else:
            # Use default parameters
            best_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
            with st.spinner("Training KNN with default parameters..."):
                best_knn.fit(X_train, y_train)
            st.success("KNN trained with default parameters")

        self.models['knn'] = best_knn
        self.training_history['knn'] = {
            'best_params': grid_search.best_params_ if cv_tuning else {'n_neighbors': 5, 'weights': 'uniform', 'metric': 'euclidean'},
            'cv_score': grid_search.best_score_ if cv_tuning else "Not calculated"
        }
        return best_knn

    def train_random_forest(self, X_train, y_train, cv_tuning=True):
        """Train Random Forest classifier"""
        st.info("Training Random Forest...")

        if cv_tuning:
            # Hyperparameter tuning with GridSearchCV
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='accuracy',
                n_jobs=-1, verbose=0, refit=True
            )

            with st.spinner("Tuning Random Forest hyperparameters..."):
                grid_search.fit(X_train, y_train)

            best_rf = grid_search.best_estimator_
            st.success(f"Best Random Forest parameters: {grid_search.best_params_}")
            st.success(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        else:
            # Use default parameters
            best_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            with st.spinner("Training Random Forest with default parameters..."):
                best_rf.fit(X_train, y_train)
            st.success("Random Forest trained with default parameters")

        self.models['random_forest'] = best_rf
        self.training_history['random_forest'] = {
            'best_params': grid_search.best_params_ if cv_tuning else {'n_estimators': 100},
            'cv_score': grid_search.best_score_ if cv_tuning else "Not calculated"
        }
        return best_rf

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        st.info(f"Evaluating {model_name.upper()}...")

        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)

        # Confusion matrix components
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        # AUC-ROC
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = auc(fpr, tpr)
        else:
            auc_score = 0.0

        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("AUC-ROC", f"{auc_score:.4f}")
        with col2:
            st.metric("Sensitivity", f"{recall:.4f}")
            st.metric("Specificity", f"{specificity:.4f}")
        with col3:
            st.metric("Precision", f"{precision:.4f}")
            st.metric("F1-Score", f"{f1:.4f}")
        with col4:
            st.metric("MCC", f"{mcc:.4f}")
            st.metric("NPV", f"{npv:.4f}")

        # Confusion matrix
        st.subheader("Confusion Matrix")
        fig_cm = px.imshow(cm,
                           text_auto=True,
                           aspect="auto",
                           labels=dict(x="Predicted", y="True", color="Count"),
                           x=self.class_names,
                           y=self.class_names,
                           title=f"Confusion Matrix - {model_name.upper()}",
                           color_continuous_scale='Blues')
        st.plotly_chart(fig_cm)

        # ROC Curve
        if y_proba is not None:
            st.subheader("ROC Curve")
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr,
                                         mode='lines',
                                         name=f'{model_name.upper()} (AUC = {auc_score:.4f})',
                                         line=dict(width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                         mode='lines',
                                         name='Random Classifier',
                                         line=dict(dash='dash', color='gray')))
            fig_roc.update_layout(title=f'ROC Curve - {model_name.upper()}',
                                  xaxis_title='False Positive Rate',
                                  yaxis_title='True Positive Rate',
                                  width=600, height=400)
            st.plotly_chart(fig_roc)

        # Classification report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}").highlight_max(axis=0))

        return {
            'accuracy': accuracy,
            'auc_roc': auc_score,
            'sensitivity': recall,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'mcc': mcc,
            'npv': npv,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'confusion_matrix': cm,
            'classification_report': report
        }

    def compare_models(self, X_test, y_test):
        """Compare performance of all trained models"""
        st.header("Model Comparison")

        results = {}
        progress_bar = st.progress(0)
        models_list = list(self.models.keys())

        for i, model_name in enumerate(models_list):
            st.subheader(f"{model_name.upper()} Evaluation")
            results[model_name] = self.evaluate_model(self.models[model_name], X_test, y_test, model_name)
            progress_bar.progress((i + 1) / len(models_list))

        # Create comparison table
        comparison_df = pd.DataFrame(results).T
        metrics_to_show = ['accuracy', 'auc_roc', 'sensitivity', 'specificity', 'precision', 'f1_score', 'mcc']
        comparison_df = comparison_df[metrics_to_show].round(4)

        st.subheader("Comparison Summary")
        st.dataframe(comparison_df.style.format("{:.4f}").highlight_max(axis=0))

        # Plot comparison
        st.subheader("Performance Comparison Chart")
        fig = go.Figure()
        for metric in metrics_to_show:
            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=list(results.keys()),
                y=[results[model][metric] for model in results.keys()],
                text=[f'{results[model][metric]:.4f}' for model in results.keys()],
                textposition='auto',
            ))

        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            yaxis=dict(range=[0, 1]),
            width=800, height=500
        )
        st.plotly_chart(fig)

        # Find best model
        best_model_name = self.select_best_model(results)
        st.success(f"BEST MODEL: {best_model_name.upper()}")

        return results, best_model_name

    def select_best_model(self, results):
        """Select best model based on multiple metrics"""
        scores = {}

        for model_name, metrics in results.items():
            # Weighted score giving importance to accuracy, AUC, and F1
            score = (
                    metrics['accuracy'] * 0.4 +
                    metrics['auc_roc'] * 0.3 +
                    metrics['f1_score'] * 0.2 +
                    metrics['mcc'] * 0.1
            )
            scores[model_name] = score

        return max(scores, key=scores.get)

    def predict_single_image(self, image, model_name='svm'):
        """Predict a single image using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

        # Convert PIL image to numpy array
        img_array = np.array(image)

        # Convert to RGB if needed
        if len(img_array.shape) == 2:  # Grayscale image
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # Already RGB
            pass
        else:
            raise ValueError("Unsupported image format")

        # Resize and preprocess
        img = cv2.resize(img_array, (self.img_width, self.img_height))
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

        return result, confidence, prediction_proba

    def save_models(self, filename='chest_xray_models.pkl'):
        """Save all models and preprocessors"""
        save_data = {
            'models': self.models,
            'scaler': self.scaler,
            'pca': self.pca,
            'le': self.le,
            'img_height': self.img_height,
            'img_width': self.img_width,
            'class_names': self.class_names
        }
        joblib.dump(save_data, filename)
        st.success(f"Models saved to {filename}")

    def load_models(self, filename='chest_xray_models.pkl'):
        """Load models and preprocessors"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")

        save_data = joblib.load(filename)
        self.models = save_data['models']
        self.scaler = save_data['scaler']
        self.pca = save_data['pca']
        self.le = save_data['le']
        self.img_height = save_data['img_height']
        self.img_width = save_data['img_width']
        self.class_names = save_data['class_names']
        st.success(f"Models loaded from {filename}")
        st.info(f"Available models: {list(self.models.keys())}")


def main():
    st.set_page_config(
        page_title="Chest X-Ray Pneumonia Detection",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #2e86ab;
            margin-bottom: 1rem;
        }
        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .warning-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">Chest X-Ray Pneumonia Detection</h1>', unsafe_allow_html=True)

    # Initialize predictor in session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = ChestXRayMLPredictor(img_height=100, img_width=100)

    predictor = st.session_state.predictor

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Home", "Train Model", "Prediction"])

    with tab1:
        st.markdown('<h2 class="sub-header">Welcome to Chest X-Ray Pneumonia Detection System</h2>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            ### About This Application

            This web application uses **Machine Learning** to detect pneumonia from chest X-ray images. 
            The system employs three powerful classifiers:

            - **Support Vector Machine (SVM)** with RBF kernel
            - **K-Nearest Neighbors (KNN)**
            - **Random Forest**

            ### Key Features:

            **Advanced Feature Extraction**: Histogram, texture, and edge features  
            **Hyperparameter Tuning**: Grid search with cross-validation  
            **Comprehensive Evaluation**: Multiple metrics including AUC-ROC, MCC, Specificity  
            **Research-Ready Visualizations**: Learning curves, calibration curves, ROC curves  

            ### How to Use:

            1. **Train Model Tab**: Train new models or load existing ones
            2. **Prediction Tab**: Upload X-ray images for classification
            3. **View Results**: See detailed analysis and confidence scores

            ### Dataset Information:

            The model is trained on the Chest X-Ray Images (Pneumonia) dataset, 
            which contains X-ray images classified as **Normal** or **Pneumonia**.
            """)

        with col2:
            st.image(
                "https://www.mdpi.com/children/children-11-00122/article_deploy/html/images/children-11-00122-g006-550.jpg",
                caption="Sample Chest X-Ray Image", use_container_width=True)

            st.markdown("""
            ### Model Specifications:
            - **Classes**: Normal, Pneumonia
            - **Image Size**: 100x100 pixels
            - **Models**: SVM, KNN, Random Forest
            - **Feature Extraction**: Advanced feature engineering
            - **Target Accuracy**: >95% on test data
            """)

        # Quick actions
        st.markdown("---")
        st.subheader("Quick Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Check for Pre-trained Models", use_container_width=True):
                if os.path.exists('chest_xray_models.pkl'):
                    st.success("Pre-trained models found!")
                    try:
                        predictor.load_models('chest_xray_models.pkl')
                        st.session_state.models_loaded = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading models: {e}")
                else:
                    st.warning("No pre-trained models found. Please train models first.")

        with col2:
            if st.button("Train Default Models", use_container_width=True):
                st.info("Please go to the 'Train Model' tab to train models with your preferred settings.")

        with col3:
            if st.button("Try Prediction", use_container_width=True):
                st.info("Please go to the 'Prediction' tab to upload and analyze X-ray images.")

    with tab2:
        st.markdown('<h2 class="sub-header">Model Training & Evaluation</h2>', unsafe_allow_html=True)

        # Training configuration
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Training Configuration")
            data_dir = st.text_input("Dataset Directory", "dataset/chest_xray")
            use_advanced_features = st.checkbox("Use Advanced Features", value=False)
            use_pca = st.checkbox("Use PCA", value=True)
            n_components = st.slider("PCA Components", 10, 100, 50)
            cv_tuning = st.checkbox("Hyperparameter Tuning", value=True)

        with col2:
            st.subheader("Model Selection")
            train_svm = st.checkbox("Train SVM", value=True)
            train_knn = st.checkbox("Train KNN", value=True)
            train_rf = st.checkbox("Train Random Forest", value=True)

            # Load existing models
            st.subheader("Model Management")
            model_file = st.text_input("Model File", "chest_xray_models.pkl")

            col_load, col_save = st.columns(2)
            with col_load:
                if st.button("Load Models"):
                    try:
                        predictor.load_models(model_file)
                        st.session_state.models_loaded = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading models: {e}")

            with col_save:
                if st.button("Save Models"):
                    try:
                        predictor.save_models(model_file)
                    except Exception as e:
                        st.error(f"Error saving models: {e}")

        # Training section
        st.markdown("---")
        st.subheader("Start Training")

        if st.button("Train Selected Models", type="primary", use_container_width=True):
            if not any([train_svm, train_knn, train_rf]):
                st.error("Please select at least one model to train.")
            else:
                try:
                    # Debug dataset first
                    with st.spinner("Checking dataset structure..."):
                        if not predictor.debug_dataset_structure(data_dir):
                            st.error("Dataset structure issue detected. Please check your dataset path.")
                            return

                    # Load and preprocess data
                    with st.spinner("Loading and preprocessing images..."):
                        features, labels = predictor.load_and_preprocess_images(data_dir)
                        if features is None:
                            return
                        X_train, X_test, y_train, y_test = predictor.preprocess_data(
                            use_advanced_features=use_advanced_features,
                            use_pca=use_pca,
                            n_components=n_components
                        )

                    # Train selected models
                    if train_svm:
                        predictor.train_svm(X_train, y_train, cv_tuning=cv_tuning)

                    if train_knn:
                        predictor.train_knn(X_train, y_train, cv_tuning=cv_tuning)

                    if train_rf:
                        predictor.train_random_forest(X_train, y_train, cv_tuning=cv_tuning)

                    # Store test data for evaluation
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.training_complete = True

                    st.success("Training completed successfully!")

                except Exception as e:
                    st.error(f"Error during training: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # Display training history if available
        if hasattr(predictor, 'training_history') and predictor.training_history:
            st.markdown("---")
            st.subheader("Training History")

            for model_name, history in predictor.training_history.items():
                with st.expander(f"{model_name.upper()} Training Details"):
                    st.write("Best Parameters:", history['best_params'])
                    st.write("Cross-validation Score:", history['cv_score'])

        # Model evaluation section
        if st.session_state.get('training_complete', False) and predictor.models:
            st.markdown("---")
            st.subheader("Model Evaluation")

            # Individual model evaluation
            model_to_evaluate = st.selectbox("Select model for detailed evaluation", list(predictor.models.keys()))

            if st.button("Evaluate Selected Model"):
                results = predictor.evaluate_model(
                    predictor.models[model_to_evaluate],
                    st.session_state.X_test,
                    st.session_state.y_test,
                    model_to_evaluate
                )

            # Compare all models
            if len(predictor.models) > 1:
                st.markdown("---")
                if st.button("Compare All Models"):
                    with st.spinner("Comparing models..."):
                        results, best_model = predictor.compare_models(
                            st.session_state.X_test,
                            st.session_state.y_test
                        )
                        st.session_state.comparison_results = results
                        st.session_state.best_model = best_model

    with tab3:
        st.markdown('<h2 class="sub-header">Pneumonia Prediction</h2>', unsafe_allow_html=True)

        if not predictor.models:
            st.warning("No models loaded. Please train or load models first.")
            st.info("Go to the 'Train Model' tab to train new models or load existing ones.")
        else:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Upload X-Ray Image")
                uploaded_file = st.file_uploader(
                    "Choose a chest X-ray image",
                    type=['jpeg', 'jpg', 'png'],
                    help="Upload a chest X-ray image in JPG, JPEG, or PNG format"
                )

                if uploaded_file is not None:
                    # Display uploaded image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)

                    # Model selection
                    st.subheader("Model Selection")
                    selected_model = st.selectbox(
                        "Choose model for prediction",
                        list(predictor.models.keys()),
                        index=0
                    )

                    # Single image prediction button
                    if st.button("Analyze Image", type="primary", use_container_width=True):
                        with st.spinner("Analyzing image..."):
                            try:
                                result, confidence, probabilities = predictor.predict_single_image(
                                    image, selected_model
                                )

                                st.session_state.prediction_result = {
                                    'result': result,
                                    'confidence': confidence,
                                    'probabilities': probabilities,
                                    'model': selected_model
                                }

                            except Exception as e:
                                st.error(f"Error during prediction: {e}")

            with col2:
                st.subheader("Prediction Results")

                if 'prediction_result' in st.session_state:
                    result = st.session_state.prediction_result['result']
                    confidence = st.session_state.prediction_result['confidence']
                    probabilities = st.session_state.prediction_result['probabilities']
                    model_used = st.session_state.prediction_result['model']

                    # Display result with color coding
                    if result == 'NORMAL':
                        st.markdown(f"""
                        <div style='padding: 2rem; border-radius: 1rem; background-color: #d4edda; border: 2px solid #c3e6cb; text-align: center;'>
                            <h1 style='color: #155724; margin: 0;'>NORMAL</h1>
                            <p style='color: #155724; font-size: 1.2rem; margin: 1rem 0 0 0;'>
                            Confidence: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='padding: 2rem; border-radius: 1rem; background-color: #f8d7da; border: 2px solid #f5c6cb; text-align: center;'>
                            <h1 style='color: #721c24; margin: 0;'>PNEUMONIA DETECTED</h1>
                            <p style='color: #721c24; font-size: 1.2rem; margin: 1rem 0 0 0;'>
                            Confidence: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.subheader("Probability Distribution")

                    if probabilities is not None:
                        fig = go.Figure(data=[
                            go.Bar(x=['NORMAL', 'PNEUMONIA'],
                                   y=probabilities,
                                   marker_color=['green', 'red'],
                                   text=[f'{prob:.4f}' for prob in probabilities],
                                   textposition='auto')
                        ])
                        fig.update_layout(
                            title="Class Probabilities",
                            yaxis_title="Probability",
                            yaxis=dict(range=[0, 1]),
                            width=400, height=400
                        )
                        st.plotly_chart(fig)

                    st.subheader("Model Information")
                    st.write(f"Model used: **{model_used.upper()}**")
                    st.write(f"Prediction: **{result}**")
                    st.write(f"Confidence: **{confidence:.4f}**")

                    if probabilities is not None:
                        col_prob1, col_prob2 = st.columns(2)
                        with col_prob1:
                            st.metric("Normal Probability", f"{probabilities[0]:.4f}")
                        with col_prob2:
                            st.metric("Pneumonia Probability", f"{probabilities[1]:.4f}")

                else:
                    st.info("Upload an X-ray image and click 'Analyze Image' to see prediction results.")


if __name__ == "__main__":
    main()
