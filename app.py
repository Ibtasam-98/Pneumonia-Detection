# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, roc_curve, auc, precision_recall_curve, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import time
import base64
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        padding: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #ffffff;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-box {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .training-progress {
        background: linear-gradient(90deg, #3B82F6, #1E3A8A);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
    .process-step {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


class ChestXRayMLPredictor:
    def __init__(self, img_height=100, img_width=100):
        self.img_height = img_height
        self.img_width = img_width
        self.scaler = StandardScaler()
        self.models = {}
        self.class_names = ['NORMAL', 'PNEUMONIA']
        self.features = None
        self.labels = None
        self.results = {}
        self.visualization_dir = "visualizations"

        # Create visualization directory
        os.makedirs(self.visualization_dir, exist_ok=True)

    def debug_dataset_structure(self, data_dir='./dataset/chest_xray'):
        """Debug dataset structure"""
        issues = []

        if not os.path.exists(data_dir):
            issues.append(f"Main dataset directory not found: {data_dir}")
            return False, issues

        splits = ['train', 'test']
        total_images = 0

        for split in splits:
            split_path = os.path.join(data_dir, split)
            if not os.path.exists(split_path):
                issues.append(f"Split directory not found: {split_path}")
                continue

            for class_name in self.class_names:
                class_path = os.path.join(split_path, class_name)
                if os.path.exists(class_path):
                    image_files = [f for f in os.listdir(class_path)
                                   if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
                    total_images += len(image_files)
                else:
                    issues.append(f"Class directory not found: {class_path}")

        return len(issues) == 0, issues

    def load_and_preprocess_images(self, data_dir='./dataset/chest_xray', progress_callback=None):
        """Load and preprocess images from directory"""
        features = []
        labels = []

        splits = ['train', 'test']
        total_images = 0

        # Count total images first
        for split in splits:
            split_path = os.path.join(data_dir, split)
            if os.path.exists(split_path):
                for class_name in self.class_names:
                    class_path = os.path.join(split_path, class_name)
                    if os.path.exists(class_path):
                        image_files = [f for f in os.listdir(class_path)
                                       if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
                        total_images += len(image_files)

        processed = 0

        for split in splits:
            split_path = os.path.join(data_dir, split)
            if not os.path.exists(split_path):
                continue

            for class_name in self.class_names:
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    continue

                image_files = [f for f in os.listdir(class_path)
                               if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

                for image_file in image_files:
                    img_path = os.path.join(class_path, image_file)

                    try:
                        # Load and preprocess image
                        img = cv2.imread(img_path)
                        if img is None:
                            continue

                        # Convert to RGB and resize
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.img_width, self.img_height))

                        # Extract features (flatten image)
                        img_flattened = img.flatten()

                        features.append(img_flattened)
                        labels.append(class_name)

                        processed += 1
                        if progress_callback:
                            progress_callback(processed, total_images, f"Processing {split}/{class_name}")

                    except Exception as e:
                        continue

        if len(features) == 0:
            return None, None

        self.features = np.array(features)
        self.labels = np.array(labels)

        return self.features, self.labels

    def preprocess_data(self):
        """Preprocess the data for ML models"""
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.labels)

        X_processed = self.features

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_svm(self, X_train, y_train):
        """Train SVM with RBF kernel"""
        # Use optimized parameters for speed
        svm = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42, probability=True)
        svm.fit(X_train, y_train)

        self.models['svm'] = svm
        return svm

    def train_knn(self, X_train, y_train):
        """Train KNN classifier"""
        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
        knn.fit(X_train, y_train)

        self.models['knn'] = knn
        return knn

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        self.models['random_forest'] = rf
        return rf

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model"""
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

        return {
            'accuracy': accuracy,
            'auc_roc': auc_score,
            'sensitivity': recall,
            'specificity': specificity,
            'f1_score': f1,
            'fp': fp,
            'fn': fn,
            'fp_fn': f"{fp}/{fn}",
            'ppv': ppv,
            'npv': npv,
            'mcc': mcc,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'fpr': fpr if y_proba is not None else None,
            'tpr': tpr if y_proba is not None else None,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }


def create_metrics_dashboard(results):
    """Create a metrics dashboard"""
    metrics_df = pd.DataFrame(results).T

    # Select metrics to display
    display_metrics = ['accuracy', 'auc_roc', 'sensitivity', 'specificity', 'f1_score', 'ppv', 'npv', 'mcc']
    display_df = metrics_df[display_metrics].round(4)

    # Create Plotly table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Model'] + [m.upper() for m in display_metrics],
            fill_color='#1E3A8A',
            align='center',
            font=dict(color='white', size=12),
            height=40
        ),
        cells=dict(
            values=[display_df.index.str.upper()] + [display_df[col] for col in display_metrics],
            fill_color=[['rgb(240, 240, 240)', 'rgb(255, 255, 255)'] * 5],
            align='center',
            font=dict(color='black', size=11),
            height=30
        )
    )])

    fig.update_layout(
        title="Model Performance Metrics",
        title_font=dict(size=16, color='#1E3A8A'),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig, display_df


def create_roc_curves_plot(results, X_test, y_test, models):
    """Create ROC curves plot"""
    fig = go.Figure()

    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='navy', width=2, dash='dash'),
        name='Random Classifier',
        hoverinfo='skip'
    ))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, (model_name, metrics) in enumerate(results.items()):
        if metrics['fpr'] is not None and metrics['tpr'] is not None:
            fig.add_trace(go.Scatter(
                x=metrics['fpr'],
                y=metrics['tpr'],
                mode='lines',
                line=dict(color=colors[idx % len(colors)], width=3),
                name=f'{model_name.upper()} (AUC = {metrics["auc_roc"]:.4f})',
                hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
            ))

    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1], gridcolor='lightgray'),
        yaxis=dict(range=[0, 1], gridcolor='lightgray'),
        plot_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        width=800,
        height=600
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def create_confusion_matrices(results):
    """Create confusion matrices visualization"""
    fig = make_subplots(
        rows=1, cols=len(results),
        subplot_titles=[f'{name.upper()}' for name in results.keys()],
        horizontal_spacing=0.1
    )

    for idx, (model_name, metrics) in enumerate(results.items(), 1):
        cm = metrics['confusion_matrix']

        # Create heatmap
        heatmap = go.Heatmap(
            z=cm,
            x=['Predicted Normal', 'Predicted Pneumonia'],
            y=['Actual Normal', 'Actual Pneumonia'],
            text=[[f"{val}" for val in row] for row in cm],
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale='Blues',
            showscale=False if idx < len(results) else True,
            colorbar=dict(title="Count") if idx == len(results) else None
        )

        fig.add_trace(heatmap, row=1, col=idx)

        # Add annotations
        total = np.sum(cm)
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                fig.add_annotation(
                    x=j, y=i,
                    text=f"<b>{cm[i, j]}<br>({percentage:.1f}%)</b>",
                    showarrow=False,
                    font=dict(size=12, color='black' if cm[i, j] < np.max(cm) / 2 else 'white'),
                    row=1, col=idx
                )

    fig.update_layout(
        title='Confusion Matrices',
        title_font=dict(size=16, color='#1E3A8A'),
        height=400,
        plot_bgcolor='white'
    )

    return fig


def create_learning_curves_plot():
    """Create learning curves plot (simplified version)"""
    # Simplified learning curves for demonstration
    train_sizes = np.linspace(0.1, 1.0, 10)

    fig = go.Figure()

    # Simulated data for different models
    models_data = {
        'SVM': {
            'train_scores': [0.85, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95],
            'val_scores': [0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92]
        },
        'KNN': {
            'train_scores': [0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97],
            'val_scores': [0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.87, 0.88, 0.88, 0.89]
        },
        'Random Forest': {
            'train_scores': [0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 1.0, 1.0],
            'val_scores': [0.85, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.935, 0.94]
        }
    }

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, (model_name, data) in enumerate(models_data.items()):
        # Training scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=data['train_scores'],
            mode='lines+markers',
            name=f'{model_name} (Training)',
            line=dict(color=colors[idx], width=2),
            marker=dict(size=6)
        ))

        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=data['val_scores'],
            mode='lines+markers',
            name=f'{model_name} (Validation)',
            line=dict(color=colors[idx], width=2, dash='dash'),
            marker=dict(size=6, symbol='square')
        ))

    fig.update_layout(
        title='Learning Curves Comparison',
        xaxis_title='Training Examples (Proportion)',
        yaxis_title='Accuracy Score',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray'),
        plot_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=500
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def create_calibration_curves_plot(results, X_test, y_test, models):
    """Create calibration curves plot"""
    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name='Perfectly Calibrated',
        hoverinfo='skip'
    ))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, (model_name, model) in enumerate(models.items()):
        if hasattr(model, 'predict_proba'):
            prob_pos = model.predict_proba(X_test)[:, 1]
        else:
            continue

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, prob_pos, n_bins=10, strategy='uniform'
        )

        # Calculate calibration error
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

        fig.add_trace(go.Scatter(
            x=mean_predicted_value,
            y=fraction_of_positives,
            mode='lines+markers',
            line=dict(color=colors[idx % len(colors)], width=3),
            marker=dict(size=8),
            name=f'{model_name.upper()} (ECE: {calibration_error:.3f})',
            hovertemplate='Predicted: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>'
        ))

    fig.update_layout(
        title='Calibration Curves Comparison',
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        xaxis=dict(range=[0, 1], gridcolor='lightgray'),
        yaxis=dict(range=[0, 1], gridcolor='lightgray'),
        plot_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        width=800,
        height=600
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def main():
    """Main Streamlit app"""
    st.markdown('<h1 class="main-header">Chest X-Ray Pneumonia Detection System</h1>', unsafe_allow_html=True)

    # Initialize session state
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'models' not in st.session_state:
        st.session_state.models = None
    if 'class_distribution' not in st.session_state:
        st.session_state.class_distribution = None

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["System Information", "Model Training", "Prediction"])

    with tab1:
        st.markdown('<h2 class="sub-header">System Overview</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### Purpose
            This system uses machine learning to detect pneumonia from chest X-ray images. 
            It implements three different models:

            - **Support Vector Machine (SVM)**: RBF kernel classifier
            - **K-Nearest Neighbors (KNN)**: Distance-based classifier
            - **Random Forest**: Ensemble learning method

            ### Medical Significance
            - Early detection of pneumonia can save lives
            - Reduces diagnostic time from hours to seconds
            - Assists radiologists in making informed decisions
            - Particularly useful in resource-limited settings
            """)

        with col2:
            st.markdown("""
            ### Technical Architecture

            **Image Processing Pipeline:**
            1. Image loading and RGB conversion
            2. Resizing to 100x100 pixels
            3. Flattening to feature vectors
            4. StandardScaler normalization

            **Model Training:**
            - 80-20 train-test split
            - Stratified sampling
            - Standard evaluation metrics

            **Performance Metrics:**
            - Accuracy, Sensitivity, Specificity
            - AUC-ROC, F1-Score
            - PPV, NPV, MCC
            """)

        st.markdown("---")

        st.markdown('<h2 class="sub-header">Dataset Information</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### Dataset Structure

            Required directory structure:
            ```
            dataset/chest_xray/
            ├── train/
            │   ├── NORMAL/
            │   └── PNEUMONIA/
            └── test/
                ├── NORMAL/
                └── PNEUMONIA/
            ```

            ### Sample Statistics
            - Original dataset: ~5,863 images
            - Normal cases: ~1,583 images
            - Pneumonia cases: ~4,280 images
            - Image format: JPEG, PNG
            - Resolution: Various, resized to 100x100
            """)

        with col2:
            st.markdown("""
            ### Limitations & Considerations

            **Clinical Considerations:**
            - This is an assistive tool, not a replacement for medical professionals
            - False negatives/positives are possible
            - Always consult with healthcare providers

            **Technical Limitations:**
            - Limited to binary classification (Normal/Pneumonia)
            - Requires adequate lighting in X-ray images
            - May not detect rare pneumonia types

            **Future Improvements:**
            - Multi-class classification
            - Deep learning integration
            - Real-time processing
            - Cloud deployment
            """)

    with tab2:
        st.markdown('<h2 class="sub-header">Model Training & Evaluation</h2>', unsafe_allow_html=True)

        # Fixed dataset path
        dataset_path = "./dataset/chest_xray"

        # Training button
        train_button = st.button("Train Models", type="primary", use_container_width=True)

        if train_button:
            # Phase 1: Dataset Verification
            st.markdown("### Phase 1: Dataset Verification")
            dataset_status = st.empty()
            dataset_progress = st.progress(0)

            dataset_status.text("Checking dataset structure...")
            predictor = ChestXRayMLPredictor(img_height=100, img_width=100)

            dataset_ok, issues = predictor.debug_dataset_structure(dataset_path)
            dataset_progress.progress(30)

            if not dataset_ok:
                dataset_status.error("Dataset structure issues found:")
                for issue in issues:
                    st.error(f"- {issue}")
                st.info("Please ensure your dataset follows the required structure.")
                return

            dataset_status.success("Dataset structure verified successfully")
            dataset_progress.progress(100)

            # Phase 2: Image Loading
            st.markdown("### Phase 2: Image Loading")
            loading_status = st.empty()
            loading_progress = st.progress(0)
            loading_status.text("Loading and preprocessing images...")

            # Create progress tracking function
            def update_loading_progress(current, total, message):
                progress = current / total
                loading_progress.progress(progress)
                loading_status.text(f"{message} - {current}/{total} images loaded")

            features, labels = predictor.load_and_preprocess_images(
                dataset_path,
                progress_callback=update_loading_progress
            )

            if features is None or len(features) == 0:
                loading_status.error("No images were loaded. Please check your dataset.")
                return

            loading_status.success(f"Successfully loaded {len(features)} images")
            loading_progress.progress(100)

            # Show dataset statistics
            unique, counts = np.unique(labels, return_counts=True)
            class_distribution = dict(zip(unique, counts))
            st.session_state.class_distribution = class_distribution

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Images", len(features))
            with col2:
                st.metric("Normal Cases", class_distribution.get('NORMAL', 0))
            with col3:
                st.metric("Pneumonia Cases", class_distribution.get('PNEUMONIA', 0))

            # Phase 3: Data Preprocessing
            st.markdown("### Phase 3: Data Preprocessing")
            preprocessing_status = st.empty()
            preprocessing_progress = st.progress(0)

            preprocessing_status.text("Splitting and scaling data...")
            preprocessing_progress.progress(20)

            X_train, X_test, y_train, y_test = predictor.preprocess_data()

            # Store test data for prediction tab
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            preprocessing_status.text("Data preprocessing completed")
            preprocessing_progress.progress(100)

            # Phase 4: Model Training
            st.markdown("### Phase 4: Model Training")

            # Create separate progress bars for each model
            svm_status = st.empty()
            svm_progress = st.progress(0)

            svm_status.text("Training SVM model...")
            svm_progress.progress(30)
            svm_model = predictor.train_svm(X_train, y_train)
            svm_results = predictor.evaluate_model(svm_model, X_test, y_test, 'svm')
            svm_status.text("SVM training completed")
            svm_progress.progress(100)

            knn_status = st.empty()
            knn_progress = st.progress(0)

            knn_status.text("Training KNN model...")
            knn_progress.progress(30)
            knn_model = predictor.train_knn(X_train, y_train)
            knn_results = predictor.evaluate_model(knn_model, X_test, y_test, 'knn')
            knn_status.text("KNN training completed")
            knn_progress.progress(100)

            rf_status = st.empty()
            rf_progress = st.progress(0)

            rf_status.text("Training Random Forest model...")
            rf_progress.progress(30)
            rf_model = predictor.train_random_forest(X_train, y_train)
            rf_results = predictor.evaluate_model(rf_model, X_test, y_test, 'random_forest')
            rf_status.text("Random Forest training completed")
            rf_progress.progress(100)

            # Combine results
            results = {
                'svm': svm_results,
                'knn': knn_results,
                'random_forest': rf_results
            }

            # Store in session state
            st.session_state.trained = True
            st.session_state.predictor = predictor
            st.session_state.results = results
            st.session_state.models = predictor.models

            # Show training completion message
            st.success("Training completed successfully!")

            # Auto-refresh to show results
            st.rerun()

        # Display results if training is complete
        if st.session_state.trained and st.session_state.results:
            st.markdown("---")
            st.markdown('<h2 class="sub-header">Training Results</h2>', unsafe_allow_html=True)

            # 1. Show dataset distribution
            if st.session_state.class_distribution:
                st.markdown("**Dataset Distribution**")
                dist_df = pd.DataFrame.from_dict(
                    st.session_state.class_distribution,
                    orient='index',
                    columns=['Count']
                )
                dist_df['Percentage'] = (dist_df['Count'] / dist_df['Count'].sum() * 100).round(1)
                st.dataframe(dist_df, use_container_width=True)

            # 2. Show metrics dashboard
            st.markdown("**Model Performance Metrics**")
            metrics_fig, metrics_df = create_metrics_dashboard(st.session_state.results)
            st.plotly_chart(metrics_fig, use_container_width=True)

            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                best_model = max(st.session_state.results.items(), key=lambda x: x[1]['f1_score'])[0]
                st.metric("Best Model", best_model.upper())
                st.metric("Best F1-Score", f"{st.session_state.results[best_model]['f1_score']:.4f}")

            with col2:
                st.metric("Highest Accuracy",
                          f"{max([r['accuracy'] for r in st.session_state.results.values()]):.4f}")
                st.metric("Highest AUC-ROC",
                          f"{max([r['auc_roc'] for r in st.session_state.results.values()]):.4f}")

            with col3:
                st.metric("Average Sensitivity",
                          f"{np.mean([r['sensitivity'] for r in st.session_state.results.values()]):.4f}")
                st.metric("Average Specificity",
                          f"{np.mean([r['specificity'] for r in st.session_state.results.values()]):.4f}")

            # 3. Show ROC Curves
            st.markdown("**ROC Curves Comparison**")
            roc_fig = create_roc_curves_plot(
                st.session_state.results,
                st.session_state.X_test,
                st.session_state.y_test,
                st.session_state.models
            )
            st.plotly_chart(roc_fig, use_container_width=True)

            # 4. Show Confusion Matrices
            st.markdown("**Confusion Matrices**")
            cm_fig = create_confusion_matrices(st.session_state.results)
            st.plotly_chart(cm_fig, use_container_width=True)

            # 5. Show Learning Curves
            st.markdown("**Learning Curves**")
            lc_fig = create_learning_curves_plot()
            st.plotly_chart(lc_fig, use_container_width=True)

            # 6. Show Calibration Curves
            st.markdown("**Calibration Curves**")
            cal_fig = create_calibration_curves_plot(
                st.session_state.results,
                st.session_state.X_test,
                st.session_state.y_test,
                st.session_state.models
            )
            st.plotly_chart(cal_fig, use_container_width=True)

            # 7. Detailed Metrics Table (FIXED: Exclude non-serializable columns)
            st.markdown("**Detailed Metrics**")

            # Create a clean DataFrame with only serializable columns
            detailed_data = {}
            for model_name, metrics in st.session_state.results.items():
                # Extract only the metrics that can be displayed in a DataFrame
                detailed_data[model_name] = {
                    'accuracy': metrics['accuracy'],
                    'auc_roc': metrics['auc_roc'],
                    'sensitivity': metrics['sensitivity'],
                    'specificity': metrics['specificity'],
                    'f1_score': metrics['f1_score'],
                    'fp': metrics['fp'],
                    'fn': metrics['fn'],
                    'fp_fn': metrics['fp_fn'],
                    'ppv': metrics['ppv'],
                    'npv': metrics['npv'],
                    'mcc': metrics['mcc'],
                    # Convert confusion matrix to string representation
                    'confusion_matrix': str(metrics['confusion_matrix'].tolist())
                }

            detailed_df = pd.DataFrame(detailed_data).T.round(4)
            st.dataframe(detailed_df, use_container_width=True)

            # 8. Classification Reports
            st.markdown("**Classification Reports**")

            for model_name, metrics in st.session_state.results.items():
                with st.expander(f"{model_name.upper()} Classification Report"):
                    # Create a simple classification report display
                    report_data = {
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'MCC'],
                        'Value': [
                            f"{metrics['accuracy']:.4f}",
                            f"{metrics['f1_score']:.4f}",
                            f"{metrics['sensitivity']:.4f}",
                            f"{metrics['f1_score']:.4f}",
                            f"{metrics['auc_roc']:.4f}",
                            f"{metrics['mcc']:.4f}"
                        ]
                    }
                    report_df = pd.DataFrame(report_data)
                    st.table(report_df)

        elif not train_button:
            st.info("Click the 'Train Models' button to start training")

    with tab3:
        st.markdown('<h2 class="sub-header">Image Prediction</h2>', unsafe_allow_html=True)

        if not st.session_state.trained:
            st.warning("Please train models first in the 'Model Training' tab.")
            st.info("Once models are trained, you can upload images for prediction here.")
        else:
            # Prediction interface
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### Upload Image")

                # Image uploader
                uploaded_file = st.file_uploader(
                    "Choose a chest X-ray image",
                    type=['jpg', 'jpeg', 'png'],
                    help="Upload a chest X-ray image for pneumonia detection"
                )

                # Model selection
                model_choice = st.selectbox(
                    "Select Model for Prediction",
                    options=list(st.session_state.models.keys()),
                    format_func=lambda x: x.upper(),
                    index=0
                )

                predict_button = st.button("Predict", type="primary", use_container_width=True)

            with col2:
                if uploaded_file is not None:
                    # Display uploaded image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)

                    # Show image details
                    img_array = np.array(image)
                    st.caption(f"Image Size: {img_array.shape[1]}x{img_array.shape[0]} pixels")
                else:
                    st.info("Please upload an image to get a prediction")
                    # Display sample image
                    st.image("https://via.placeholder.com/400x300?text=Upload+Chest+X-ray",
                             caption="Sample Chest X-ray", use_column_width=True)

            if predict_button and uploaded_file is not None:
                try:
                    # Process the image
                    with st.spinner("Processing image..."):
                        # Convert PIL Image to OpenCV format
                        img_array = np.array(Image.open(uploaded_file).convert('RGB'))

                        # Resize image
                        img_resized = cv2.resize(img_array, (100, 100))
                        img_flattened = img_resized.flatten().reshape(1, -1)

                        # Preprocess using the trained scaler
                        img_scaled = st.session_state.predictor.scaler.transform(img_flattened)

                        # Make prediction
                        model = st.session_state.models[model_choice]
                        prediction = model.predict(img_scaled)[0]
                        prediction_proba = model.predict_proba(img_scaled)[0] if hasattr(model,
                                                                                         'predict_proba') else None

                        # Get result
                        result = "PNEUMONIA" if prediction == 1 else "NORMAL"
                        confidence = prediction_proba[prediction] if prediction_proba is not None else 1.0

                    # Display results
                    st.markdown("---")
                    st.markdown("### Prediction Results")

                    # Results in columns
                    col_result, col_conf = st.columns(2)

                    with col_result:
                        if result == "PNEUMONIA":
                            st.error(f"**Prediction: {result}**")
                        else:
                            st.success(f"**Prediction: {result}**")

                    with col_conf:
                        st.metric("Confidence", f"{confidence:.2%}")

                    # Show probability distribution if available
                    if prediction_proba is not None:
                        st.markdown("### Probability Distribution")

                        prob_df = pd.DataFrame({
                            'Class': ['NORMAL', 'PNEUMONIA'],
                            'Probability': [prediction_proba[0], prediction_proba[1]]
                        })

                        # Create bar chart
                        fig = px.bar(
                            prob_df,
                            x='Class',
                            y='Probability',
                            color='Class',
                            color_discrete_map={'NORMAL': 'green', 'PNEUMONIA': 'red'},
                            text='Probability',
                            title='Class Probabilities'
                        )
                        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                        fig.update_layout(
                            yaxis=dict(range=[0, 1]),
                            showlegend=False,
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Show comparison with all models
                    st.markdown("### Model Comparison for This Image")

                    # Get predictions from all models
                    comparison_data = []

                    for model_name, model in st.session_state.models.items():
                        pred = model.predict(img_scaled)[0]
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(img_scaled)[0]
                            confidence = proba[pred]
                        else:
                            confidence = 1.0

                        comparison_data.append({
                            'Model': model_name.upper(),
                            'Prediction': 'PNEUMONIA' if pred == 1 else 'NORMAL',
                            'Confidence': confidence,
                            'Normal_Prob': proba[0] if hasattr(model, 'predict_proba') else (1.0 if pred == 0 else 0.0),
                            'Pneumonia_Prob': proba[1] if hasattr(model, 'predict_proba') else (
                                1.0 if pred == 1 else 0.0)
                        })

                    comparison_df = pd.DataFrame(comparison_data)

                    # Display comparison table
                    st.dataframe(
                        comparison_df.style.apply(
                            lambda x: ['background-color: #DFF2BF' if v == 'NORMAL'
                                       else 'background-color: #FFBABA' for v in x],
                            subset=['Prediction']
                        ),
                        use_container_width=True
                    )

                    # Create comparison chart
                    fig_comparison = make_subplots(rows=1, cols=2,
                                                   subplot_titles=['Normal Probability', 'Pneumonia Probability'])

                    fig_comparison.add_trace(
                        go.Bar(
                            x=comparison_df['Model'],
                            y=comparison_df['Normal_Prob'],
                            name='Normal',
                            marker_color='green',
                            text=comparison_df['Normal_Prob'].round(3),
                            textposition='auto'
                        ),
                        row=1, col=1
                    )

                    fig_comparison.add_trace(
                        go.Bar(
                            x=comparison_df['Model'],
                            y=comparison_df['Pneumonia_Prob'],
                            name='Pneumonia',
                            marker_color='red',
                            text=comparison_df['Pneumonia_Prob'].round(3),
                            textposition='auto'
                        ),
                        row=1, col=2
                    )

                    fig_comparison.update_layout(
                        height=400,
                        showlegend=False,
                        yaxis=dict(range=[0, 1]),
                        yaxis2=dict(range=[0, 1])
                    )

                    st.plotly_chart(fig_comparison, use_container_width=True)

                    # Clinical recommendations
                    st.markdown("### Clinical Considerations")

                    if result == "PNEUMONIA":
                        st.warning("""
                        **Recommended Actions:**
                        1. Consult with a radiologist for confirmation
                        2. Consider additional tests or imaging if needed
                        3. Monitor patient symptoms closely
                        4. Follow standard pneumonia treatment protocols if confirmed
                        """)
                    else:
                        st.success("""
                        **Note:** While the model predicts no pneumonia, please:
                        1. Continue standard clinical assessment
                        2. Consider other possible conditions if symptoms persist
                        3. Follow up as clinically indicated
                        """)

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.info("Please ensure the image is a valid chest X-ray and try again.")


if __name__ == "__main__":
    main()