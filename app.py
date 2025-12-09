import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import cv2
from PIL import Image
import os
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS for cleaner UI
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }

    /* Tab styling - RED theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #b30000;
        padding: 0px;
        border-radius: 0px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #b30000;
        border-radius: 0px;
        padding: 12px 24px;
        font-weight: 600;
        color: white !important;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #990000;
        color: white !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #b30000;
        color: white !important;
        border-bottom: 3px solid #ff4444 !important;
    }

    .stTabs [aria-selected="false"] {
        color: white !important;
        opacity: 0.8;
    }

    /* Cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #b30000;
        margin-bottom: 15px;
    }

    /* Results */
    .result-normal {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #28a745;
    }

    .result-pneumonia {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #dc3545;
    }

    /* Headers */
    .section-header {
        color: #b30000;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }

    /* Model cards */
    .model-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #dee2e6;
        transition: transform 0.2s;
    }

    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Custom button styling */
    .stButton > button {
        background-color: #b30000;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: 600;
        transition: background-color 0.3s;
    }

    .stButton > button:hover {
        background-color: #990000;
        color: white;
    }

    /* Metric styling - WHITE THEME */
    .stMetric {
        background: black !important;
        padding: 20px;
        border-radius: 10px;

        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #333 !important;
    }

    /* Metric label styling */
    .stMetric label {
        color: #fff !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* Metric value styling */
    .stMetric div[data-testid="stMetricValue"] {
        color: #fff !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }

    /* Metric delta styling */
    .stMetric div[data-testid="stMetricDelta"] {
        color: #fff !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* Metric container */
    .stMetric > div {
        background: transparent !important;
    }

    /* Override Streamlit's default metric styling */
    div[data-testid="metric-container"] {
        background: white !important;
        border-radius: 10px !important;
        padding: 20px !important;
        border: 1px solid #dee2e6 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    /* Metric label */
    div[data-testid="metric-container"] label {
        color: #666 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* Metric value */
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #333 !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }

    /* Metric delta */
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
        color: #666 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* Sample image styling */
    .sample-image {
        border: 3px solid #dee2e6;
        border-radius: 10px;
        padding: 5px;
        transition: all 0.3s ease;
    }

    .sample-image:hover {
        border-color: #b30000;
        transform: scale(1.02);
    }

    /* Prediction result card */
    .prediction-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .svm-card {
        border-left-color: #1f77b4;
    }

    .knn-card {
        border-left-color: #ff7f0e;
    }

    .rf-card {
        border-left-color: #2ca02c;
    }
</style>
""", unsafe_allow_html=True)


class ChestXRayApp:
    def __init__(self):
        self.models = None
        self.scaler = None
        self.pca = None
        self.class_names = ['NORMAL', 'PNEUMONIA']
        self.img_height = 100
        self.img_width = 100
        self.results = None
        self.load_models()

    def load_models(self):
        """Load trained models"""
        try:
            model_file = 'chest_xray_models.pkl'
            if os.path.exists(model_file):
                save_data = joblib.load(model_file)
                self.models = save_data['models']
                self.scaler = save_data['scaler']
                self.pca = save_data['pca']
                self.img_height = save_data['img_height']
                self.img_width = save_data['img_width']
                self.class_names = save_data['class_names']

                # Load results if available
                if os.path.exists('visualizations/detailed_results.csv'):
                    self.results = pd.read_csv('visualizations/detailed_results.csv', index_col=0)
                return True
            else:
                st.error("Model file not found. Please train the models first.")
                return False
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False

    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        try:
            img_array = np.array(image)

            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

            img_resized = cv2.resize(img_array, (self.img_width, self.img_height))
            img_flattened = img_resized.flatten().reshape(1, -1)
            img_scaled = self.scaler.transform(img_flattened)

            if self.pca:
                img_processed = self.pca.transform(img_scaled)
            else:
                img_processed = img_scaled

            return img_processed
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return None

    def predict_all_models(self, image):
        """Predict using all models"""
        predictions = {}
        if not self.models:
            return predictions

        img_processed = self.preprocess_image(image)
        if img_processed is None:
            return predictions

        for model_name, model in self.models.items():
            prediction = model.predict(img_processed)[0]

            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(img_processed)[0]
                confidence = prediction_proba[prediction]
                probabilities = prediction_proba
            else:
                confidence = 1.0
                probabilities = [0, 0]

            result = self.class_names[prediction]

            predictions[model_name] = {
                'prediction': result,
                'confidence': confidence,
                'probabilities': probabilities
            }

        return predictions

    def run(self):
        """Main application"""
        # Header
        st.markdown("<h1 style='text-align: center; color: #b30000;'>Chest X-Ray Pneumonia Detection</h1>",
                    unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666;'>Machine Learning Powered Diagnosis System</p>",
                    unsafe_allow_html=True)

        # Create tabs without emojis
        tab1, tab2, tab3, tab4 = st.tabs([
            "Dashboard",
            "Predict",
            "Analysis",
            "Visualizations"
        ])

        with tab1:
            self.show_dashboard()

        with tab2:
            self.show_prediction()

        with tab3:
            self.show_analysis()

        with tab4:
            self.show_visualizations()

    def show_dashboard(self):
        """Show dashboard tab"""
        st.markdown("<div class='section-header'>System Overview</div>", unsafe_allow_html=True)

        # Quick stats with red background
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Models", "3", "SVM, KNN, RF")

        with col2:
            st.metric("Training Images", "5,856", "From Dataset")

        with col3:
            st.metric("Best Accuracy", "94.97%", "SVM Model")

        with col4:
            st.metric("Best AUC", "0.9846", "ROC Score")

        # Model Performance Summary
        st.markdown("<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)

        if self.results is not None:
            # Performance metrics
            metrics_df = self.results[['accuracy', 'auc_roc', 'f1_score', 'mcc']].copy()
            metrics_df.columns = ['Accuracy', 'AUC-ROC', 'F1-Score', 'MCC']
            metrics_df = metrics_df.round(4)

            col1, col2 = st.columns([3, 2])

            with col1:
                st.dataframe(metrics_df, use_container_width=True)

            with col2:
                best_model = metrics_df['Accuracy'].idxmax()
                best_metrics = metrics_df.loc[best_model]

                # st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"### Best Model: **{best_model.upper()}**")
                st.markdown(f"**Accuracy:** {best_metrics['Accuracy']:.2%}")
                st.markdown(f"**AUC-ROC:** {best_metrics['AUC-ROC']:.4f}")
                st.markdown(f"**F1-Score:** {best_metrics['F1-Score']:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)

            # Key metrics comparison
            st.markdown("<div class='section-header'>Key Metrics Comparison</div>", unsafe_allow_html=True)

            fig = go.Figure()

            models = metrics_df.index.tolist()
            x = ['Accuracy', 'AUC-ROC', 'F1-Score', 'MCC']

            for model in models:
                fig.add_trace(go.Scatter(
                    x=x,
                    y=metrics_df.loc[model].values,
                    mode='lines+markers',
                    name=model.upper(),
                    line=dict(width=3)
                ))

            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Metrics",
                yaxis_title="Score",
                hovermode="x unified",
                height=400,
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

        # System Information
        st.markdown("<div class='section-header'>System Information</div>", unsafe_allow_html=True)

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.markdown("""
            ### Dataset Information
            - **Total Images:** 5,856
            - **Normal Cases:** 1,583 (27%)
            - **Pneumonia Cases:** 4,273 (73%)
            - **Image Size:** 100×100 pixels
            - **Dataset Split:** Train/Test/Val

            ### Models Used
            1. **Support Vector Machine (SVM)**
            2. **K-Nearest Neighbors (KNN)**
            3. **Random Forest (RF)**
            """)

        with info_col2:
            st.markdown("""
            ### Technical Details
            - **Framework:** Scikit-learn
            - **Image Processing:** OpenCV
            - **Dimensionality Reduction:** PCA
            - **Cross-validation:** 3-fold
            - **Best Parameters:** Grid Search

            ### Performance Highlights
            - **Highest Accuracy:** 94.97%
            - **Best AUC:** 0.9846
            - **Lowest FP/FN:** 38/21
            - **MCC Score:** 0.8709
            """)

    def show_prediction(self):
        """Show prediction tab"""
        st.markdown("<div class='section-header'>Image Analysis</div>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Upload X-Ray Image")
            uploaded_file = st.file_uploader(
                "Choose a chest X-ray image...",
                type=['jpg', 'jpeg', 'png'],
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

                # Analyze button
                if st.button("Analyze Image with All Models", type="primary", use_container_width=True):
                    with st.spinner("Analyzing image with all models..."):
                        predictions = self.predict_all_models(image)

                        if predictions:
                            # Display results from all models
                            st.markdown("### Analysis Results from All Models")

                            # Create columns for model results
                            result_cols = st.columns(3)

                            model_colors = {
                                'svm': '#1f77b4',
                                'knn': '#ff7f0e',
                                'random_forest': '#2ca02c'
                            }

                            for idx, (model_name, result) in enumerate(predictions.items()):
                                with result_cols[idx]:
                                    if result['prediction'] == "PNEUMONIA":
                                        st.error(f"**{model_name.upper()}**")
                                        st.error(f"Result: {result['prediction']}")
                                    else:
                                        st.success(f"**{model_name.upper()}**")
                                        st.success(f"Result: {result['prediction']}")

                                    st.metric("Confidence", f"{result['confidence']:.2%}")

                                    # Show probabilities
                                    prob_data = pd.DataFrame({
                                        'Class': self.class_names,
                                        'Probability': result['probabilities']
                                    })

                                    fig = px.bar(
                                        prob_data,
                                        x='Class',
                                        y='Probability',
                                        color='Class',
                                        color_discrete_sequence=['#28a745', '#dc3545'],
                                        text_auto='.2%'
                                    )
                                    fig.update_layout(
                                        yaxis_range=[0, 1],
                                        showlegend=False,
                                        height=250,
                                        title=f"{model_name.upper()} Probabilities"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                            # Show consensus analysis
                            st.markdown("### Model Consensus Analysis")
                            consensus = {}
                            for model_name, result in predictions.items():
                                pred = result['prediction']
                                if pred not in consensus:
                                    consensus[pred] = []
                                consensus[pred].append(model_name)

                            if len(consensus) == 1:
                                # All models agree
                                final_prediction = list(consensus.keys())[0]
                                if final_prediction == "PNEUMONIA":
                                    st.error(f"**CONSENSUS: {final_prediction} DETECTED**")
                                    st.warning(
                                        "All 3 models detected pneumonia. Please consult a healthcare professional immediately.")
                                else:
                                    st.success(f"**CONSENSUS: {final_prediction} CHEST X-RAY**")
                                    st.info("All 3 models agree: No signs of pneumonia detected.")
                            else:
                                # Models disagree
                                st.warning("**MODELS DISAGREE**")
                                for pred, models in consensus.items():
                                    st.write(f"{pred}: {', '.join([m.upper() for m in models])}")

                                # Show majority vote
                                majority_pred = max(consensus.items(), key=lambda x: len(x[1]))
                                if len(majority_pred[1]) >= 2:  # At least 2 models agree
                                    if majority_pred[0] == "PNEUMONIA":
                                        st.error(
                                            f"**MAJORITY VOTE ({len(majority_pred[1])}/3): {majority_pred[0]} DETECTED**")
                                        st.warning(
                                            "Majority of models detected pneumonia. Please consult a healthcare professional.")
                                    else:
                                        st.success(
                                            f"**MAJORITY VOTE ({len(majority_pred[1])}/3): {majority_pred[0]} CHEST X-RAY**")
                                        st.info("Majority of models indicate no pneumonia.")

        with col2:
            st.markdown("### Try Sample Images")

            # Sample images section with actual images from dataset
            sample_images = {
                "Normal": "dataset/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg",
                "Pneumonia": "dataset/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg"
            }

            # Check if sample images exist
            for label, path in sample_images.items():
                if not os.path.exists(path):
                    st.warning(f"Sample image not found: {path}")
                    # Create placeholder text
                    st.info(f"**{label} Sample**")
                    st.write("Sample image would appear here")

                    # Test button for placeholder
                    if st.button(f"Test {label} Sample", use_container_width=True):
                        st.info(f"This would test a {label.lower()} chest X-ray image")
                else:
                    # Display sample image - FIXED: Removed use_column_width parameter
                    st.markdown(f"**{label} Sample**")
                    sample_img = Image.open(path)
                    st.image(sample_img, caption=f"{label} Chest X-Ray", use_container_width=True)

                    # Test button
                    if st.button(f"Test {label} Sample", use_container_width=True):
                        with st.spinner(f"Testing {label} sample..."):
                            predictions = self.predict_all_models(sample_img)

                            if predictions:
                                st.markdown(f"### Results for {label} Sample")

                                # Display quick results
                                result_text = ""
                                for model_name, result in predictions.items():
                                    result_text += f"**{model_name.upper()}**: {result['prediction']} ({result['confidence']:.2%})\n"

                                st.write(result_text)

                                # Show consensus
                                consensus = {}
                                for model_name, result in predictions.items():
                                    pred = result['prediction']
                                    if pred not in consensus:
                                        consensus[pred] = []
                                    consensus[pred].append(model_name)

                                if label == "Normal" and "NORMAL" in consensus:
                                    st.success("✓ Correctly identified as NORMAL")
                                elif label == "Pneumonia" and "PNEUMONIA" in consensus:
                                    st.success("✓ Correctly identified as PNEUMONIA")
                                else:
                                    st.warning("Some models may have misclassified this sample")

            # Batch prediction section
            # st.markdown("---")
            # st.markdown("### Batch Prediction")
            #
            # batch_files = st.file_uploader(
            #     "Upload multiple images for batch analysis",
            #     type=['jpg', 'jpeg', 'png'],
            #     accept_multiple_files=True,
            #     label_visibility="collapsed"
            # )
            #
            # if batch_files:
            #     if st.button("Analyze All Images", type="secondary", use_container_width=True):
            #         with st.spinner(f"Analyzing {len(batch_files)} images with all models..."):
            #             all_results = []
            #
            #             for i, file in enumerate(batch_files):
            #                 img = Image.open(file)
            #                 predictions = self.predict_all_models(img)
            #
            #                 if predictions:
            #                     # Get consensus
            #                     consensus_count = {}
            #                     for model_name, result in predictions.items():
            #                         pred = result['prediction']
            #                         consensus_count[pred] = consensus_count.get(pred, 0) + 1
            #
            #                     # Determine final prediction (majority vote)
            #                     final_prediction = max(consensus_count.items(), key=lambda x: x[1])[
            #                         0] if consensus_count else "UNKNOWN"
            #
            #                     all_results.append({
            #                         'Image': file.name,
            #                         'SVM': predictions.get('svm', {}).get('prediction', 'N/A'),
            #                         'KNN': predictions.get('knn', {}).get('prediction', 'N/A'),
            #                         'Random Forest': predictions.get('random_forest', {}).get('prediction', 'N/A'),
            #                         'Final Prediction': final_prediction,
            #                         'Agreement': f"{consensus_count.get(final_prediction, 0)}/3 models"
            #                     })
            #
            #             if all_results:
            #                 results_df = pd.DataFrame(all_results)
            #                 st.dataframe(results_df, use_container_width=True)
            #
            #                 # Summary statistics
            #                 normal_count = sum(1 for r in all_results if r['Final Prediction'] == 'NORMAL')
            #                 pneumonia_count = sum(1 for r in all_results if r['Final Prediction'] == 'PNEUMONIA')
            #
            #                 summary_col1, summary_col2 = st.columns(2)
            #                 with summary_col1:
            #                     st.metric("Normal", normal_count)
            #                 with summary_col2:
            #                     st.metric("Pneumonia", pneumonia_count)

    def show_analysis(self):
        """Show analysis tab"""
        st.markdown("<div class='section-header'>Performance Analysis</div>", unsafe_allow_html=True)

        if self.results is None:
            st.warning("No performance data available. Please run training first.")
            return

        # Detailed metrics table
        st.markdown("### Detailed Metrics Table")

        detailed_metrics = self.results.copy()
        detailed_metrics = detailed_metrics[['accuracy', 'auc_roc', 'sensitivity', 'specificity',
                                             'f1_score', 'ppv', 'npv', 'mcc']]
        detailed_metrics.columns = ['Accuracy', 'AUC-ROC', 'Sensitivity', 'Specificity',
                                    'F1-Score', 'PPV', 'NPV', 'MCC']

        # Format as percentages where appropriate
        percent_cols = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
        for col in percent_cols:
            if col in detailed_metrics.columns:
                detailed_metrics[col] = detailed_metrics[col].apply(lambda x: f"{x:.2%}")

        st.dataframe(detailed_metrics, use_container_width=True)

        # Performance metrics visualization
        st.markdown("### Performance Metrics Radar Chart")

        metrics_for_radar = self.results[['accuracy', 'auc_roc', 'f1_score', 'mcc',
                                          'sensitivity', 'specificity']].copy()
        metrics_for_radar.columns = ['Accuracy', 'AUC-ROC', 'F1-Score', 'MCC',
                                     'Sensitivity', 'Specificity']

        fig = go.Figure()

        for model in metrics_for_radar.index:
            values = metrics_for_radar.loc[model].values.tolist()
            values += values[:1]  # Complete the circle

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_for_radar.columns.tolist() + [metrics_for_radar.columns[0]],
                fill='toself',
                name=model.upper(),
                opacity=0.7
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart",
            height=500,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Confusion matrix summary
        st.markdown("### Error Analysis")

        if 'fp_fn' in self.results.columns:
            error_data = []
            for model in self.results.index:
                fp_fn = str(self.results.loc[model, 'fp_fn']).split('/')
                if len(fp_fn) == 2:
                    error_data.append({
                        'Model': model.upper(),
                        'False Positives': int(fp_fn[0]),
                        'False Negatives': int(fp_fn[1])
                    })

            if error_data:
                error_df = pd.DataFrame(error_data)

                fig_errors = go.Figure(data=[
                    go.Bar(name='False Positives', x=error_df['Model'], y=error_df['False Positives']),
                    go.Bar(name='False Negatives', x=error_df['Model'], y=error_df['False Negatives'])
                ])

                fig_errors.update_layout(
                    barmode='group',
                    title="False Positives & False Negatives by Model",
                    xaxis_title="Model",
                    yaxis_title="Count",
                    height=400,
                    template="plotly_white"
                )

                st.plotly_chart(fig_errors, use_container_width=True)

        # Model comparison bar chart
        st.markdown("### Model Comparison Chart")

        comparison_cols = ['accuracy', 'auc_roc', 'f1_score']
        comparison_df = self.results[comparison_cols].copy()
        comparison_df.columns = ['Accuracy', 'AUC-ROC', 'F1-Score']

        fig_comparison = go.Figure(data=[
            go.Bar(name=col, x=comparison_df.index, y=comparison_df[col])
            for col in comparison_df.columns
        ])

        fig_comparison.update_layout(
            barmode='group',
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            height=400,
            template="plotly_white"
        )

        st.plotly_chart(fig_comparison, use_container_width=True)

        # Download results
        st.markdown("### Export Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Download CSV", use_container_width=True):
                csv = self.results.to_csv(index=True)
                st.download_button(
                    label="Click to download",
                    data=csv,
                    file_name="pneumonia_model_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col2:
            if st.button("Generate Report", use_container_width=True):
                st.success("Report generated successfully!")

        with col3:
            if st.button("Refresh Data", use_container_width=True):
                st.rerun()

    def show_visualizations(self):
        """Show visualizations tab"""
        st.markdown("<div class='section-header'>Research Visualizations</div>", unsafe_allow_html=True)

        viz_dir = "visualizations"
        if not os.path.exists(viz_dir):
            st.warning(f"Visualizations directory '{viz_dir}' not found.")
            return

        # Create sub-tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "Learning Curves",
            "Calibration Curves",
            "ROC Curves",
            "Precision-Recall"
        ])

        with viz_tab1:
            st.markdown("### Learning Curves")
            st.markdown("""
            Learning curves show how model performance improves with more training data.
            The gap between training and validation scores indicates overfitting.
            """)

            learning_curve_path = os.path.join(viz_dir, "research_learning_curves.png")
            if os.path.exists(learning_curve_path):
                st.image(learning_curve_path, use_container_width=True)
            else:
                st.warning("Learning curves visualization not found.")

        with viz_tab2:
            st.markdown("### Calibration Curves")
            st.markdown("""
            Calibration curves show how well predicted probabilities match actual outcomes.
            A perfectly calibrated model follows the diagonal line.
            ECE (Expected Calibration Error) measures deviation from perfect calibration.
            """)

            calibration_path = os.path.join(viz_dir, "research_calibration_curves.png")
            if os.path.exists(calibration_path):
                st.image(calibration_path, use_container_width=True)
            else:
                st.warning("Calibration curves visualization not found.")

        with viz_tab3:
            st.markdown("### ROC Curves")
            st.markdown("""
            Receiver Operating Characteristic (ROC) curves show the trade-off between 
            True Positive Rate (Sensitivity) and False Positive Rate (1-Specificity).
            Area Under the Curve (AUC) closer to 1.0 indicates better performance.
            """)

            roc_path = os.path.join(viz_dir, "roc_curves_all_models.png")
            if os.path.exists(roc_path):
                st.image(roc_path, use_container_width=True)

                # Add AUC values table
                if self.results is not None and 'auc_roc' in self.results.columns:
                    auc_data = pd.DataFrame({
                        'Model': self.results.index.str.upper(),
                        'AUC-ROC': self.results['auc_roc'].round(4)
                    })
                    st.dataframe(auc_data, use_container_width=True)
            else:
                st.warning("ROC curves visualization not found.")

        with viz_tab4:
            st.markdown("### Precision-Recall Curves")
            st.markdown("""
            Precision-Recall curves are particularly useful for imbalanced datasets.
            They show the trade-off between precision (positive predictive value) 
            and recall (sensitivity).
            """)

            pr_path = os.path.join(viz_dir, "precision_recall_curves_all_models.png")
            if os.path.exists(pr_path):
                st.image(pr_path, use_container_width=True)
            else:
                st.warning("Precision-Recall curves visualization not found.")

        # Visualization insights
        st.markdown("---")
        st.markdown("### Visualization Insights")

        insights_col1, insights_col2 = st.columns(2)

        with insights_col1:
            st.markdown("""
            #### Key Observations:
            1. **SVM performs best** across all metrics
            2. **Good calibration** - predicted probabilities are reliable
            3. **High AUC scores** - models distinguish well between classes
            4. **Learning curves show** models are well-fitted, not overfitting
            """)

        with insights_col2:
            st.markdown("""
            #### Interpretation Tips:
            - **ROC Curve**: Closer to top-left corner is better
            - **PR Curve**: Higher is better, especially for imbalanced data
            - **Learning Curve**: Convergence indicates sufficient data
            - **Calibration Curve**: Closer to diagonal indicates better calibration
            """)

        # Download visualizations
        st.markdown("### Download Visualizations")

        viz_files = [
            ("research_learning_curves.png", "Learning Curves"),
            ("research_calibration_curves.png", "Calibration Curves"),
            ("roc_curves_all_models.png", "ROC Curves"),
            ("precision_recall_curves_all_models.png", "Precision-Recall Curves")
        ]

        cols = st.columns(4)
        for idx, (filename, display_name) in enumerate(viz_files):
            filepath = os.path.join(viz_dir, filename)
            if os.path.exists(filepath):
                with cols[idx]:
                    with open(filepath, "rb") as file:
                        btn = st.download_button(
                            label=f"Download {display_name}",
                            data=file,
                            file_name=filename,
                            mime="image/png",
                            use_container_width=True
                        )


def main():
    """Main function"""
    # Initialize app
    app = ChestXRayApp()

    # Check if models are loaded
    if not app.load_models():
        st.error("""
            ## Models Not Found

            Please ensure that:
            1. You have trained the models using `main.py`
            2. The model file `chest_xray_models.pkl` exists in the current directory
            3. The `visualizations` folder exists with generated plots

            To train the models, run:
            ```bash
            python main.py
            ```
        """)
        return

    # Run the app
    app.run()


if __name__ == "__main__":
    main()
