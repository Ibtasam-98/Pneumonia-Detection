"""Utility functions"""
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


def save_models(models, scaler, pca, img_height, img_width, class_names,
                metrics_history, training_times, inference_times, filename='chest_xray_models.pkl'):
    """Save all models and preprocessors"""
    save_data = {
        'models': models,
        'scaler': scaler,
        'pca': pca,
        'img_height': img_height,
        'img_width': img_width,
        'class_names': class_names,
        'metrics_history': metrics_history,
        'training_times': training_times,
        'inference_times': inference_times
    }
    joblib.dump(save_data, filename)
    print(f"✅ Models and metrics saved to {filename}")
    return save_data


def load_models(filename='chest_xray_models.pkl'):
    """Load models and preprocessors"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file not found: {filename}")

    save_data = joblib.load(filename)

    print(f"✅ Models loaded from {filename}")

    # Print available models
    from tabulate import tabulate
    table_data = [[model_name, type(model).__name__] for model_name, model in save_data['models'].items()]
    print(tabulate(table_data, headers=["Model Name", "Type"], tablefmt="grid"))

    return save_data