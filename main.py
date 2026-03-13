#!/usr/bin/env python3
"""Main entry point for Chest X-Ray ML Predictor"""
import os
import sys
import argparse

# Change these imports
from predictor import ChestXRayMLPredictor
from config import config
from utils import print_header, print_error, print_success


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Chest X-Ray Pneumonia Detection with Machine Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train
  python main.py --mode predict --image_path test.jpeg
  python main.py --mode compare
  python main.py --mode train --use_advanced_features --use_pca
        """
    )

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'compare'],
                        help='Mode: train, predict, or compare (default: train)')

    parser.add_argument('--data_dir', type=str, default=config.DEFAULT_DATA_DIR,
                        help=f'Path to dataset directory (default: {config.DEFAULT_DATA_DIR})')

    parser.add_argument('--image_path', type=str,
                        help='Path to image for prediction (required for predict mode)')

    parser.add_argument('--model_file', type=str, default=config.DEFAULT_MODEL_FILE,
                        help=f'Path to saved models (default: {config.DEFAULT_MODEL_FILE})')

    parser.add_argument('--use_advanced_features', action='store_true',
                        help='Use advanced feature extraction')

    parser.add_argument('--use_pca', action='store_true',
                        help='Use PCA for dimensionality reduction')

    parser.add_argument('--pca_components', type=int, default=config.PCA_COMPONENTS,
                        help=f'Number of PCA components (default: {config.PCA_COMPONENTS})')

    parser.add_argument('--no_cv_tuning', action='store_true',
                        help='Disable cross-validation hyperparameter tuning')

    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'predict' and not args.image_path:
        print_error("--image_path is required for prediction mode")
        sys.exit(1)

    # Create predictor
    predictor = ChestXRayMLPredictor()

    # Execute based on mode
    if args.mode == 'train':
        run_train_mode(predictor, args)
    elif args.mode == 'predict':
        run_predict_mode(predictor, args)
    elif args.mode == 'compare':
        run_compare_mode(predictor, args)


def run_train_mode(predictor, args):
    """Run training mode"""
    print_header("CHEST X-RAY ML PREDICTOR - TRAINING MODE")

    # Debug dataset structure
    if not predictor.debug_dataset_structure(args.data_dir):
        print_error("Dataset structure issue detected. Please check the dataset.")
        return

    # Load images
    features, labels = predictor.load_and_preprocess_images(args.data_dir)

    if features is None or len(features) == 0:
        print_error("No images were loaded.")
        return

    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(
        features, labels,
        use_advanced_features=args.use_advanced_features,
        use_pca=args.use_pca,
        n_components=args.pca_components
    )

    # Train models
    cv_tuning = not args.no_cv_tuning

    print_header("Training All Models")
    predictor.train_svm(X_train, y_train, cv_tuning=cv_tuning)
    predictor.train_knn(X_train, y_train, cv_tuning=cv_tuning)
    predictor.train_random_forest(X_train, y_train, cv_tuning=cv_tuning)

    # Evaluate models
    results = predictor.evaluate_all_models(X_train, X_test, y_train, y_test)

    # Save models
    predictor.save_models(args.model_file)

    print_success("Training completed successfully!")


def run_predict_mode(predictor, args):
    """Run prediction mode"""
    print_header("CHEST X-RAY ML PREDICTOR - PREDICTION MODE")

    # Load models
    try:
        predictor.load_models(args.model_file)
    except FileNotFoundError:
        print_error(f"Model file not found: {args.model_file}")
        print("Please train the models first using --mode train")
        return

    # Check if image exists
    if not os.path.exists(args.image_path):
        print_error(f"Image not found: {args.image_path}")
        return

    # Predict with all models
    print_header(f"Predicting: {os.path.basename(args.image_path)}")

    for model_name in predictor.models.keys():
        try:
            predictor.predict_single_image(args.image_path, model_name)
        except Exception as e:
            print_error(f"Error with {model_name}: {e}")


def run_compare_mode(predictor, args):
    """Run comparison mode (evaluate saved models on test data)"""
    print_header("CHEST X-RAY ML PREDICTOR - COMPARISON MODE")

    # Load models
    try:
        predictor.load_models(args.model_file)
    except FileNotFoundError:
        print_error(f"Model file not found: {args.model_file}")
        print("Please train the models first using --mode train")
        return

    # Debug dataset structure
    if not predictor.debug_dataset_structure(args.data_dir):
        print_error("Dataset structure issue detected. Please check the dataset.")
        return

    # Load images
    features, labels = predictor.load_and_preprocess_images(args.data_dir)

    if features is None or len(features) == 0:
        print_error("No images were loaded.")
        return

    # Prepare data (use same preprocessing as training)
    X_train, X_test, y_train, y_test = predictor.prepare_data(
        features, labels,
        use_advanced_features=args.use_advanced_features,
        use_pca=args.use_pca,
        n_components=args.pca_components
    )

    # Evaluate models
    results = predictor.evaluate_all_models(X_train, X_test, y_train, y_test)


def quick_start():
    """Quick start function for testing"""
    print_header("CHEST X-RAY ML PREDICTOR - QUICK START")

    predictor = ChestXRayMLPredictor()

    if os.path.exists(config.DEFAULT_MODEL_FILE):
        print_success("Pre-trained models found!")
        predictor.load_models()

        # Test with sample images if they exist
        sample_images = [
            os.path.join(config.DEFAULT_DATA_DIR, "test/NORMAL/IM-0001-0001.jpeg"),
            os.path.join(config.DEFAULT_DATA_DIR, "test/PNEUMONIA/person1_virus_6.jpeg")
        ]

        for img_path in sample_images:
            if os.path.exists(img_path):
                print(f"\n🔍 Testing with: {os.path.basename(img_path)}")
                for model_name in predictor.models.keys():
                    try:
                        predictor.predict_single_image(img_path, model_name)
                    except Exception as e:
                        print_error(f"Error with {model_name}: {e}")
            else:
                print(f"Sample image not found: {img_path}")
    else:
        print("No pre-trained models found. Starting training...")
        # Run with default parameters
        args = argparse.Namespace(
            mode='train',
            data_dir=config.DEFAULT_DATA_DIR,
            use_advanced_features=False,
            use_pca=True,
            pca_components=50,
            no_cv_tuning=False,
            model_file=config.DEFAULT_MODEL_FILE
        )
        run_train_mode(predictor, args)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        quick_start()