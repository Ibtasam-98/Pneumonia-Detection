"""Main entry point"""
import argparse
import os
import sys
import traceback
from config import Config
from data_loader import DataLoader
from model.svm_model import SVMModel
from model.knn_model import KNNModel
from model.random_forest_model import RandomForestModel
from evaluator import ModelEvaluator
from visualizer import Visualizer
from predictor import Predictor
from utils import save_models, load_models


class ChestXRayMLPredictor:
    def __init__(self):
        self.config = Config()
        self.config.setup_directories()
        self.data_loader = DataLoader(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.visualizer = Visualizer(self.config)
        self.predictor = Predictor(self.config)
        self.models = {}

    def train(self, args):
        try:
            if not self.data_loader.debug_dataset_structure(args.data_dir):
                print("Dataset structure issue detected.")
                return

            features, labels = self.data_loader.load_and_preprocess_images(args.data_dir)

            if features is None or len(features) == 0:
                print("No images were loaded.")
                return

            X_train, X_test, y_train, y_test = self.data_loader.preprocess_data(
                use_advanced_features=args.use_advanced_features,
                use_pca=args.use_pca,
                n_components=args.pca_components
            )

            # Train models
            print("\n" + "=" * 100)
            print("TRAINING MODELS")
            print("=" * 100)

            svm_model = SVMModel(random_state=self.config.RANDOM_STATE)
            svm_model.train(X_train, y_train, cv_tuning=True)
            self.models['svm'] = svm_model.get_model()
            self.evaluator.training_times['svm'] = svm_model.get_training_time()

            knn_model = KNNModel(random_state=self.config.RANDOM_STATE)
            knn_model.train(X_train, y_train, cv_tuning=True)
            self.models['knn'] = knn_model.get_model()
            self.evaluator.training_times['knn'] = knn_model.get_training_time()

            rf_model = RandomForestModel(random_state=self.config.RANDOM_STATE)
            rf_model.train(X_train, y_train, cv_tuning=True)
            self.models['random_forest'] = rf_model.get_model()
            self.evaluator.training_times['random_forest'] = rf_model.get_training_time()

            # Evaluate models
            results = self.evaluator.evaluate_all_models(
                X_train, X_test, y_train, y_test,
                self.models, self.config.CLASS_NAMES
            )

            # Create visualizations
            self.visualizer.create_consolidated_metrics_visualization(
                self.evaluator.get_metrics_history(),
                self.config.CLASS_NAMES
            )

            # Print summary table
            self.visualizer.print_summary_table(self.evaluator.get_metrics_history())

            # Save models
            save_data = save_models(
                self.models,
                self.data_loader.scaler,
                self.data_loader.pca,
                self.config.IMG_HEIGHT,
                self.config.IMG_WIDTH,
                self.config.CLASS_NAMES,
                self.evaluator.get_metrics_history(),
                self.evaluator.training_times,
                self.evaluator.inference_times,
                args.model_file
            )

            print("\n✅ Training completed successfully!")

        except Exception as e:
            print(f"❌ Error during training: {e}")
            traceback.print_exc()

    def predict(self, args):
        """Predict mode"""
        if not args.image_path:
            print("❌ Error: --image_path is required for prediction mode")
            return

        try:
            saved_data = load_models(args.model_file)

            print("\n" + "=" * 60)
            print("MAKING PREDICTIONS")
            print("=" * 60)

            # Predict with all models
            for model_name, model in saved_data['models'].items():
                try:
                    print(f"\n📊 Using {model_name.upper()} model:")
                    self.predictor.predict_single_image(
                        args.image_path,
                        model,
                        saved_data['scaler'],
                        saved_data['pca'],
                        saved_data['class_names']
                    )
                except Exception as e:
                    print(f"❌ Error with {model_name}: {e}")

        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            traceback.print_exc()

    def compare(self, args):
        """Compare mode"""
        try:
            saved_data = load_models(args.model_file)
            self.models = saved_data['models']
            self.evaluator.metrics_history = saved_data.get('metrics_history', {})
            self.evaluator.training_times = saved_data.get('training_times', {})
            self.evaluator.inference_times = saved_data.get('inference_times', {})

            if not self.data_loader.debug_dataset_structure(args.data_dir):
                print("❌ Dataset structure issue detected.")
                return

            features, labels = self.data_loader.load_and_preprocess_images(args.data_dir)

            if features is None or len(features) == 0:
                print("❌ No images were loaded.")
                return

            X_train, X_test, y_train, y_test = self.data_loader.preprocess_data(
                use_advanced_features=args.use_advanced_features,
                use_pca=args.use_pca,
                n_components=args.pca_components
            )

            # Re-evaluate all loaded models
            results = self.evaluator.evaluate_all_models(
                X_train, X_test, y_train, y_test,
                self.models, self.config.CLASS_NAMES
            )

            # Create visualizations
            self.visualizer.create_consolidated_metrics_visualization(
                self.evaluator.get_metrics_history(),
                self.config.CLASS_NAMES
            )

            # Print summary table
            self.visualizer.print_summary_table(self.evaluator.get_metrics_history())

        except Exception as e:
            print(f"❌ Error during comparison: {e}")
            traceback.print_exc()


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

    predictor = ChestXRayMLPredictor()

    if args.mode == 'train':
        predictor.train(args)
    elif args.mode == 'predict':
        predictor.predict(args)
    elif args.mode == 'compare':
        predictor.compare(args)


def quick_start():
    """Quick start function"""
    print("=" * 60)
    print("QUICK START: CHEST X-RAY PNEUMONIA DETECTION WITH ML")
    print("=" * 60)

    predictor = ChestXRayMLPredictor()

    if os.path.exists('chest_xray_models.pkl'):
        print("\n✅ Pre-trained models found!")
        print("\nOptions:")
        print("1. Make predictions on sample images")
        print("2. Train new models")
        print("3. Compare models")

        choice = input("\nEnter your choice (1/2/3): ").strip()

        if choice == '1':
            sample_images = [
                "dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg",
                "dataset/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
            ]

            for img_path in sample_images:
                if os.path.exists(img_path):
                    print(f"\n🔍 Testing with: {os.path.basename(img_path)}")
                    try:
                        saved_data = load_models('chest_xray_models.pkl')
                        for model_name, model in saved_data['models'].items():
                            try:
                                predictor.predict_single_image(
                                    img_path,
                                    model,
                                    saved_data['scaler'],
                                    saved_data['pca'],
                                    saved_data['class_names']
                                )
                            except Exception as e:
                                print(f"Error with {model_name}: {e}")
                    except Exception as e:
                        print(f"Error loading models: {e}")
                else:
                    print(f"Sample image not found: {img_path}")
                    print("Please ensure the dataset is in the correct location.")
        elif choice == '2':
            predictor.train(argparse.Namespace(
                mode='train',
                data_dir='dataset/chest_xray',
                use_advanced_features=False,
                use_pca=True,
                pca_components=50,
                model_file='chest_xray_models.pkl'
            ))
        elif choice == '3':
            predictor.compare(argparse.Namespace(
                mode='compare',
                data_dir='dataset/chest_xray',
                use_advanced_features=False,
                use_pca=True,
                pca_components=50,
                model_file='chest_xray_models.pkl'
            ))
        else:
            print("Invalid choice. Exiting.")
    else:
        print("\n❌ No pre-trained models found.")
        print("Starting training process...")
        predictor.train(argparse.Namespace(
            mode='train',
            data_dir='dataset/chest_xray',
            use_advanced_features=False,
            use_pca=True,
            pca_components=50,
            model_file='chest_xray_models.pkl'
        ))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        quick_start()