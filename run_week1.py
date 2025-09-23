from src.data_loader import CICIDSDataLoader
from src.preprocessor import DataPreprocessor
from src.train_model import train_baseline


def main():
    print("=" * 50)
    print("WEEK 1: BASELINE MODEL PIPELINE")
    print("=" * 50)

    # Step 1: Load data
    print("\n📊 Loading CICIDS2017 data...")
    loader = CICIDSDataLoader()
    df = loader.load_friday_data(sample_size=100000)  # Use 100k samples for speed

    # Step 2: Preprocess
    print("\n🔧 Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.process_data(df)

    # Step 3: Train
    print("\n🎯 Training baseline model...")
    model, accuracy = train_baseline()

    if accuracy > 0.95:
        print("\n✅ Week 1 Complete! Baseline achieved >95% accuracy")
    else:
        print(f"\n⚠️ Accuracy {accuracy:.2%} - may need tuning")


if __name__ == "__main__":
    main()
