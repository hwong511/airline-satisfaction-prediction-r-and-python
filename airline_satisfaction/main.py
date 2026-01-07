import os
import argparse
from data.loader import DataLoader
from models.trainers import ModelTrainer
from models.evaluator import ModelEvaluator
from visualization.plots import Visualizer
from utils.cache import train_or_load_model, predict_or_load


def main(data_path=None, cache_dir='cache', output_dir='graphs', use_cache=True):
    print("=" * 70)
    print("Airline Passenger Satisfaction Prediction")
    print("=" * 70)

    # 1. Load Data
    print("\n[1/7] Loading data...")
    loader = DataLoader(data_path=data_path)
    df, df_test = loader.load_data()
    print(f"Training data shape: {df.shape}")

    # 2. Prepare Data
    print("\n[2/7] Preparing data...")
    loader.reclass_variables()
    X, y = loader.prepare_features_target()
    cat_cols, num_cols, rating_cols, _ = loader.get_column_groups()

    # 3. Split Data
    print("\n[3/7] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val = loader.split_data(X, y)
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    # Create sample for faster training
    X_sample, y_sample = loader.create_sample(X_train, y_train, sample_size=0.4)
    print(f"Sample size for hyperparameter tuning: {len(X_sample)}")

    # 4. Train Models
    print("\n[4/7] Training models...")
    os.makedirs(cache_dir, exist_ok=True)

    trainer = ModelTrainer(cat_cols, num_cols, rating_cols, cache_dir)

    # KNN
    print("\n  Training K-Nearest Neighbors...")
    if use_cache:
        knn_cache = os.path.join(cache_dir, "fits_knn.pkl")
        fits_knn = train_or_load_model(knn_cache, trainer.train_knn, X_sample, y_sample)
    else:
        fits_knn = trainer.train_knn(X_sample, y_sample)
    print(f"  Best KNN params: {fits_knn.best_params_}")
    print(f"  Best KNN CV score: {fits_knn.best_score_:.4f}")

    # Logistic Regression
    print("\n  Training Logistic Regression...")
    if use_cache:
        log_cache = os.path.join(cache_dir, "fits_log.pkl")
        fits_log = train_or_load_model(log_cache, trainer.train_logistic, X_sample, y_sample)
    else:
        fits_log = trainer.train_logistic(X_sample, y_sample)
    print(f"  Best LogReg params: {fits_log.best_params_}")
    print(f"  Best LogReg CV score: {fits_log.best_score_:.4f}")

    # Random Forest
    print("\n  Training Random Forest...")
    if use_cache:
        rf_cache = os.path.join(cache_dir, "fits_rf.pkl")
        fits_rf = train_or_load_model(rf_cache, trainer.train_random_forest, X_sample, y_sample)
    else:
        fits_rf = trainer.train_random_forest(X_sample, y_sample)
    print(f"  Best RF params: {fits_rf.best_params_}")
    print(f"  Best RF CV score: {fits_rf.best_score_:.4f}")

    # 5. Evaluate on Validation Set
    print("\n[5/7] Evaluating models on validation set...")
    evaluator = ModelEvaluator()

    # KNN
    knn_pipeline = trainer.get_knn_pipeline()
    knn_pipeline.set_params(**fits_knn.best_params_)
    knn_pipeline.fit(X_train, y_train)
    if use_cache:
        y_pred_knn = predict_or_load(os.path.join(cache_dir, "pred_knn.pkl"), knn_pipeline, X_val)
    else:
        y_pred_knn = knn_pipeline.predict(X_val)
    evaluator.evaluate_model('KNN', y_val, y_pred_knn)

    # Logistic Regression
    log_pipeline = trainer.get_logistic_pipeline()
    log_pipeline.set_params(**fits_log.best_params_)
    log_pipeline.fit(X_train, y_train)
    if use_cache:
        y_pred_log = predict_or_load(os.path.join(cache_dir, "pred_log.pkl"), log_pipeline, X_val)
    else:
        y_pred_log = log_pipeline.predict(X_val)
    evaluator.evaluate_model('Logistic Regression', y_val, y_pred_log)

    # Random Forest
    rf_pipeline = trainer.get_random_forest_pipeline()
    rf_pipeline.set_params(**fits_rf.best_params_)
    rf_pipeline.fit(X_train, y_train)
    if use_cache:
        y_pred_rf = predict_or_load(os.path.join(cache_dir, "pred_rf.pkl"), rf_pipeline, X_val)
    else:
        y_pred_rf = rf_pipeline.predict(X_val)
    evaluator.evaluate_model('Random Forest', y_val, y_pred_rf)

    evaluator.print_all_results()

    # 6. Evaluate Best Model on Test Set
    print("\n[6/7] Evaluating best model on test set...")
    best_model_name, best_score = evaluator.get_best_model()
    print(f"Best model: {best_model_name} (validation accuracy: {best_score:.4f})")

    # Retrain on train+val
    rf_pipeline.fit(X_train_val, y_train_val)
    if use_cache:
        y_test_pred = predict_or_load(os.path.join(cache_dir, "pred_test.pkl"), rf_pipeline, X_test)
    else:
        y_test_pred = rf_pipeline.predict(X_test)
    y_test_prob = rf_pipeline.predict_proba(X_test)[:, 1]

    test_results = evaluator.evaluate_model('Random Forest (Test)', y_test, y_test_pred, y_test_prob)
    evaluator.print_results('Random Forest (Test)')

    # 7. Visualizations
    print("\n[7/7] Generating visualizations...")
    visualizer = Visualizer(output_dir=output_dir)

    # Model comparison
    model_names = ['KNN', 'Logistic Regression', 'Random Forest']
    accuracies = [
        evaluator.results['KNN']['accuracy'],
        evaluator.results['Logistic Regression']['accuracy'],
        evaluator.results['Random Forest']['accuracy']
    ]
    visualizer.plot_model_comparison(model_names, accuracies)

    # Confusion matrix
    class_names = loader.label_encoder.classes_
    visualizer.plot_confusion_matrix(y_test, y_test_pred, class_names,
                                     title='Random Forest - Test Set Confusion Matrix')

    # ROC curve
    visualizer.plot_roc_curve(y_test, y_test_prob, title='Random Forest - Test Set ROC Curve')

    # Feature importance
    preprocessor = rf_pipeline.named_steps['preprocessor']
    best_rf_model = rf_pipeline.named_steps['classifier']

    cat_names = list(preprocessor.named_transformers_['cat']
                    .named_steps['onehot'].get_feature_names_out(cat_cols))
    rating_names = list(preprocessor.named_transformers_['rating']
                       .named_steps['onehot'].get_feature_names_out(rating_cols))
    num_names = num_cols
    feature_names = cat_names + rating_names + num_names

    importances = best_rf_model.feature_importances_
    top_features = visualizer.plot_feature_importance(feature_names, importances, top_n=10)
    print("\nTop 10 Features:")
    print(top_features)

    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Airline Passenger Satisfaction Prediction')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data directory (default: download from Kaggle)')
    parser.add_argument('--cache-dir', type=str, default='cache',
                       help='Directory for caching models and predictions')
    parser.add_argument('--output-dir', type=str, default='graphs',
                       help='Directory for saving visualizations')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching')

    args = parser.parse_args()

    main(data_path=args.data_path,
         cache_dir=args.cache_dir,
         output_dir=args.output_dir,
         use_cache=not args.no_cache)
