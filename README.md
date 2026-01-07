# Airline Satisfaction Prediction - Python Package

This is a Python package for predicting airline passenger satisfaction using machine learning. It implements three different classification algorithms: K-Nearest Neighbors (KNN), Logistic Regression, and Random Forest.

## Project Structure

```
airline_satisfaction/
├── __init__.py              # Package initialization
├── main.py                  # Main execution script
├── data/
│   ├── __init__.py
│   └── loader.py           # Data loading and preprocessing
├── models/
│   ├── __init__.py
│   ├── preprocessors.py    # Preprocessing pipelines
│   ├── trainers.py         # Model training logic
│   └── evaluator.py        # Model evaluation
├── utils/
│   ├── __init__.py
│   └── cache.py            # Caching utilities
└── visualization/
    ├── __init__.py
    └── plots.py            # Visualization functions
```

## Installation

### Option 1: Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/airline-satisfaction-prediction.git
cd airline-satisfaction-prediction

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Option 2: Install dependencies only

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

Run the complete pipeline:

```bash
python airline_satisfaction/main.py
```

Command line options:

```bash
python airline_satisfaction/main.py \
    --data-path /path/to/data \
    --cache-dir cache \
    --output-dir graphs \
    --no-cache
```

Arguments:
- `--data-path`: Path to data directory (default: downloads from Kaggle)
- `--cache-dir`: Directory for caching models and predictions (default: 'cache')
- `--output-dir`: Directory for saving visualizations (default: 'graphs')
- `--no-cache`: Disable caching to retrain models from scratch

### Python API

Use the package components in your own scripts:

```python
from airline_satisfaction import DataLoader, ModelTrainer, ModelEvaluator, Visualizer

# Load and prepare data
loader = DataLoader()
df, df_test = loader.load_data()
loader.reclass_variables()
X, y = loader.prepare_features_target()

# Split data
X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val = loader.split_data(X, y)

# Create sample for faster training
X_sample, y_sample = loader.create_sample(X_train, y_train)

# Get column groups
cat_cols, num_cols, rating_cols, _ = loader.get_column_groups()

# Train models
trainer = ModelTrainer(cat_cols, num_cols, rating_cols)

# Train Random Forest
fits_rf = trainer.train_random_forest(X_sample, y_sample)
print(f"Best params: {fits_rf.best_params_}")
print(f"Best score: {fits_rf.best_score_}")

# Build pipeline with best params
rf_pipeline = trainer.get_random_forest_pipeline()
rf_pipeline.set_params(**fits_rf.best_params_)
rf_pipeline.fit(X_train, y_train)

# Evaluate
evaluator = ModelEvaluator()
y_pred = rf_pipeline.predict(X_val)
results = evaluator.evaluate_model('Random Forest', y_val, y_pred)
evaluator.print_results('Random Forest')

# Visualize
visualizer = Visualizer(output_dir='graphs')
visualizer.plot_confusion_matrix(y_val, y_pred, loader.label_encoder.classes_)
```

## Performance

On the full airline satisfaction dataset (100K+ samples):

| Model | Validation Accuracy | Test Accuracy |
|-------|-------------------|---------------|
| KNN | 92.3% | - |
| Logistic Regression | 93.2% | - |
| Random Forest | 96.1% | 96.5% |

## Requirements

- Python >= 3.8
- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- scikit-learn >= 1.2.0
- xgboost >= 1.7.0
- kagglehub >= 0.1.0

## Output

The pipeline generates:
- Cached models in `cache/` directory
- Visualization plots in `graphs/` directory:
  - `model_accuracy_comparison.png`
  - `confusion_matrix.png`
  - `roc_curve.png`
  - `feature_importance.png`

## License

MIT License