# Airline Passenger Satisfaction Prediction: R vs Python Implementation

## Project Overview

This project implements and compares machine learning approaches for predicting airline passenger satisfaction using both R and Python. It demonstrates how programming language choices and dataset sizes can affect model performance while maintaining consistent methodology across both implementations.

## Dataset

* **Source**: Airline Passenger Satisfaction Dataset (via Kaggle)
* **Target Variable**: Binary satisfaction rating (satisfied vs. neutral/dissatisfied)
* **Features**: \~20 predictor variables including:

  * Customer demographics (age, gender, customer type)
  * Service quality ratings (WiFi, food, cleanliness, etc.)
  * Flight characteristics (distance, delays)
  * Travel details (class, purpose)

**Dataset Sizes**:

* **R Implementation**: 600 observations (subset used for coursework)
* **Python Implementation**: 100,000+ observations (full dataset)

## Methodology

### Model Selection Workflow

1. **Preprocessing**: Variable reclassification, missing data imputation, and feature engineering
2. **Data Splitting**: Nested train/validation/test approach with stratified sampling
3. **Algorithms Compared**: K-Nearest Neighbors, Logistic Regression, Random Forest
4. **Hyperparameter Tuning**: Grid search using cross-validation
5. **Final Evaluation**: Test set performance assessment

### Methodological Highlights

* **Metric Selection**: auROC in R (for class imbalance); Accuracy in Python (due to class balance)
* **Resampling Strategies**: Bootstrap in R vs. Stratified K-Fold Cross-Validation in Python
* **Efficiency Considerations**: Strategic downsampling for hyperparameter tuning in Python

## Results Summary

| Implementation | Best Algorithm | Performance                 | Dataset Size |
| -------------- | -------------- | --------------------------- | ------------ |
| **R**          | Random Forest  | 96.3% auROC, 90.7% Accuracy | 600          |
| **Python**     | Random Forest  | 96.5% Accuracy              | 100K+        |

### Additional Model Results (Python)

* **K-Nearest Neighbors**: 92.3% accuracy
* **Logistic Regression**: 93.2% accuracy
* **Random Forest**: 96.5% accuracy

## Getting Started

### Prerequisites

#### R Packages

```r
library(tidyverse)
library(tidymodels)
library(xfun)
```

#### Python Packages

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost kagglehub
```

### Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/airline-satisfaction-prediction.git
cd airline-satisfaction-prediction
```

2. **Install R Dependencies**

```r
install.packages(c("tidyverse", "tidymodels", "xfun"))
```

3. **Install Python Dependencies**

```bash
pip install -r requirements.txt
```

### Running the Analysis

* **R**: Run `r_implementation.qmd` using RStudio or via `rmarkdown::render()`
* **Python**: Open and run `python_implementation.ipynb` in Jupyter Notebook

## Repository Structure

```
├── README.md
├── r_implementation.qmd                 # R implementation
├── python_implementation.ipynb          # Python implementation
├── data/
│   ├── airline_passenger_satisfaction.csv
│   └── processed/
├── cache/                      # Cached model results
├── graphs/                     # Generated visualizations
└── requirements.txt            # Python dependency list
```

## Visualizations

The project generates several plots and graphs:

* Bar charts comparing algorithm performance
* Confusion matrices to visualize classification results
* ROC curves for evaluating model discriminative ability
* Feature importance visualizations to identify satisfaction drivers

## Insights

### Technical Highlights

* Random Forest consistently outperforms simpler models
* Larger datasets improve model reliability and stability
* Tailored preprocessing is essential for optimal performance
* Ensemble models effectively capture nonlinear relationships

### Business Implications

* In-flight service quality is the primary driver of satisfaction
* Delays significantly impact perceived experience
* Class and loyalty status affect satisfaction ratings
* Predictive models can flag dissatisfied passengers early for intervention

## Implementation Differences

| Feature                 | R Implementation     | Python Implementation  |
| ----------------------- | -------------------- | ---------------------- |
| Dataset Size            | 600                  | 100,000+               |
| Evaluation Metric       | auROC                | Accuracy               |
| Resampling Strategy     | Bootstrap            | Stratified K-Fold      |
| Preprocessing Framework | tidymodels `recipes` | scikit-learn pipelines |
| Caching Method          | `xfun::cache_rds`    | `pickle`               |
| Parallel Processing     | `doParallel`         | `n_jobs` parameter     |

## Learning Outcomes

This project demonstrates:

* Cross-language implementation of ML pipelines
* Scalable and reproducible preprocessing workflows
* Effective model selection and evaluation strategies
* Communication of technical insights and business relevance

## Contributing

Contributions are welcome. You may suggest or add:

* Additional machine learning models
* Improved data visualizations
* Alternative preprocessing techniques
* Performance optimization strategies

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

* **Author**: Ho Wong
* **Email**: howong112@outlook.com
* **LinkedIn**: www.linkedin.com/in/ho-wong-1oo012

## Acknowledgments

* Original R implementation created for PSYC752 at University of Wisconsin-Madison
* Dataset sourced from Kaggle’s Airline Passenger Satisfaction dataset
* Inspired by the goal of comparing R and Python ML workflows for reproducibility and performance
