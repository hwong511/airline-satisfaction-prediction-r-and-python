from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .preprocessors import (
    get_knn_preprocessor,
    get_logistic_preprocessor,
    get_random_forest_preprocessor
)


class ModelTrainer:
    def __init__(self, cat_cols, num_cols, rating_cols, cache_dir=None):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.rating_cols = rating_cols
        self.cache_dir = cache_dir
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)

    def get_knn_pipeline(self):
        preprocessor = get_knn_preprocessor(self.cat_cols, self.rating_cols, self.num_cols)
        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier())
        ])

    def get_knn_param_grid(self):
        return {
            'classifier__n_neighbors': range(1, 51, 2),
            'classifier__weights': ['uniform', 'distance']
        }

    def get_logistic_pipeline(self):
        preprocessor = get_logistic_preprocessor(self.cat_cols, self.rating_cols, self.num_cols)
        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
        ])

    def get_logistic_param_grid(self):
        return {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2', 'elasticnet'],
            'classifier__l1_ratio': [0.1, 0.5, 0.9]
        }

    def get_random_forest_pipeline(self):
        preprocessor = get_random_forest_preprocessor(self.cat_cols, self.rating_cols, self.num_cols)
        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier())
        ])

    def get_random_forest_param_grid(self):
        return {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_features': [5, 10, 20, 'sqrt'],
            'classifier__min_samples_split': [2, 5, 10, 20],
            'classifier__min_samples_leaf': [1, 2, 5]
        }

    def train_model(self, pipeline, param_grid, X_sample, y_sample, n_jobs=2, verbose=2):
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=self.cv,
            scoring='accuracy', n_jobs=n_jobs, verbose=verbose
        )
        grid_search.fit(X_sample, y_sample)
        return grid_search

    def train_knn(self, X_sample, y_sample, n_jobs=2, verbose=2):
        pipeline = self.get_knn_pipeline()
        param_grid = self.get_knn_param_grid()
        return self.train_model(pipeline, param_grid, X_sample, y_sample, n_jobs, verbose)

    def train_logistic(self, X_sample, y_sample, n_jobs=2, verbose=2):
        pipeline = self.get_logistic_pipeline()
        param_grid = self.get_logistic_param_grid()
        return self.train_model(pipeline, param_grid, X_sample, y_sample, n_jobs, verbose)

    def train_random_forest(self, X_sample, y_sample, n_jobs=2, verbose=2):
        pipeline = self.get_random_forest_pipeline()
        param_grid = self.get_random_forest_param_grid()
        return self.train_model(pipeline, param_grid, X_sample, y_sample, n_jobs, verbose)
