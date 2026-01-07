"""
Airline Passenger Satisfaction Prediction Package

A machine learning package for predicting airline passenger satisfaction
using various classification algorithms including KNN, Logistic Regression,
and Random Forest.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data.loader import DataLoader
from .models.trainers import ModelTrainer
from .models.evaluator import ModelEvaluator
from .visualization.plots import Visualizer

__all__ = [
    'DataLoader',
    'ModelTrainer',
    'ModelEvaluator',
    'Visualizer',
]
