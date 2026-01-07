import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc


class Visualizer:
    def __init__(self, output_dir='graphs'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def plot_model_comparison(self, model_names, accuracies, save=True):
        plt.figure(figsize=(8, 6))
        sns.barplot(x=model_names, y=accuracies)
        plt.title('Comparison of Model Accuracy on Validation Set')
        plt.ylabel('Accuracy %')
        plt.ylim(0.8, 1.0)
        if save:
            plt.savefig(os.path.join(self.output_dir, 'model_accuracy_comparison.png'))
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, class_names, title='Confusion Matrix', save=True):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        if save:
            plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.show()

    def plot_roc_curve(self, y_true, y_prob, title='ROC Curve', save=True):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        if save:
            plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'))
        plt.show()

    def plot_feature_importance(self, feature_names, importances, top_n=10, save=True):
        feature_importances = pd.Series(importances, index=feature_names)
        sorted_importances = feature_importances.sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(8, 6))
        sns.barplot(x=sorted_importances.values, y=sorted_importances.index)
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        if save:
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
        plt.show()

        return sorted_importances
