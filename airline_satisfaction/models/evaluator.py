from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, classification_report
)


class ModelEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate_model(self, model_name, y_true, y_pred, y_prob=None):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        if y_prob is not None:
            roc_auc = roc_auc_score(y_true, y_prob)
            results['roc_auc'] = roc_auc

        self.results[model_name] = results
        return results

    def print_results(self, model_name):
        if model_name in self.results:
            print(f"\n{model_name} Results:")
            print("-" * 50)
            for metric, value in self.results[model_name].items():
                print(f"{metric}: {value:.4f}")
        else:
            print(f"No results found for {model_name}")

    def print_all_results(self):
        for model_name in self.results:
            self.print_results(model_name)

    def get_classification_report(self, y_true, y_pred, target_names=None):
        return classification_report(y_true, y_pred, target_names=target_names)

    def get_best_model(self, metric='accuracy'):
        if not self.results:
            return None

        best_model = max(self.results.items(), key=lambda x: x[1].get(metric, 0))
        return best_model[0], best_model[1][metric]
