import os
import pickle


def cache_model(cache_file, model):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(model, f)


def load_cached_model(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None


def cache_predictions(cache_file, predictions):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(predictions, f)


def load_cached_predictions(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None


def train_or_load_model(cache_file, train_func, *args, **kwargs):
    cached_model = load_cached_model(cache_file)
    if cached_model is not None:
        print(f"Loaded model from cache: {cache_file}")
        return cached_model
    else:
        print(f"Training new model...")
        model = train_func(*args, **kwargs)
        cache_model(cache_file, model)
        print(f"Model saved to cache: {cache_file}")
        return model


def predict_or_load(cache_file, pipeline, data):
    cached_preds = load_cached_predictions(cache_file)
    if cached_preds is not None:
        print(f"Loaded predictions from cache: {cache_file}")
        return cached_preds
    else:
        print(f"Generating predictions...")
        predictions = pipeline.predict(data)
        cache_predictions(cache_file, predictions)
        print(f"Predictions saved to cache: {cache_file}")
        return predictions
