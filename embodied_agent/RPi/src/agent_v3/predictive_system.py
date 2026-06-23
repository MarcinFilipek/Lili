import numpy as np
from predictive_matrix import PredictiveMatrix


class PredictiveSystem:
    """
    Forward model: uczy się jak akcja (Δpan, Δtilt) transformuje obraz.

    Dwa niezależne predyktory:
        brightness: input=[121 + 2] → ReLU(64) → 121
        edges:      input=[121 + 2] → ReLU(64) → 121
    """

    def __init__(self):
        self.predictors = {
            "brightness": PredictiveMatrix(feature_dim=121, action_dim=2, hidden_dim=64),
            "edges":      PredictiveMatrix(feature_dim=121, action_dim=2, hidden_dim=64),
        }

    def predict(self, features: dict, action: np.ndarray) -> dict:
        predictions = {}
        for name, vec in features.items():
            if name in self.predictors:
                predictions[name] = self.predictors[name].predict(vec, action)
        return predictions

    def compute_error(self, predictions: dict, actual: dict):
        total = 0.0
        breakdown = {}
        for name, pred in predictions.items():
            err = float(np.mean((pred - actual[name]) ** 2))
            breakdown[name] = err
            total += err
        avg = total / len(predictions) if predictions else 0.0
        return avg, breakdown

    def online_step(
        self,
        prev_features: dict,
        action:        np.ndarray,
        features:      dict,
        lr:            float = 0.001,
    ) -> dict:
        errors = {}
        for name, predictor in self.predictors.items():
            if name in prev_features and name in features:
                errors[name] = predictor.online_step(
                    prev_features[name], action, features[name], lr=lr
                )
        return errors

    def consolidate(
        self,
        sequences_by_feature: dict,
        learning_rate:        float = 0.01,
        epochs:               int   = 5,
    ) -> bool:
        all_converged = True
        for name, sequences in sequences_by_feature.items():
            if name in self.predictors:
                err = self.predictors[name].update(
                    sequences, learning_rate=learning_rate, epochs=epochs
                )
                print(f"    [{name}] batch_err={err:.4f}  "
                      f"stats={self.predictors[name].get_stats()}")
                if err > 0.01:
                    all_converged = False
        return all_converged