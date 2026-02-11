import numpy as np
from predictive_matrix import PredictiveMatrix


class PredictiveSystem:
    """
    Multi-feature predictive models.
    Each feature type has its own 3D tensor for prediction.
    Handles forward model prediction and Hebbian learning.
    """

    def __init__(self):
        # Predictive matrices for each feature type
        self.predictors = {
            "brightness": PredictiveMatrix(
                feature_dim=16,  # 4×4 grid
                action_dim=2,  # [Δ_head, Δ_eye]
            )
        }

        # Later you can add:
        # 'edges': PredictiveMatrix(feature_dim=64, action_dim=5),
        # 'motion': PredictiveMatrix(feature_dim=32, action_dim=5),
        # 'corners': PredictiveMatrix(feature_dim=36, action_dim=5),

    def predict(self, features, action):
        """
        Predict next features given current features + action.

        Args:
            features: dict {'brightness': np.array([16]), ...}
            action: np.array([2]) - one-hot encoded action

        Returns:
            predictions: dict {'brightness': np.array([16]), ...}
        """
        predictions = {}

        for feature_name, feature_vec in features.items():
            if feature_name in self.predictors:
                predictions[feature_name] = self.predictors[feature_name].predict(
                    feature_vec, action
                )

        return predictions

    def compute_error(self, predictions, actual_features):
        """
        Compute prediction error (MSE between predicted and actual).

        Args:
            predictions: dict of predicted feature vectors
            actual_features: dict of actual observed feature vectors

        Returns:
            avg_error: float (average across all features)
            error_breakdown: dict {feature_name: error_value}
        """
        total_error = 0.0
        error_breakdown = {}

        for feature_name in predictions.keys():
            pred = predictions[feature_name]
            actual = actual_features[feature_name]

            # Mean squared error
            error = np.mean((pred - actual) ** 2)
            error_breakdown[feature_name] = error
            total_error += error

        # Average across all feature types
        avg_error = total_error / len(predictions) if len(predictions) > 0 else 0.0

        return avg_error, error_breakdown

    def consolidate(self, sequences_by_feature, learning_rate):
        """
        Learn from sequences during sleep.
        Updates weights using Hebbian learning.

        Args:
            sequences_by_feature: dict from Hippocampus
                {'brightness': [seq1, seq2, ...], 'edges': [...], ...}
                Each sequence: {'feature_t', 'action', 'feature_t1'}

        Returns:
            converged: bool (True if all predictors have low error)
        """
        all_converged = True

        for feature_name, sequences in sequences_by_feature.items():
            if feature_name in self.predictors:
                # Update this predictor with its sequences
                avg_error = self.predictors[feature_name].update(
                    sequences, learning_rate
                )

                print(f"    {feature_name}: error={avg_error:.4f}")

                # Check convergence
                if avg_error > 0.01:  # convergence threshold
                    all_converged = False

        return all_converged

    def add_feature_predictor(self, feature_name, feature_dim, action_dim=5):
        """
        Dynamically add a new feature predictor.
        Useful for developmental progression.

        Args:
            feature_name: str (e.g., 'edges')
            feature_dim: int (e.g., 64 for 8×8 grid)
            action_dim: int (default 2)
        """
        if feature_name not in self.predictors:
            self.predictors[feature_name] = PredictiveMatrix(
                feature_dim=feature_dim, action_dim=action_dim
            )
            print(f"Added predictor for '{feature_name}' (dim={feature_dim})")
