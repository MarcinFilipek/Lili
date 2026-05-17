import numpy as np


class PredictiveMatrix:
    """
    3D tensor for learning feature→action→next_feature mappings.
    Uses Hebbian learning to strengthen associations.
    """

    def __init__(self, feature_dim, action_dim):
        """
        Args:
            feature_dim: int (size of feature vector)
            action_dim: int (number of possible actions)
        """
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        # 3D weight tensor: W[i, j, k]
        # i: current feature dimension
        # j: action dimension
        # k: next feature dimension
        self.W = np.random.randn(feature_dim, action_dim, feature_dim) * 0.01

        # Statistics (optional, for monitoring)
        self.total_updates = 0
        self.recent_errors = []

    def predict(self, feature_vec, action_vec):
        """
        Forward model: predict next feature given current feature + action.

        Uses Einstein summation for 3D tensor contraction:
        next_feature[k] = Σ_i Σ_j W[i,j,k] * feature[i] * action[j]

        Args:
            feature_vec: np.array([feature_dim])
            action_vec: np.array([action_dim]) - one-hot encoded

        Returns:
            prediction: np.array([feature_dim])
        """
        # Einstein notation: 'ijk,i,j->k'
        # W[i,j,k] * feature[i] * action[j] → next_feature[k]
        prediction = np.einsum("ijk,i,j->k", self.W, feature_vec, action_vec)

        return prediction

    def update(self, sequences, learning_rate=0.01):
        if len(sequences) == 0:
            return 0.0

        total_error = 0.0

        for seq in sequences:
            f_t  = seq["feature_t"]
            a    = seq["action"]
            f_t1 = seq["feature_t1"]

            # Aktualna predykcja
            prediction = self.predict(f_t, a)

            # Błąd predykcji
            error_vec = prediction - f_t1
            mse = np.mean(error_vec ** 2)
            total_error += mse

            # Hebbian z korektą błędu (delta rule):
            # zamiast wzmacniać f_t1 bezwarunkowo,
            # wzmacniamy proporcjonalnie do błędu
            self.W -= learning_rate * np.einsum("i,j,k->ijk", f_t, a, error_vec)

        # Weight decay — zapobiega eksplozji wag
        self.W *= 0.999

        avg_error = total_error / len(sequences)
        self.total_updates += len(sequences)
        self.recent_errors.append(avg_error)
        if len(self.recent_errors) > 10:
            self.recent_errors.pop(0)

        return avg_error
    
    def get_stats(self):
        """Get learning statistics (for debugging)"""
        return {
            "total_updates": self.total_updates,
            "weight_mean": np.mean(self.W),
            "weight_std": np.std(self.W),
            "weight_max": np.max(np.abs(self.W)),
            "recent_avg_error": (
                np.mean(self.recent_errors) if self.recent_errors else 0.0
            ),
        }
