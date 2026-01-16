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
        """
        Hebbian learning from sequences.
        Strengthens associations: feature_t → action → feature_t+1

        Args:
            sequences: list of dicts
                [{'feature_t': array, 'action': array, 'feature_t1': array}, ...]
            learning_rate: float

        Returns:
            avg_error: float (average prediction error across sequences)
        """
        if len(sequences) == 0:
            return 0.0

        total_error = 0.0

        for seq in sequences:
            f_t = seq["feature_t"]  # current feature
            a = seq["action"]  # action taken
            f_t1 = seq["feature_t1"]  # next feature (target)

            # Current prediction
            prediction = self.predict(f_t, a)

            # Prediction error
            error = np.mean((prediction - f_t1) ** 2)
            total_error += error

            # Hebbian update: strengthen the 3-way association
            # ΔW[i,j,k] = lr * f_t[i] * a[j] * f_t1[k]
            # This is the outer product of three vectors
            for i in range(self.feature_dim):
                for j in range(self.action_dim):
                    for k in range(self.feature_dim):
                        self.W[i, j, k] += learning_rate * f_t[i] * a[j] * f_t1[k]

        avg_error = total_error / len(sequences)

        # Track statistics
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
