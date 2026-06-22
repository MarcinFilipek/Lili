import numpy as np


class PredictiveMatrix:
    """
    Forward model: MLP predicting next_feature given (current_feature, action).

    Architecture:  [feature_dim + action_dim]  →  ReLU(32)  →  [feature_dim]
    Input:   np.concatenate([feature_vec, action_vec])   shape (feature_dim + action_dim,)
    Output:  predicted next_feature                       shape (feature_dim,)

    Replaces the 3D tensor W[f,a,f] which required discrete/one-hot actions.
    MLP handles continuous (Δpan, Δtilt) naturally and generalises between actions.

    Learning modes:
        online_step()  — single SGD step after every perception cycle  (lr ≈ 0.001)
        update()       — batch SGD over STM sequences during sleep      (lr ≈ 0.01)
    Both modes share the same weights; sleep consolidation simply runs more epochs.
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 32):
        self.feature_dim = feature_dim
        self.action_dim  = action_dim
        self.hidden_dim  = hidden_dim
        self.input_dim   = feature_dim + action_dim

        # He initialisation — good default for ReLU networks
        self.W1 = np.random.randn(self.input_dim, hidden_dim) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, feature_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(feature_dim)

        # Monitoring
        self.total_updates  = 0
        self.recent_errors  = []   # rolling window of 20 batch errors
        self.online_errors  = []   # rolling window of 50 online errors

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def predict(self, feature_vec: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
        """
        Predict next feature vector.

        Args:
            feature_vec: shape (feature_dim,)  — current observation
            action_vec:  shape (action_dim,)   — continuous action (Δpan, Δtilt)

        Returns:
            prediction:  shape (feature_dim,)
        """
        x  = np.concatenate([feature_vec, action_vec])
        h  = np.maximum(0.0, x @ self.W1 + self.b1)   # ReLU
        return h @ self.W2 + self.b2

    # ------------------------------------------------------------------
    # Internal backward pass — returns weight gradients for one sample
    # ------------------------------------------------------------------

    def _gradients(self, feature_vec, action_vec, feature_next):
        x      = np.concatenate([feature_vec, action_vec])
        z1     = x @ self.W1 + self.b1
        h      = np.maximum(0.0, z1)
        pred   = h @ self.W2 + self.b2

        error  = pred - feature_next          # shape (feature_dim,)
        mse    = np.mean(error ** 2)

        # Output layer gradients
        dL_dpred = 2.0 * error / self.feature_dim
        dW2 = np.outer(h, dL_dpred)           # (hidden, feature_dim)
        db2 = dL_dpred

        # Hidden layer gradients (ReLU mask)
        dh      = dL_dpred @ self.W2.T        # (hidden,)
        dz1     = dh * (z1 > 0).astype(float)
        dW1     = np.outer(x, dz1)            # (input, hidden)
        db1     = dz1

        return mse, dW1, db1, dW2, db2

    # ------------------------------------------------------------------
    # Online learning — one SGD step per perception cycle
    # ------------------------------------------------------------------

    def online_step(
        self,
        feature_vec:  np.ndarray,
        action_vec:   np.ndarray,
        feature_next: np.ndarray,
        lr: float = 0.001,
    ) -> float:
        """
        Single SGD step on one (f_t, a, f_t+1) triplet.
        Called every step during waking exploration.

        Returns MSE for this sample.
        """
        mse, dW1, db1, dW2, db2 = self._gradients(feature_vec, action_vec, feature_next)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        self.online_errors.append(mse)
        if len(self.online_errors) > 50:
            self.online_errors.pop(0)

        return mse

    # ------------------------------------------------------------------
    # Batch learning — multiple SGD epochs over STM sequences (sleep)
    # ------------------------------------------------------------------

    def update(
        self,
        sequences: list,
        learning_rate: float = 0.01,
        epochs: int = 5,
    ) -> float:
        """
        Batch SGD over a list of (f_t, action, f_t+1) sequences.
        Called during sleep/consolidation.

        Args:
            sequences:     list of dicts with keys 'feature_t', 'action', 'feature_t1'
            learning_rate: step size (typically 5-10× online lr)
            epochs:        number of passes over the batch

        Returns:
            avg MSE of the last epoch.
        """
        if not sequences:
            return 0.0

        total_error = 0.0

        for _ in range(epochs):
            # Shuffle to break temporal correlations
            idxs = np.random.permutation(len(sequences))
            epoch_error = 0.0

            for i in idxs:
                seq  = sequences[i]
                mse, dW1, db1, dW2, db2 = self._gradients(
                    seq["feature_t"], seq["action"], seq["feature_t1"]
                )
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
                epoch_error += mse

            total_error = epoch_error / len(sequences)

        self.total_updates += len(sequences) * epochs
        self.recent_errors.append(total_error)
        if len(self.recent_errors) > 20:
            self.recent_errors.pop(0)

        return total_error

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "total_updates":    self.total_updates,
            "W1_mean":          round(float(np.mean(self.W1)), 5),
            "W1_std":           round(float(np.std(self.W1)),  5),
            "W2_std":           round(float(np.std(self.W2)),  5),
            "recent_batch_err": round(float(np.mean(self.recent_errors))  if self.recent_errors  else 0.0, 5),
            "recent_online_err":round(float(np.mean(self.online_errors))  if self.online_errors  else 0.0, 5),
        }