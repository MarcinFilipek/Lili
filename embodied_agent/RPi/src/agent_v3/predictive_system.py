import numpy as np
from predictive_matrix import PredictiveMatrix


# Zakres pozycji serwomechanizmów do normalizacji → [-1, 1]
PAN_RANGE  = 90.0
TILT_RANGE = 90.0


class PlaceModel:
    """
    Model miejsca: f(brightness, pan_norm, tilt_norm) → brightness_expected

    Uczy się co agent *powinien* widzieć w danej pozycji absolutnej,
    niezależnie od tego skąd tam przyszedł.

    Wejście: [brightness(9), pan_norm, tilt_norm] = 11 dim
    Wyjście: brightness(9)
    Architektura: Linear(11→32) → ReLU → Linear(32→9)
    """

    def __init__(self, feature_dim: int = 9, hidden_dim: int = 32):
        self.feature_dim = feature_dim
        input_dim = feature_dim + 2  # +pan_norm, +tilt_norm

        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, feature_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(feature_dim)

        self.recent_errors = []

    @staticmethod
    def _norm_pos(pan_deg: float, tilt_deg: float) -> np.ndarray:
        return np.array([pan_deg / PAN_RANGE, tilt_deg / TILT_RANGE])

    def _forward(self, feature_vec, pos_norm):
        x = np.concatenate([feature_vec, pos_norm])
        h = np.maximum(0.0, x @ self.W1 + self.b1)
        return h, x @ self.W1 + self.b1, x, h @ self.W2 + self.b2

    def predict(self, feature_vec: np.ndarray, pan_deg: float, tilt_deg: float) -> np.ndarray:
        pos = self._norm_pos(pan_deg, tilt_deg)
        x = np.concatenate([feature_vec, pos])
        h = np.maximum(0.0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def _gradients(self, feature_vec, pan_deg, tilt_deg, target):
        pos  = self._norm_pos(pan_deg, tilt_deg)
        x    = np.concatenate([feature_vec, pos])
        z1   = x @ self.W1 + self.b1
        h    = np.maximum(0.0, z1)
        pred = h @ self.W2 + self.b2

        error     = pred - target
        mse       = float(np.mean(error ** 2))
        dL        = 2.0 * error / self.feature_dim
        dW2, db2  = np.outer(h, dL), dL
        dh        = dL @ self.W2.T
        dz1       = dh * (z1 > 0).astype(float)
        dW1, db1  = np.outer(x, dz1), dz1
        return mse, dW1, db1, dW2, db2

    def online_step(self, feature_vec, pan_deg, tilt_deg, target, lr=0.001) -> float:
        mse, dW1, db1, dW2, db2 = self._gradients(feature_vec, pan_deg, tilt_deg, target)
        self.W1 -= lr * dW1;  self.b1 -= lr * db1
        self.W2 -= lr * dW2;  self.b2 -= lr * db2
        self.recent_errors.append(mse)
        if len(self.recent_errors) > 50:
            self.recent_errors.pop(0)
        return mse

    def update(self, sequences, learning_rate=0.01, epochs=5) -> float:
        """
        sequences: list of dicts z kluczami:
            'feature': brightness(9), 'pan': float, 'tilt': float
        """
        if not sequences:
            return 0.0
        total = 0.0
        for _ in range(epochs):
            idxs = np.random.permutation(len(sequences))
            epoch_err = 0.0
            for i in idxs:
                s = sequences[i]
                mse, dW1, db1, dW2, db2 = self._gradients(
                    s["feature"], s["pan"], s["tilt"], s["feature"]
                )
                self.W1 -= learning_rate * dW1;  self.b1 -= learning_rate * db1
                self.W2 -= learning_rate * dW2;  self.b2 -= learning_rate * db2
                epoch_err += mse
            total = epoch_err / len(sequences)
        self.recent_errors.append(total)
        if len(self.recent_errors) > 20:
            self.recent_errors.pop(0)
        return total

    def recent_avg_error(self) -> float:
        return float(np.mean(self.recent_errors)) if self.recent_errors else 0.0


class PredictiveSystem:
    """
    Orchestrates per-feature MLP forward models + PlaceModel.

    Motion predictors (istniejące):
        brightness: input=[9 + 2] → ReLU(32) → 9
        corners:    input=[36 + 2] → ReLU(32) → 36

    Place model (nowy):
        input=[brightness(9), pan_norm, tilt_norm] → ReLU(32) → brightness(9)

    Supports two learning modes:
        online_step()  — called every waking step (lr=0.001)
        consolidate()  — called during sleep, batch over STM (lr=0.01, epochs=5)
    """

    def __init__(self):
        self.predictors = {
            "brightness": PredictiveMatrix(feature_dim=121, action_dim=2, hidden_dim=64),
            "corners":    PredictiveMatrix(feature_dim=36,  action_dim=2, hidden_dim=32),
        }
        self.place_model = PlaceModel(feature_dim=121, hidden_dim=64)

    # ------------------------------------------------------------------

    def predict(self, features: dict, action: np.ndarray) -> dict:
        """
        Predict next features given current features + continuous action.

        Args:
            features: dict {name: np.array}
            action:   np.array([Δpan, Δtilt])  — raw degrees, NOT one-hot

        Returns:
            predictions: dict {name: np.array}
        """
        predictions = {}
        for name, vec in features.items():
            if name in self.predictors:
                predictions[name] = self.predictors[name].predict(vec, action)
        return predictions

    def compute_error(self, predictions: dict, actual: dict):
        """
        MSE between predicted and actual features.

        Returns:
            avg_error:       float
            error_breakdown: dict {name: float}
        """
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
        actual_pos:    np.ndarray,
        lr:            float = 0.001,
    ) -> dict:
        """
        One SGD step per motion predictor + place model.

        Args:
            prev_features: features at t
            action:        action taken at t  (Δpan, Δtilt)
            features:      features observed at t+1
            actual_pos:    actual servo position at t+1  (pan_deg, tilt_deg)
            lr:            online learning rate

        Returns:
            online_errors: dict {name: mse, 'place': mse}
        """
        errors = {}
        for name, predictor in self.predictors.items():
            if name in prev_features and name in features:
                errors[name] = predictor.online_step(
                    prev_features[name], action, features[name], lr=lr
                )

        # Place model: ucz się że w pozycji actual_pos widzisz features["brightness"]
        if "brightness" in features:
            errors["place"] = self.place_model.online_step(
                features["brightness"],
                pan_deg=float(actual_pos[0]),
                tilt_deg=float(actual_pos[1]),
                target=features["brightness"],
                lr=lr,
            )

        return errors

    def predict_place(self, feature_vec: np.ndarray, pan_deg: float, tilt_deg: float) -> np.ndarray:
        """Przewiduj co powinno być widoczne w danej pozycji absolutnej."""
        return self.place_model.predict(feature_vec, pan_deg, tilt_deg)

    def compute_place_error(self, feature_vec: np.ndarray, pan_deg: float, tilt_deg: float) -> float:
        """MSE między tym co widzimy a tym czego oczekuje model miejsca."""
        pred = self.place_model.predict(feature_vec, pan_deg, tilt_deg)
        return float(np.mean((pred - feature_vec) ** 2))

    def consolidate(
        self,
        sequences_by_feature: dict,
        place_samples:        list  = None,
        learning_rate:        float = 0.01,
        epochs:               int   = 5,
    ) -> bool:
        """
        Batch SGD over STM sequences during sleep.

        Args:
            sequences_by_feature: from hippocampus.extract_sequences()
            place_samples:        from hippocampus.extract_place_samples()
            learning_rate:        sleep lr
            epochs:               SGD passes over the batch
        """
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

        # Place model konsolidacja batch
        if place_samples:
            place_err = self.place_model.update(
                place_samples, learning_rate=learning_rate, epochs=epochs
            )
            print(f"    [place] batch_err={place_err:.4f}")
            if place_err > 0.01:
                all_converged = False

        return all_converged

    def add_feature_predictor(
        self,
        feature_name: str,
        feature_dim:  int,
        action_dim:   int = 2,
        hidden_dim:   int = 32,
    ):
        """Dynamically add a new MLP predictor."""
        if feature_name not in self.predictors:
            self.predictors[feature_name] = PredictiveMatrix(
                feature_dim=feature_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
            )
            print(f"Added MLP predictor for '{feature_name}' "
                  f"(in={feature_dim + action_dim}, hidden=32, out={feature_dim})")