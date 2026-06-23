import time
import csv
import os
import numpy as np
from datetime import datetime
from eye import Eye
from motor_cortex import MotorCortex
from hippocampus import Hippocampus
from vision_cortex import VisionCortex
from predictive_system import PredictiveSystem


LOG_DIR     = "logs"
WEIGHTS_DIR = "weights"

ONLINE_LR    = 0.001
SLEEP_LR     = 0.01
SLEEP_EPOCHS = 5

# Siatka kotwic
GRID_SIZE        = 9          # 9×9 = 81 punktów
PAN_RANGE        = (-90, 90)
TILT_RANGE       = (-90, 90)
ANCHOR_RADIUS    = 10.0
SIGMA            = 3.0
SETTLE_STEPS     = 200        # kroków poniżej error_threshold → przesunięcie kotwicy
ERROR_THRESHOLD  = 0.05
SLEEP_EVERY      = 1500       # wymuszony sen co N kroków (niezależnie od frustracji)


class AnchorGrid:
    """
    Siatka 9×9 punktów równomiernie rozłożonych w zakresie pan/tilt.
    Agent odwiedza je po kolei — lewa→prawa, góra→dół (raster).
    """

    def __init__(self, grid_size=9,
                 pan_range=(-90, 90), tilt_range=(-90, 90)):
        pan_pts  = np.linspace(pan_range[0],  pan_range[1],  grid_size)
        tilt_pts = np.linspace(tilt_range[0], tilt_range[1], grid_size)
        # Raster: najpierw tilt (wiersze), potem pan (kolumny)
        self.points = [
            np.array([pan, tilt])
            for tilt in tilt_pts
            for pan  in pan_pts
        ]
        self.idx = 0

    @property
    def current(self) -> np.ndarray:
        return self.points[self.idx].copy()

    def advance(self) -> np.ndarray:
        """Przejdź do następnego punktu (z zawijaniem)."""
        self.idx = (self.idx + 1) % len(self.points)
        print(f"    🗺  Anchor → {self.points[self.idx]} "
              f"({self.idx + 1}/{len(self.points)})")
        return self.current


class AgentV3:
    def __init__(self):
        self.eye               = Eye()
        self.motor             = MotorCortex()
        self.hippocampus       = Hippocampus()
        self.vision_cortex     = VisionCortex()
        self.predictive_system = PredictiveSystem()

        self.frustration           = 0.0
        self.frustration_gain      = 1.0
        self.frustration_decay     = 0.05
        self.frustration_threshold = 10.0

        self.grid          = AnchorGrid(GRID_SIZE, PAN_RANGE, TILT_RANGE)
        self.anchor        = self.grid.current
        self.anchor_radius = ANCHOR_RADIUS
        self.sigma         = SIGMA
        self.settle_count  = 0   # kroki z rzędu poniżej progu

        self.step        = 0
        self.sleep_count = 0
        self.logger      = StateLogger(LOG_DIR)

        self._load_weights()

    def _select_action(self) -> np.ndarray:
        delta = np.random.normal(0.0, self.sigma, size=2)
        estimated_next = self.motor.estimated_position + delta
        dist = np.linalg.norm(estimated_next - self.anchor)
        if dist > self.anchor_radius:
            direction = (self.anchor - estimated_next) / (dist + 1e-9)
            delta += direction * (dist - self.anchor_radius + self.sigma)
        return delta

    def _return_to_anchor(self):
        delta = self.anchor - self.motor.estimated_position
        if np.linalg.norm(delta) > 0.5:
            self.motor.execute(delta)
            time.sleep(0.5)

    def run(self):
        print("🤖 Agent v3 (MLP) starting...")
        self.motor.reset()
        self.hippocampus.wake_up()
        time.sleep(5.0)

        prev_features = self._capture_features()
        prev_action   = np.array([0.0, 0.0])

        while True:
            self.step += 1

            # 1. Propriocepcja
            actual_pos = self.motor.read_position()

            # 2. Percepcja
            features = self._capture_features()

            # 3. Predykcja i błąd
            predictions = self.predictive_system.predict(prev_features, prev_action)
            avg_error, error_breakdown = self.predictive_system.compute_error(
                predictions, features
            )

            # 4. Online SGD
            self.predictive_system.online_step(
                prev_features, prev_action, features, lr=ONLINE_LR
            )

            print(
                f"[{self.step:5d}] "
                f"Pos:[{actual_pos[0]:6.1f},{actual_pos[1]:6.1f}] "
                f"Anchor:[{self.anchor[0]:5.1f},{self.anchor[1]:5.1f}] "
                f"#{self.grid.idx+1:02d}/{len(self.grid.points)} "
                f"br:{error_breakdown.get('brightness', 0):.4f} "
                f"ed:{error_breakdown.get('edges', 0):.4f} "
                f"settle:{self.settle_count}/{SETTLE_STEPS}"
            )

            # 5. Frustracja / sen
            if avg_error > ERROR_THRESHOLD:
                self.frustration += avg_error * self.frustration_gain
                self.settle_count = 0
            else:
                self.frustration = max(0.0, self.frustration - self.frustration_decay)
                self.settle_count += 1

            # Przesunięcie kotwicy gdy scena jest poznana
            if self.settle_count >= SETTLE_STEPS:
                self.settle_count = 0
                self.anchor = self.grid.advance()
                self._return_to_anchor()
                prev_features = self._capture_features()
                prev_action   = np.array([0.0, 0.0])
                self.logger.log(
                    step=self.step, actual_pos=actual_pos,
                    estimated_pos=self.motor.estimated_position,
                    anchor=self.anchor, anchor_idx=self.grid.idx,
                    avg_error=avg_error, error_breakdown=error_breakdown,
                    frustration=self.frustration, settle_count=self.settle_count,
                    action=prev_action, sleeping=False, sleep_count=self.sleep_count,
                )
                continue

            if self.frustration > self.frustration_threshold:
                self._do_sleep("frustracja")
                prev_features = self._capture_features()
                prev_action   = np.array([0.0, 0.0])
                self.logger.log(
                    step=self.step, actual_pos=actual_pos,
                    estimated_pos=self.motor.estimated_position,
                    anchor=self.anchor, anchor_idx=self.grid.idx,
                    avg_error=avg_error, error_breakdown=error_breakdown,
                    frustration=self.frustration, settle_count=self.settle_count,
                    action=prev_action, sleeping=True, sleep_count=self.sleep_count,
                )
                continue

            if self.step % SLEEP_EVERY == 0:
                self._do_sleep("timer")
                prev_features = self._capture_features()
                prev_action   = np.array([0.0, 0.0])
                self.logger.log(
                    step=self.step, actual_pos=actual_pos,
                    estimated_pos=self.motor.estimated_position,
                    anchor=self.anchor, anchor_idx=self.grid.idx,
                    avg_error=avg_error, error_breakdown=error_breakdown,
                    frustration=self.frustration, settle_count=self.settle_count,
                    action=prev_action, sleeping=True, sleep_count=self.sleep_count,
                )
                continue

            # 6. Zapis do STM
            self.hippocampus.record(prev_features, prev_action)

            # 7. Wybór i wykonanie akcji
            action = self._select_action()
            self.motor.execute(action)

            # 8. Log
            self.logger.log(
                step=self.step, actual_pos=actual_pos,
                estimated_pos=self.motor.estimated_position,
                anchor=self.anchor, anchor_idx=self.grid.idx,
                avg_error=avg_error, error_breakdown=error_breakdown,
                frustration=self.frustration, settle_count=self.settle_count,
                action=action, sleeping=False, sleep_count=self.sleep_count,
            )

            prev_features = features
            prev_action   = action
            time.sleep(0.2)

    def _do_sleep(self, reason: str):
        self.sleep_count += 1
        self.hippocampus.go_sleep()
        sequences     = self.hippocampus.extract_sequences()
        ltm_sequences = self.hippocampus.sample_ltm(200)

        n_stm = len(sequences.get('brightness', []))
        n_ltm = len(ltm_sequences.get('brightness', []))
        print(f"    💤 Sleep #{self.sleep_count} [{reason}] — "
              f"STM:{n_stm} + LTM:{n_ltm} replay, {SLEEP_EPOCHS} epochs")

        combined = {}
        for name in set(list(sequences.keys()) + list(ltm_sequences.keys())):
            combined[name] = sequences.get(name, []) + ltm_sequences.get(name, [])

        self.predictive_system.consolidate(
            combined, learning_rate=SLEEP_LR, epochs=SLEEP_EPOCHS
        )
        self.hippocampus.consolidate_to_ltm(sequences)
        self.hippocampus.wake_up()
        self.frustration = 0.0
        self._save_weights()
        self._return_to_anchor()

    def _save_weights(self):
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        for name, p in self.predictive_system.predictors.items():
            np.savez(os.path.join(WEIGHTS_DIR, f"{name}.npz"),
                     W1=p.W1, b1=p.b1, W2=p.W2, b2=p.b2)
        print(f"    💾 Weights saved (sleep #{self.sleep_count})")

    def _load_weights(self):
        loaded = []
        skipped = []
        for name, p in self.predictive_system.predictors.items():
            path = os.path.join(WEIGHTS_DIR, f"{name}.npz")
            if os.path.exists(path):
                d = np.load(path)
                if d["W1"].shape == p.W1.shape:
                    p.W1, p.b1, p.W2, p.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
                    loaded.append(name)
                else:
                    skipped.append(f"{name}(kształt {d['W1'].shape}≠{p.W1.shape})")
        if loaded:
            print(f"📂 Loaded weights: {', '.join(loaded)}")
        if skipped:
            print(f"⚠️  Skipped (incompatible): {', '.join(skipped)} — starting fresh")
        if not loaded and not skipped:
            print("📂 No saved weights found — starting fresh")

    def _capture_features(self):
        frame    = self.eye.capture(gray=True, blur=False)
        features = self.vision_cortex.extract_features(frame)
        return features


class StateLogger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"agent_{ts}.csv")
        self._fields = [
            "timestamp", "step",
            "pan_actual", "tilt_actual",
            "pan_estimated", "tilt_estimated",
            "anchor_pan", "anchor_tilt", "anchor_idx",
            "avg_error", "error_brightness", "error_edges",
            "frustration", "settle_count",
            "action_pan", "action_tilt",
            "sleeping", "sleep_count",
        ]
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fields).writeheader()
        print(f"📝 Logging to {self.path}")

    def log(self, step, actual_pos, estimated_pos, anchor, anchor_idx,
            avg_error, error_breakdown, frustration, settle_count, action,
            sleeping, sleep_count):
        row = {
            "timestamp":        datetime.now().isoformat(timespec="milliseconds"),
            "step":             step,
            "pan_actual":       round(float(actual_pos[0]), 3),
            "tilt_actual":      round(float(actual_pos[1]), 3),
            "pan_estimated":    round(float(estimated_pos[0]), 3),
            "tilt_estimated":   round(float(estimated_pos[1]), 3),
            "anchor_pan":       round(float(anchor[0]), 3),
            "anchor_tilt":      round(float(anchor[1]), 3),
            "anchor_idx":       anchor_idx,
            "avg_error":        round(float(avg_error), 6),
            "error_brightness": round(float(error_breakdown.get("brightness", 0.0)), 6),
            "error_edges":      round(float(error_breakdown.get("edges",      0.0)), 6),
            "frustration":      round(float(frustration), 4),
            "settle_count":     settle_count,
            "action_pan":       round(float(action[0]), 3),
            "action_tilt":      round(float(action[1]), 3),
            "sleeping":         int(sleeping),
            "sleep_count":      sleep_count,
        }
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fields).writerow(row)


def main():
    agent = AgentV3()
    try:
        agent.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        agent.eye.close()


if __name__ == "__main__":
    main()