"""Kopia eferentna: komenda dla serw -> przewidywana poza kamery.

Przewidywanie, nie pomiar: nie zawiera histerezy serw (ok. 0.3-0.4 deg), którą można
doprecyzować z obrazu (perception/rotation.py).
"""

import json
from pathlib import Path

from .geometry import Head, joint_behind_camera
from .servo_model import load_servo_models


class Efference:
    def __init__(self, head, pan_model, tilt_model):
        self.head = head
        self.pan_model = pan_model
        self.tilt_model = tilt_model

    @classmethod
    def from_calibration(cls, calib_dir):
        """Z calib/servo.json (skale i osie serw) oraz calib/head.json (ramiona dźwigni)."""
        calib_dir = Path(calib_dir)
        servos = load_servo_models(calib_dir / "servo.json")
        head = json.loads((calib_dir / "head.json").read_text())
        pan, tilt = servos["pan"], servos["tilt"]
        return cls(
            Head(
                pan=joint_behind_camera(pan.axis, head["pan_lever_m"]),
                tilt=joint_behind_camera(tilt.axis, head["tilt_lever_m"]),
            ),
            pan,
            tilt,
        )

    def pose(self, pan_cmd, tilt_cmd):
        """Poza kamery dla komend (jednostki protokołu) względem pozy dla komend (0, 0)."""
        return self.head.pose(self.pan_model.angle(pan_cmd), self.tilt_model.angle(tilt_cmd))

    def motion(self, cmd_a, cmd_b):
        """Ruch kamery z pozy cmd_a = (pan, tilt) do cmd_b, w układzie kamery a: Pose(R, T)."""
        return self.pose(*cmd_b).relative_to(self.pose(*cmd_a))
