from collections import deque
from torchvision.datasets._stereo_matching import FallingThingsStereo

STM_CAPACITY = 30


class Hippocampus:
    def __init__(self):
        self.slepping = True
        self.stm = deque(maxlen=STM_CAPACITY)
        self.ltm = []

    def wake_up(self):
        self.slepping = False
        print("\n☀️  Waking up — STM cleared, LTM intact")
