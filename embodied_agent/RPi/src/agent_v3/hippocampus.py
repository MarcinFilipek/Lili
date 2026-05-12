from collections import deque

STM_CAPACITY = 30


class Hippocampus:
    def __init__(self):
        self.sleeping = True
        self.stm = deque(maxlen=STM_CAPACITY)
        self.ltm = []

    def wake_up(self):
        self.slepping = False
        print("\n☀️  Waking up — STM cleared, LTM intact")
