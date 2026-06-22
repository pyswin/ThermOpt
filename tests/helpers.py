import numpy as np


class DummyThermalBackend:
    name = "dummy"
    runtime_mode = "dummy"

    def __init__(self, shape: tuple[int, int] = (16, 16), value: float = 42.0):
        self.shape = shape
        self.value = value

    def simulate(self, case, layout):
        return np.full(self.shape, self.value, dtype=float)
