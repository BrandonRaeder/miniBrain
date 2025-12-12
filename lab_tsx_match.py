
import numpy as np
import math
from collections import deque

class MiniBrainSim:
    def __init__(self, N=128):
        self.N = N
        self.neurons