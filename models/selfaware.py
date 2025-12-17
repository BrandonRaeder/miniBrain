
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from sklearn.metrics import r2_score

# Tuned phi_proxy for Φ/N=1.0
def phi_proxy(x):
    N = len(x)
    if N == 0:
        return 0.0
    mid = N // 2
    left = x[:mid]
    right = x[mid:]
    # Simplified shannon
    def shannon(data, bins=16):
        if len(data) == 0:
            return 0.0
        hist, _ = np.histogram(data, bins=bins)
        hist = hist[hist > 0] / np.sum(hist)
        return -np.sum(hist * np.log2(hist))
    H_left = shannon(left)
    H_right = shannon(right)
    H_total = shannon(x)
    mutualInfo = max(0.0, H_left + H_right - H_total)
    coherence = np.abs(np.mean(np.exp(1j * x)))
    phi = mutualInfo * coherence * N * 0.2  # tuned 0.12 -> 0.2
    return max(0.0, phi)

class OptimizedMetrics:
    def __init__(self):
        self.entropy_cache = {}
    
    def cached_entropy(self, data):
        if len(data) < 5:
            return 0.0
        key = str(len(data[-20:]))
        if key in self.entropy_cache:
            return self.entropy_cache[key]
        h, _ = np.histogram(data[-20:], bins=10)
        h = h[h > 0] / np.sum(h)
        ent = -np.sum(h * np.log(h + 1e-10))
        self.entropy_cache[key] = ent
        return ent