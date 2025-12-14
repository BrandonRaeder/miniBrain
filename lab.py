import time
script_start_time = time.time()
print(f"[{time.time() - script_start_time:.4f}s] Script execution started.")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
try:
    matplotlib.use('Qt5Agg')
except ImportError:
    try:
        matplotlib.use('GTK3Agg')
    except ImportError:
        matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAVE_TORCH = True
    HAVE_IPEX = False # Initialize HAVE_IPEX here
    # --- MODIFIED DEVICE DETECTION for Intel, AMD, and NVIDIA GPU support ---
    if torch.cuda.is_available():
        # Differentiate between NVIDIA and AMD GPUs
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            device = torch.device('cuda')
            print("Using device: AMD GPU (ROCm)")
        else:
            device = torch.device('cuda')
            print("Using device: NVIDIA GPU (CUDA)")
    else:
        # Check for Intel GPU (XPU) if IPEX is installed
        ipex = None # Initialize ipex to None
        try:
            import intel_extension_for_pytorch as ipex
            HAVE_IPEX = True # Set HAVE_IPEX to True if import succeeds
        except ImportError:
            pass # ipex remains None if import fails

        if HAVE_IPEX and torch.xpu.is_available(): # Guard with HAVE_IPEX
            device = torch.device('xpu')
            print("Using device: Intel GPU (XPU)")
        else:
            device = torch.device('cpu')
            print("Using device: CPU")
    print(f"Selected device: {device}")
    print(f"[{time.time() - script_start_time:.4f}s] PyTorch device setup finished.")
except Exception:
    torch = None
    nn = None
    optim = None
    HAVE_TORCH = False
    HAVE_IPEX = False # Set HAVE_IPEX to False here as well
    device = None
try:
    import numba as nb
    HAVE_NUMBA = True
except ImportError:
    nb = None
    HAVE_NUMBA = False
import threading
import sys
import time
from collections import deque
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mutual_info_score
print(f"[{time.time() - script_start_time:.4f}s] All imports finished.")

# Performance optimization configuration
ENABLE_CACHING = True
CACHE_SIZE = 30
METRIC_UPDATE_INTERVAL = 5  # Update metrics every frame for immediate display
REDUCED_HISTORY_SIZE = 200  # Reduced from 2000
ROLLING_WINDOW = 100  # Rolling window of 1k steps

class OptimizedMetrics:
    def __init__(self):
        self.entropy_cache = {}
        self.lyap_cache = {}
        self.frame_counter = 0
        
    def should_update_metrics(self):
        """Decide whether to update heavy metrics this frame"""
        self.frame_counter += 1
        return self.frame_counter % METRIC_UPDATE_INTERVAL == 0
    
    def cached_entropy(self, data, key_suffix=""):
        """Cache entropy calculations"""
        if not ENABLE_CACHING or len(data) < 10:
            return self._compute_entropy_fast(data)
        
        key = f"{len(data)}_{hash(str(data[-20:]))}_{key_suffix}"
        if key in self.entropy_cache:
            return self.entropy_cache[key]
        
        result = self._compute_entropy_fast(data)
        self.entropy_cache[key] = result
        
        if len(self.entropy_cache) > CACHE_SIZE:
            self.entropy_cache.clear()
            
        return result
    
    def _compute_entropy_fast(self, data, bins=15):  # Reduced bins
        """Fast entropy computation on recent data only"""
        if len(data) < 5:
            return 0.0
        # Use last 25 values only for speed
        recent_data = data[-25:] if len(data) > 25 else data
        hist, _ = np.histogram(recent_data, bins=bins, density=False)
        if np.sum(hist) == 0:
            return 0.0
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist + 1e-10))
    
    def cached_lyapunov(self, data, key_suffix=""):
        """Cache lyapunov calculations"""
        if not ENABLE_CACHING or len(data) < 10:
            return self._compute_lyapunov_fast(data)
        
        key = f"lyap_{len(data)}_{hash(str(data[-20:]))}_{key_suffix}"
        if key in self.lyap_cache:
            return self.lyap_cache[key]
        
        result = self._compute_lyapunov_fast(data)
        self.lyap_cache[key] = result
        
        if len(self.lyap_cache) > CACHE_SIZE:
            self.lyap_cache.clear()
            
        return result
    
    def _compute_lyapunov_fast(self, data):
        """Fast lyapunov proxy computation on recent data only"""
        if len(data) < 2:
            return 0.0
        diffs = np.abs(np.diff(data))
        return np.mean(diffs)
    

# Fast LZ complexity
@nb.njit(cache=True)
def shannon_entropy(data, bins=32):
    """Shannon entropy from miniBrain.tsx"""
    if len(data) == 0:
        return 0.0
    hist, _ = np.histogram(data, bins=bins)
    if np.sum(hist) == 0:
        return 0.0
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def phi_proxy(x, bins=32):
    """Φ proxy from miniBrain.tsx calculatePhiProxy"""
    N = len(x)
    if N == 0:
        return 0.0
    mid = N // 2
    left = x[:mid]
    right = x[mid:]
    H_left = shannon_entropy(left, bins)
    H_right = shannon_entropy(right, bins)
    H_total = shannon_entropy(x, bins)
    mutualInfo = max(0.0, H_left + H_right - H_total)
    effectiveInfo = 0.0
    for i in range(min(N, 32)):
        neighbors = [x[(i-1) % N], x[(i+1) % N]]
        local_data = np.array([x[i], *neighbors])
        effectiveInfo += shannon_entropy(local_data, bins=8)
    effectiveInfo /= min(N, 32)
    coherence = np.abs(np.mean(np.exp(1j * x)))
    phi = mutualInfo * effectiveInfo * (1 + coherence) * N * 0.12
    return max(0.0, phi)


@nb.njit(cache=True)
def lz_complexity_fast(binary_sequence, max_length=100):
    """Simplified LZ complexity for performance"""
    if len(binary_sequence) == 0:
        return 0
    
    # Convert to string and limit length for performance
    s = ''.join(['1' if x else '0' for x in binary_sequence[:max_length]])
    # Numba doesn't support len() on string, so we calculate it manually
    n = 0
    for _ in s:
        n += 1
    
    if n == 0:
        return 0
    
    # Very simplified LZ algorithm
    i = 0
    c = 1
    l = 1
    
    while i + l <= n and c < 30:  # Further limit iterations
        found = False
        # Numba doesn't support string slicing easily, so we do manual compare
        # This is a simplified check for a repeating pattern
        if i > 0:
            sub = s[i:i+l]
            prev = s[i-1:i-1+l]
            if sub == prev:
                found = True

        
        if found:
            l += 1
        else:
            c += 1
            i += l
            l = 1
    
    return c

# Global optimizer
metrics = OptimizedMetrics()

def train_workspace_decoder(x_hist, ws_hist, subset_idx=None, alpha=1.0):
    # x_hist: (T, n_layers) or (n_steps, n_layers) as in simulate_workspace output
    X = x_hist.copy()    # shape (T, n_layers)
    y = ws_hist.copy()   # shape (T,)
    if subset_idx is not None:
        X = X[:, subset_idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return model, r2

# ---------- Common bistable layer ----------
@nb.njit(cache=True)
def _bistable_layer_numpy_jit(x, alpha, theta_eff):
    return np.tanh(alpha * x - theta_eff)

def bistable_layer(x, alpha, theta_eff):
    # This function is polymorphic for numpy/torch
    if HAVE_TORCH and isinstance(x, torch.Tensor):
        return torch.tanh(alpha * x - theta_eff)
    else:
        return _bistable_layer_numpy_jit(x, alpha, theta_eff)



def update_bistable_np(x, input_val, tau, anti_converge, dt, perturb_phase):
    """NumPy port from miniBrain.tsx updateBistable for Model C"""
    n = len(x)
    phases = perturb_phase + np.arange(n) * 0.13
    bistable = x * (1 - x * x)
    anti_term = anti_converge * np.sin(x * 9.3 + phases)
    dx = (-x / tau) + bistable + input_val + anti_term
    return np.tanh(x + dx * dt)

# ---------- Why-loop driver ----------
@nb.njit(cache=True)
def _why_loop_driver_numpy_jit(y, gamma):
    return np.tanh(gamma * y)  # reflective recursion term

def why_loop_driver(y, gamma):
    # Polymorphic for numpy/torch
    if HAVE_TORCH and isinstance(y, torch.Tensor):
        return torch.tanh(gamma * y)  # reflective recursion term
    else:
        return _why_loop_driver_numpy_jit(y, gamma)  # reflective recursion term

# ---------- Option A: with global workspace ----------
def simulate_workspace(n_layers=100, T=1000, dt=0.01,
                        alpha=0.8, eps=0.7, theta_eff=0.3, k_ws=0.05):
    if HAVE_TORCH:
        x = torch.randn(n_layers, device=device)
        ws = torch.tensor(0.0, device=device)
        R_hist, ws_hist = [], []
        x_hist = []
        for t in range(T):
            # local dynamics
            dx = -x + bistable_layer(x, alpha, theta_eff) + eps * ws
            x += dt * dx

            # workspace collects average activity
            ws = (1 - k_ws) * ws + k_ws * torch.mean(x)

            # record coherence & workspace activity
            R = torch.abs(torch.mean(torch.exp(1j * x)))
            R_hist.append(R.item())
            ws_hist.append(ws.item())
            x_hist.append(x.cpu().numpy().copy())
        return np.array(R_hist), np.array(ws_hist), np.array(x_hist)
    else:
        x = np.random.randn(n_layers)
        ws = 0.0
        R_hist, ws_hist = [], []
        x_hist = []
        for t in range(T):
            # local dynamics
            dx = -x + bistable_layer(x, alpha, theta_eff) + eps * ws
            x += dt * dx

            # workspace collects average activity
            ws = (1 - k_ws) * ws + k_ws * np.mean(x)

            # record coherence & workspace activity
            R_hist.append(np.abs(np.mean(np.exp(1j * x))))
            ws_hist.append(ws)
            x_hist.append(x.copy())
        return np.array(R_hist), np.array(ws_hist), np.array(x_hist)

# ---------- Option B: hierarchical reflective why-loop ----------
def simulate_reflective_hierarchy(T=1000, dt=0.01,
                                   alpha=0.8, eps=0.7,
                                   theta_eff=0.3, gamma=1.2, k_ws=0.05):
    if HAVE_TORCH:
        x = torch.randn(100, device=device)             # bistable cascade
        y = torch.randn(100, device=device)             # theta_eff system
        why = torch.tensor(0.0, device=device)           # lowest reflective driver
        ws = torch.tensor(0.0, device=device)            # global workspace
        R_hist, ws_hist = [], []

        for t in range(T):
            # lowest reflective recursion
            why = why_loop_driver(why, gamma)

            # middle coupling of bistable + theta_eff + why influence
            dx = -x + bistable_layer(x, alpha, theta_eff + 0.2 * why)
            dy = -y + bistable_layer(y, alpha, theta_eff - 0.2 * why)

            # combine middle systems
            combined = (x + y) / 2.0
            x += dt * dx
            y += dt * dy

            # workspace receives combined and broadcasts back
            ws = (1 - k_ws) * ws + k_ws * torch.mean(combined)
            x += eps * ws * dt
            y += eps * ws * dt

            R = torch.abs(torch.mean(torch.exp(1j * combined)))
            R_hist.append(R.item())
            ws_hist.append(ws.item())
        return np.array(R_hist), np.array(ws_hist)
    else:
        x = np.random.randn(100)             # bistable cascade
        y = np.random.randn(100)             # theta_eff system
        why = 0.0                            # lowest reflective driver
        ws = 0.0                             # global workspace
        R_hist, ws_hist = [], []

        for t in range(T):
            # lowest reflective recursion
            why = why_loop_driver(why, gamma)

            # middle coupling of bistable + theta_eff + why influence
            dx = -x + bistable_layer(x, alpha, theta_eff + 0.2 * why)
            dy = -y + bistable_layer(y, alpha, theta_eff - 0.2 * why)

            # combine middle systems
            combined = (x + y) / 2.0
            x += dt * dx
            y += dt * dy

            # workspace receives combined and broadcasts back
            ws = (1 - k_ws) * ws + k_ws * np.mean(combined)
            x += eps * ws * dt
            y += eps * ws * dt

            R_hist.append(np.abs(np.mean(np.exp(1j * combined))))
            ws_hist.append(ws)
        return np.array(R_hist), np.array(ws_hist)


# ---------- Option B Self-Referential: Self-Referential Reflective Hierarchy ----------
def simulate_self_referential_reflective_hierarchy(T=1000, dt=0.01,
                                                   alpha=0.8, eps=0.7,
                                                   theta_eff=0.3, gamma=1.2, k_ws=0.05,
                                                   meta_learning_rate=0.01):
    if HAVE_TORCH:
        x = torch.randn(100, device=device)             # bistable cascade
        y = torch.randn(100, device=device)             # theta_eff system
        why = torch.tensor(0.0, device=device)           # lowest reflective driver
        ws = torch.tensor(0.0, device=device)            # global workspace
        self_model = np.zeros(8)  # [mean_x, std_x, mean_y, std_y, why, ws, entropy, coherence]
        predicted_self = np.zeros(8)
        predictor = None

        R_hist, ws_hist = [], []
        self_error_hist = []
        self_model_hist = []

        for t in range(T):
            # self-reference error and awareness term
            self_error = np.linalg.norm(predicted_self - self_model)
            self_awareness_term = meta_learning_rate * self_error * (self_model[0] if self_model.size>0 else 0.0)

            # lowest reflective recursion
            why = why_loop_driver(why, gamma)

            # middle coupling of bistable + theta_eff + why influence + self-awareness
            dx = -x + bistable_layer(x, alpha, theta_eff + 0.2 * why) + self_awareness_term
            dy = -y + bistable_layer(y, alpha, theta_eff - 0.2 * why) + self_awareness_term

            # combine middle systems
            combined = (x + y) / 2.0
            x += dt * dx
            y += dt * dy

            # workspace receives combined and broadcasts back
            ws = (1 - k_ws) * ws + k_ws * torch.mean(combined)
            x += eps * ws * dt
            y += eps * ws * dt

            R = torch.abs(torch.mean(torch.exp(1j * combined)))
            R_hist.append(R.item())
            ws_hist.append(ws.item())

            # encode self-model
            mean_x = torch.mean(x).item()
            std_x = torch.std(x).item()
            mean_y = torch.mean(y).item()
            std_y = torch.std(y).item()
            why_val = why.item()
            ws_val = ws.item()
            entropy = metrics.cached_entropy(np.array(R_hist[-10:]), "B_self") if len(R_hist) > 10 else 0.0
            coherence = R.item()
            self_model = np.array([mean_x, std_x, mean_y, std_y, why_val, ws_val, entropy, coherence])

            # predict own future
            if predictor is not None:
                with torch.no_grad():
                    pred = predictor(torch.tensor(self_model, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
                    predicted_self = pred

            self_error_hist.append(self_error)
            self_model_hist.append(self_model.copy())

        return np.array(R_hist), np.array(ws_hist), np.array(self_error_hist), np.array(self_model_hist)
    else:
        x = np.random.randn(100)             # bistable cascade
        y = np.random.randn(100)             # theta_eff system
        why = 0.0                            # lowest reflective driver
        ws = 0.0                             # global workspace
        self_model = np.zeros(8)
        predicted_self = np.zeros(8)
        predictor = None

        R_hist, ws_hist = [], []
        self_error_hist = []
        self_model_hist = []

        for t in range(T):
            # self-reference error and awareness term
            self_error = np.linalg.norm(predicted_self - self_model)
            self_awareness_term = meta_learning_rate * self_error * (self_model[0] if self_model.size>0 else 0.0)

            # lowest reflective recursion
            why = why_loop_driver(why, gamma)

            # middle coupling of bistable + theta_eff + why influence + self-awareness
            dx = -x + bistable_layer(x, alpha, theta_eff + 0.2 * why) + self_awareness_term
            dy = -y + bistable_layer(y, alpha, theta_eff - 0.2 * why) + self_awareness_term

            # combine middle systems
            combined = (x + y) / 2.0
            x += dt * dx
            y += dt * dy

            # workspace receives combined and broadcasts back
            ws = (1 - k_ws) * ws + k_ws * np.mean(combined)
            x += eps * ws * dt
            y += eps * ws * dt

            R_hist.append(np.abs(np.mean(np.exp(1j * combined))))
            ws_hist.append(ws)

            # encode self-model
            mean_x = np.mean(x)
            std_x = np.std(x)
            mean_y = np.mean(y)
            std_y = np.std(y)
            why_val = why
            ws_val = ws
            entropy = metrics.cached_entropy(np.array(R_hist[-10:]), "B_self") if len(R_hist) > 10 else 0.0
            coherence = R_hist[-1]
            self_model = np.array([mean_x, std_x, mean_y, std_y, why_val, ws_val, entropy, coherence])

            # predict own future
            if predictor is not None:
                with torch.no_grad():
                    pred = predictor(torch.tensor(self_model, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                    predicted_self = pred

            self_error_hist.append(self_error)
            self_model_hist.append(self_model.copy())

        return np.array(R_hist), np.array(ws_hist), np.array(self_error_hist), np.array(self_model_hist)

# ---------- Option C: Self-Referential Workspace (step-simulated version) ----------
def simulate_self_referential_workspace(n_layers=100, T=1000, dt=0.01,
                                        alpha=0.8, eps=0.7, theta_eff=0.3, k_ws=0.05,
                                        meta_learning_rate=0.01):
    if HAVE_TORCH:
        x = torch.randn(n_layers, device=device)
        ws = torch.tensor(0.0, device=device)
        self_model = np.zeros(6)
        predicted_self = np.zeros(6)
        predictor = None

        R_hist = []
        ws_hist = []
        x_hist = []
        self_error_hist = []
        self_model_hist = []

        for t in range(T):
            # self-reference error and awareness term
            self_error = np.linalg.norm(predicted_self - self_model)
            self_awareness_term = meta_learning_rate * self_error * (self_model[0] if self_model.size>0 else 0.0)

            dx = -x + bistable_layer(x, alpha, theta_eff) + eps * ws + self_awareness_term
            x += dt * dx

            ws = (1 - k_ws) * ws + k_ws * torch.mean(x)

            coherence = torch.abs(torch.mean(torch.exp(1j * x)))
            if len(R_hist) > 10:
                recent_R = np.array(R_hist[-10:])
                entropy = metrics.cached_entropy(recent_R, "C_direct")
            else:
                entropy = 0.0

            complexity_metrics = {'entropy': entropy, 'coherence': coherence.item()}

            # encode simple self-model
            mean_activity = torch.mean(x).item()
            std_activity = torch.std(x).item()
            trend = np.mean(np.diff(R_hist[-10:])) if len(R_hist) > 10 else 0.0
            ws_normalized = torch.tanh(ws).item()
            self_model = np.array([mean_activity, std_activity, trend, ws_normalized, entropy, coherence.item()])

            # predict own future (predictor not trained here)
            if predictor is not None:
                with torch.no_grad():
                    pred = predictor(torch.tensor(self_model, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
                    predicted_self = pred

            R_hist.append(coherence.item())
            ws_hist.append(ws.item())
            x_hist.append(x.cpu().numpy().copy())
            self_error_hist.append(self_error)
            self_model_hist.append(self_model.copy())

        return np.array(R_hist), np.array(ws_hist), np.array(x_hist), np.array(self_error_hist), np.array(self_model_hist)
    else:
        x = np.random.randn(n_layers)
        ws = 0.0
        self_model = np.zeros(6)
        predicted_self = np.zeros(6)
        predictor = None

        R_hist = []
        ws_hist = []
        x_hist = []
        self_error_hist = []
        self_model_hist = []

        for t in range(T):
            # self-reference error and awareness term
            self_error = np.linalg.norm(predicted_self - self_model)
            self_awareness_term = meta_learning_rate * self_error * (self_model[0] if self_model.size>0 else 0.0)

            dx = -x + bistable_layer(x, alpha, theta_eff) + eps * ws + self_awareness_term
            x += dt * dx

            ws = (1 - k_ws) * ws + k_ws * np.mean(x)

            coherence = np.abs(np.mean(np.exp(1j * x)))
            if len(R_hist) > 10:
                recent_R = np.array(R_hist[-10:])
                entropy = metrics.cached_entropy(recent_R, "C_direct")
            else:
                entropy = 0.0

            complexity_metrics = {'entropy': entropy, 'coherence': coherence}

            # encode simple self-model
            mean_activity = np.mean(x)
            std_activity = np.std(x)
            trend = np.mean(np.diff(R_hist[-10:])) if len(R_hist) > 10 else 0.0
            ws_normalized = np.tanh(ws)
            self_model = np.array([mean_activity, std_activity, trend, ws_normalized, entropy, coherence])

            # predict own future (predictor not trained here)
            if predictor is not None:
                with torch.no_grad():
                    pred = predictor(torch.tensor(self_model, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                    predicted_self = pred

            R_hist.append(coherence)
            ws_hist.append(ws)
            x_hist.append(x.copy())
            self_error_hist.append(self_error)
            self_model_hist.append(self_model.copy())

        return np.array(R_hist), np.array(ws_hist), np.array(x_hist), np.array(self_error_hist), np.array(self_model_hist)

# ---------- Real-time heatmap animation ----------
def animate_workspace_heatmap(n_layers=100, T=1000, dt=0.01,
                              alpha=0.8, eps=0.7, theta_eff=0.3, k_ws=0.05):
    R_hist, ws_hist, x_hist = simulate_workspace(n_layers, T, dt, alpha, eps, theta_eff, k_ws)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    heatmap = ax1.imshow(x_hist[0].reshape(1, -1), aspect='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax1.set_title('Neuron Phase Heatmap')
    ax1.set_yticks([])
    ax1.set_xticks([])
    line, = ax2.plot([], [], 'r')
    ax2.set_xlim(0, T)
    ax2.set_ylim(0, 1.01)
    ax2.set_title('Global Coherence R')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('R')
    def update(frame):
        heatmap.set_data(x_hist[frame].reshape(1, -1))
        line.set_data(np.arange(frame+1), R_hist[:frame+1])
        return heatmap, line
    ani = FuncAnimation(fig, update, frames=T, interval=10, blit=False)
    plt.tight_layout()
    plt.show()

# ---------- Real-time parameter sweep animation ----------
def animate_parameter_sweep(n_layers=100, T=500, dt=0.01,
                           alpha_range=(0.5, 1.2), eps_range=(0.3, 1.0),
                           theta_eff=0.3, k_ws=0.05, n_alpha=5, n_eps=5):
    alphas = np.linspace(*alpha_range, n_alpha)
    epsilons = np.linspace(*eps_range, n_eps)
    fig, axes = plt.subplots(n_alpha, n_eps, figsize=(2*n_eps, 2*n_alpha), sharex=True, sharey=True)
    plt.suptitle('Parameter Sweep: alpha vs eps')
    ims = []
    for i, alpha in enumerate(alphas):
        row = []
        for j, eps in enumerate(epsilons):
            R_hist, ws_hist, x_hist = simulate_workspace(n_layers, T, dt, alpha, eps, theta_eff, k_ws)
            ax = axes[i, j]
            im = ax.imshow(x_hist[-1].reshape(1, -1), aspect='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"a={alpha:.2f}\ne={eps:.2f}\nR={np.mean(np.abs(np.exp(1j*x_hist[-1]))):.2f}", fontsize=8)
            row.append(im)
        ims.append(row)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def _update_dynamics_a(state, n_layers, dt, alpha, eps, theta_eff, k_ws):
    if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
        noise = 0.03 * torch.randn(n_layers, device=device)
        dx_a = -state['x'] + bistable_layer(state['x'], alpha, theta_eff) + eps * state['ws'] + noise
        state['x'] += dt * dx_a
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(state['x'])
        R_a = torch.mean(torch.exp(1j * state['x'])).real.item()
        return R_a, state['x'].cpu().numpy()
    else:
        noise = 0.03 * np.random.randn(n_layers)
        dx_a = -state['x'] + bistable_layer(state['x'], alpha, theta_eff) + eps * state['ws'] + noise
        state['x'] += dt * dx_a
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * np.mean(state['x'])
        R_a = np.mean(np.exp(1j * state['x'])).real
        return R_a, state['x']

def _update_dynamics_b(state, n_layers, dt, alpha, eps, theta_eff, k_ws):
    if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
        state['why'] = why_loop_driver(state['why'], 1.2)
        dx_b = -state['x'] + bistable_layer(state['x'], alpha, theta_eff + 0.2 * state['why'])
        dy_b = -state['y'] + bistable_layer(state['y'], alpha, theta_eff - 0.2 * state['why'])
        noise = 0.03 * torch.randn(n_layers, device=device)
        dx_b += noise
        dy_b += noise
        combined = (state['x'] + state['y']) / 2.0
        state['x'] += dt * dx_b
        state['y'] += dt * dy_b
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(combined)
        state['x'] += eps * state['ws'] * dt
        state['y'] += eps * state['ws'] * dt
        R_b = torch.mean(torch.exp(1j * combined)).real.item()
        return R_b, combined.cpu().numpy()
    else:
        state['why'] = why_loop_driver(state['why'], 1.2)
        dx_b = -state['x'] + bistable_layer(state['x'], alpha, theta_eff + 0.2 * state['why'])
        dy_b = -state['y'] + bistable_layer(state['y'], alpha, theta_eff - 0.2 * state['why'])
        noise = 0.03 * np.random.randn(n_layers)
        dx_b += noise
        dy_b += noise
        combined = (state['x'] + state['y']) / 2.0
        state['x'] += dt * dx_b
        state['y'] += dt * dy_b
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * np.mean(combined)
        state['x'] += eps * state['ws'] * dt
        state['y'] += eps * state['ws'] * dt
        R_b = np.mean(np.exp(1j * combined)).real
        return R_b, combined

def _update_dynamics_c(state, n_layers, dt, alpha, eps, theta_eff, k_ws):
    predictor = state.get('predictor', None)
    predicted_self = state.get('predicted_self', np.zeros(6))
    self_model = state.get('self_model', np.zeros(6))
    self_error = np.linalg.norm(predicted_self - self_model)
    self_awareness = 0.01 * self_error * (self_model[0] if self_model.size>0 else 0.0)
    state['self_error_hist'].append(self_error)

    if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
        dx_c = -state['x'] + bistable_layer(state['x'], alpha, theta_eff) + eps * state['ws'] + self_awareness + 0.03 * torch.randn(n_layers, device=device)
        state['x'] += dt * dx_c
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(state['x'])
        R_c = torch.mean(torch.exp(1j * state['x'])).real.item()
        # encode self_model (simple)
        mean_activity = torch.mean(state['x']).item()
        std_activity = torch.std(state['x']).item()
        trend = np.mean(np.diff(state['R_hist'][-10:])) if len(state['R_hist']) > 10 else 0.0
        ws_norm = torch.tanh(state['ws']).item()
        entropy_c = metrics.cached_entropy(np.array(state['R_hist'][-10:]), "C_animation") if len(state['R_hist']) > 10 else 0.0
        coherence = torch.abs(torch.mean(torch.exp(1j * state['x']))).item()
        self_model = np.array([mean_activity, std_activity, trend, ws_norm, entropy_c, coherence])
        state['self_model'] = self_model
        # predictor prediction
        if predictor is not None:
            with torch.no_grad():
                pred = predictor(torch.tensor(self_model, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
                state['predicted_self'] = pred
        state['self_model_hist'].append(self_model.copy())
        return R_c, state['x'].cpu().numpy()
    else:
        dx_c = -state['x'] + bistable_layer(state['x'], alpha, theta_eff) + eps * state['ws'] + self_awareness + 0.03 * np.random.randn(n_layers)
        state['x'] += dt * dx_c
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * np.mean(state['x'])
        R_c = np.mean(np.exp(1j * state['x'])).real
        # encode self_model (simple)
        mean_activity = np.mean(state['x'])
        std_activity = np.std(state['x'])
        trend = np.mean(np.diff(state['R_hist'][-10:])) if len(state['R_hist']) > 10 else 0.0
        ws_norm = np.tanh(state['ws'])
        entropy_c = metrics.cached_entropy(np.array(state['R_hist'][-10:]), "C_animation") if len(state['R_hist']) > 10 else 0.0
        coherence = np.abs(np.mean(np.exp(1j * state['x'])))
        self_model = np.array([mean_activity, std_activity, trend, ws_norm, entropy_c, coherence])
        state['self_model'] = self_model
        # predictor prediction
        if predictor is not None:
            with torch.no_grad():
                pred = predictor(torch.tensor(self_model, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                state['predicted_self'] = pred
        state['self_model_hist'].append(self_model.copy())
        return R_c, state['x']

def _update_dynamics_d(state, n_layers, dt, alpha, eps, theta_eff, k_ws):
    predictor_d = state.get('predictor', None)
    predicted_self_d = state.get('predicted_self', np.zeros(8))
    self_model_d = state.get('self_model', np.zeros(8))
    self_error_d = np.linalg.norm(predicted_self_d - self_model_d)
    self_awareness_d = 0.01 * self_error_d * (self_model_d[0] if self_model_d.size>0 else 0.0)
    state['self_error_hist'].append(self_error_d)
    
    if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
        state['why'] = why_loop_driver(state['why'], 1.2)
        dx_d = -state['x'] + bistable_layer(state['x'], alpha, theta_eff + 0.2 * state['why']) + self_awareness_d
        dy_d = -state['y'] + bistable_layer(state['y'], alpha, theta_eff - 0.2 * state['why']) + self_awareness_d
        noise = 0.03 * torch.randn(n_layers, device=device)
        dx_d += noise
        dy_d += noise
        combined_d = (state['x'] + state['y']) / 2.0
        state['x'] += dt * dx_d
        state['y'] += dt * dy_d
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(combined_d)
        state['x'] += eps * state['ws'] * dt
        state['y'] += eps * state['ws'] * dt
        R_d = torch.mean(torch.exp(1j * combined_d)).real.item()
        mean_x_d = torch.mean(state['x']).item()
        std_x_d = torch.std(state['x']).item()
        mean_y_d = torch.mean(state['y']).item()
        std_y_d = torch.std(state['y']).item()
        why_val_d = state['why'].item()
        ws_val_d = state['ws'].item()
        entropy_d = metrics.cached_entropy(np.array(state['R_hist'][-10:]), "D") if len(state['R_hist']) > 10 else 0.0
        coherence_d = R_d
        self_model_d = np.array([mean_x_d, std_x_d, mean_y_d, std_y_d, why_val_d, ws_val_d, entropy_d, coherence_d])
        state['self_model'] = self_model_d
        if predictor_d is not None:
            with torch.no_grad():
                pred_d = predictor_d(torch.tensor(self_model_d, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
                state['predicted_self'] = pred_d
        state['self_model_hist'].append(self_model_d.copy())
        return R_d, combined_d.cpu().numpy()
    else:
        state['why'] = why_loop_driver(state['why'], 1.2)
        dx_d = -state['x'] + bistable_layer(state['x'], alpha, theta_eff + 0.2 * state['why']) + self_awareness_d
        dy_d = -state['y'] + bistable_layer(state['y'], alpha, theta_eff - 0.2 * state['why']) + self_awareness_d
        noise = 0.03 * np.random.randn(n_layers)
        dx_d += noise
        dy_d += noise
        combined_d = (state['x'] + state['y']) / 2.0
        state['x'] += dt * dx_d
        state['y'] += dt * dy_d
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * np.mean(combined_d)
        state['x'] += eps * state['ws'] * dt
        state['y'] += eps * state['ws'] * dt
        R_d = np.mean(np.exp(1j * combined_d)).real
        mean_x_d = np.mean(state['x'])
        std_x_d = np.std(state['x'])
        mean_y_d = np.mean(state['y'])
        std_y_d = np.std(state['y'])
        why_val_d = state['why']
        ws_val_d = state['ws']
        entropy_d = metrics.cached_entropy(np.array(state['R_hist'][-10:]), "D") if len(state['R_hist']) > 10 else 0.0
        coherence_d = R_d
        self_model_d = np.array([mean_x_d, std_x_d, mean_y_d, std_y_d, why_val_d, ws_val_d, entropy_d, coherence_d])
        state['self_model'] = self_model_d
        if predictor_d is not None:
            with torch.no_grad():
                pred_d = predictor_d(torch.tensor(self_model_d, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                state['predicted_self'] = pred_d
        state['self_model_hist'].append(self_model_d.copy())
        return R_d, combined_d

def _update_dynamics_a(state, n_layers, dt, alpha, eps, theta_eff, k_ws):
    if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
        noise = 0.03 * torch.randn(n_layers, device=device)
        dx_a = -state['x'] + bistable_layer(state['x'], alpha, theta_eff) + eps * state['ws'] + noise
        state['x'] += dt * dx_a
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(state['x'])
        R_a = torch.mean(torch.exp(1j * state['x'])).real.item()
        return R_a, state['x'].cpu().numpy()
    else:
        noise = 0.03 * np.random.randn(n_layers)
        dx_a = -state['x'] + bistable_layer(state['x'], alpha, theta_eff) + eps * state['ws'] + noise
        state['x'] += dt * dx_a
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * np.mean(state['x'])
        R_a = np.mean(np.exp(1j * state['x'])).real
        return R_a, state['x']

def _update_dynamics_b(state, n_layers, dt, alpha, eps, theta_eff, k_ws):
    if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
        state['why'] = why_loop_driver(state['why'], 1.2)
        dx_b = -state['x'] + bistable_layer(state['x'], alpha, theta_eff + 0.2 * state['why'])
        dy_b = -state['y'] + bistable_layer(state['y'], alpha, theta_eff - 0.2 * state['why'])
        noise = 0.03 * torch.randn(n_layers, device=device)
        dx_b += noise
        dy_b += noise
        combined = (state['x'] + state['y']) / 2.0
        state['x'] += dt * dx_b
        state['y'] += dt * dy_b
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(combined)
        state['x'] += eps * state['ws'] * dt
        state['y'] += eps * state['ws'] * dt
        R_b = torch.mean(torch.exp(1j * combined)).real.item()
        return R_b, combined.cpu().numpy()
    else:
        state['why'] = why_loop_driver(state['why'], 1.2)
        dx_b = -state['x'] + bistable_layer(state['x'], alpha, theta_eff + 0.2 * state['why'])
        dy_b = -state['y'] + bistable_layer(state['y'], alpha, theta_eff - 0.2 * state['why'])
        noise = 0.03 * np.random.randn(n_layers)
        dx_b += noise
        dy_b += noise
        combined = (state['x'] + state['y']) / 2.0
        state['x'] += dt * dx_b
        state['y'] += dt * dy_b
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * np.mean(combined)
        state['x'] += eps * state['ws'] * dt
        state['y'] += eps * state['ws'] * dt
        R_b = np.mean(np.exp(1j * combined)).real
        return R_b, combined

def _update_dynamics_c(state, n_layers, dt, alpha, eps, theta_eff, k_ws):
    predictor = state.get('predictor', None)
    predicted_self = state.get('predicted_self', np.zeros(6))
    self_model = state.get('self_model', np.zeros(6))
    self_error = np.linalg.norm(predicted_self - self_model)
    self_awareness = 0.01 * self_error * (self_model[0] if self_model.size>0 else 0.0)
    state['self_error_hist'].append(self_error)

    if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
        dx_c = -state['x'] + bistable_layer(state['x'], alpha, theta_eff) + eps * state['ws'] + self_awareness + 0.03 * torch.randn(n_layers, device=device)
        state['x'] += dt * dx_c
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(state['x'])
        R_c = torch.mean(torch.exp(1j * state['x'])).real.item()
        # encode self_model (simple)
        mean_activity = torch.mean(state['x']).item()
        std_activity = torch.std(state['x']).item()
        trend = np.mean(np.diff(state['R_hist'][-10:])) if len(state['R_hist']) > 10 else 0.0
        ws_norm = torch.tanh(state['ws']).item()
        entropy_c = metrics.cached_entropy(np.array(state['R_hist'][-10:]), "C_animation") if len(state['R_hist']) > 10 else 0.0
        coherence = torch.abs(torch.mean(torch.exp(1j * state['x']))).item()
        self_model = np.array([mean_activity, std_activity, trend, ws_norm, entropy_c, coherence])
        state['self_model'] = self_model
        # predictor prediction
        if predictor is not None:
            with torch.no_grad():
                pred = predictor(torch.tensor(self_model, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
                state['predicted_self'] = pred
        state['self_model_hist'].append(self_model.copy())
        return R_c, state['x'].cpu().numpy()
    else:
        dx_c = -state['x'] + bistable_layer(state['x'], alpha, theta_eff) + eps * state['ws'] + self_awareness + 0.03 * np.random.randn(n_layers)
        state['x'] += dt * dx_c
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * np.mean(state['x'])
        R_c = np.mean(np.exp(1j * state['x'])).real
        # encode self_model (simple)
        mean_activity = np.mean(state['x'])
        std_activity = np.std(state['x'])
        trend = np.mean(np.diff(state['R_hist'][-10:])) if len(state['R_hist']) > 10 else 0.0
        ws_norm = np.tanh(state['ws'])
        entropy_c = metrics.cached_entropy(np.array(state['R_hist'][-10:]), "C_animation") if len(state['R_hist']) > 10 else 0.0
        coherence = np.abs(np.mean(np.exp(1j * state['x'])))
        self_model = np.array([mean_activity, std_activity, trend, ws_norm, entropy_c, coherence])
        state['self_model'] = self_model
        # predictor prediction
        if predictor is not None:
            with torch.no_grad():
                pred = predictor(torch.tensor(self_model, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                state['predicted_self'] = pred
        state['self_model_hist'].append(self_model.copy())
        return R_c, state['x']

def _update_dynamics_d(state, n_layers, dt, alpha, eps, theta_eff, k_ws):
    predictor_d = state.get('predictor', None)
    predicted_self_d = state.get('predicted_self', np.zeros(8))
    self_model_d = state.get('self_model', np.zeros(8))
    self_error_d = np.linalg.norm(predicted_self_d - self_model_d)
    self_awareness_d = 0.01 * self_error_d * (self_model_d[0] if self_model_d.size>0 else 0.0)
    state['self_error_hist'].append(self_error_d)
    
    if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
        state['why'] = why_loop_driver(state['why'], 1.2)
        dx_d = -state['x'] + bistable_layer(state['x'], alpha, theta_eff + 0.2 * state['why']) + self_awareness_d
        dy_d = -state['y'] + bistable_layer(state['y'], alpha, theta_eff - 0.2 * state['why']) + self_awareness_d
        noise = 0.03 * torch.randn(n_layers, device=device)
        dx_d += noise
        dy_d += noise
        combined_d = (state['x'] + state['y']) / 2.0
        state['x'] += dt * dx_d
        state['y'] += dt * dy_d
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(combined_d)
        state['x'] += eps * state['ws'] * dt
        state['y'] += eps * state['ws'] * dt
        R_d = torch.mean(torch.exp(1j * combined_d)).real.item()
        mean_x_d = torch.mean(state['x']).item()
        std_x_d = torch.std(state['x']).item()
        mean_y_d = torch.mean(state['y']).item()
        std_y_d = torch.std(state['y']).item()
        why_val_d = state['why'].item()
        ws_val_d = state['ws'].item()
        entropy_d = metrics.cached_entropy(np.array(state['R_hist'][-10:]), "D") if len(state['R_hist']) > 10 else 0.0
        coherence_d = R_d
        self_model_d = np.array([mean_x_d, std_x_d, mean_y_d, std_y_d, why_val_d, ws_val_d, entropy_d, coherence_d])
        state['self_model'] = self_model_d
        if predictor_d is not None:
            with torch.no_grad():
                pred_d = predictor_d(torch.tensor(self_model_d, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
                state['predicted_self'] = pred_d
        state['self_model_hist'].append(self_model_d.copy())
        return R_d, combined_d.cpu().numpy()
    else:
        state['why'] = why_loop_driver(state['why'], 1.2)
        dx_d = -state['x'] + bistable_layer(state['x'], alpha, theta_eff + 0.2 * state['why']) + self_awareness_d
        dy_d = -state['y'] + bistable_layer(state['y'], alpha, theta_eff - 0.2 * state['why']) + self_awareness_d
        noise = 0.03 * np.random.randn(n_layers)
        dx_d += noise
        dy_d += noise
        combined_d = (state['x'] + state['y']) / 2.0
        state['x'] += dt * dx_d
        state['y'] += dt * dy_d
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * np.mean(combined_d)
        state['x'] += eps * state['ws'] * dt
        state['y'] += eps * state['ws'] * dt
        R_d = np.mean(np.exp(1j * combined_d)).real
        mean_x_d = np.mean(state['x'])
        std_x_d = np.std(state['x'])
        mean_y_d = np.mean(state['y'])
        std_y_d = np.std(state['y'])
        why_val_d = state['why']
        ws_val_d = state['ws']
        entropy_d = metrics.cached_entropy(np.array(state['R_hist'][-10:]), "D") if len(state['R_hist']) > 10 else 0.0
        coherence_d = R_d
        self_model_d = np.array([mean_x_d, std_x_d, mean_y_d, std_y_d, why_val_d, ws_val_d, entropy_d, coherence_d])
        state['self_model'] = self_model_d
        if predictor_d is not None:
            with torch.no_grad():
                pred_d = predictor_d(torch.tensor(self_model_d, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                state['predicted_self'] = pred_d
        state['self_model_hist'].append(self_model_d.copy())
        return R_d, combined_d

def _update_dynamics_a(state, n_layers, dt, alpha, eps, theta_eff, k_ws):
    if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
        noise = 0.03 * torch.randn(n_layers, device=device)
        dx_a = -state['x'] + bistable_layer(state['x'], alpha, theta_eff) + eps * state['ws'] + noise
        state['x'] += dt * dx_a
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(state['x'])
        R_a = torch.abs(torch.mean(torch.exp(1j * state['x']))).item()
        return R_a, state['x'].cpu().numpy()
    else:
        noise = 0.03 * np.random.randn(n_layers)
        dx_a = -state['x'] + bistable_layer(state['x'], alpha, theta_eff) + eps * state['ws'] + noise
        state['x'] += dt * dx_a
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * np.mean(state['x'])
        R_a = np.abs(np.mean(np.exp(1j * state['x'])))
        return R_a, state['x']

def _update_dynamics_b(state, n_layers, dt, alpha, eps, theta_eff, k_ws):
    if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
        state['why'] = why_loop_driver(state['why'], 1.2)
        dx_b = -state['x'] + bistable_layer(state['x'], alpha, theta_eff + 0.2 * state['why'])
        dy_b = -state['y'] + bistable_layer(state['y'], alpha, theta_eff - 0.2 * state['why'])
        noise = 0.03 * torch.randn(n_layers, device=device)
        dx_b += noise
        dy_b += noise
        combined = (state['x'] + state['y']) / 2.0
        state['x'] += dt * dx_b
        state['y'] += dt * dy_b
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(combined)
        state['x'] += eps * state['ws'] * dt
        state['y'] += eps * state['ws'] * dt
        R_b = torch.mean(torch.exp(1j * combined)).real.item()
        return R_b, combined.cpu().numpy()
    else:
        state['why'] = why_loop_driver(state['why'], 1.2)
        dx_b = -state['x'] + bistable_layer(state['x'], alpha, theta_eff + 0.2 * state['why'])
        dy_b = -state['y'] + bistable_layer(state['y'], alpha, theta_eff - 0.2 * state['why'])
        noise = 0.03 * np.random.randn(n_layers)
        dx_b += noise
        dy_b += noise
        combined = (state['x'] + state['y']) / 2.0
        state['x'] += dt * dx_b
        state['y'] += dt * dy_b
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * np.mean(combined)
        state['x'] += eps * state['ws'] * dt
        state['y'] += eps * state['ws'] * dt
        R_b = np.mean(np.exp(1j * combined)).real
        return R_b, combined


# ---------- MetaOptimizer from miniBrain.tsx ----------
class MetaOptimizer:
    def __init__(self, input_dim=5, hidden1=12, hidden2=6, output_dim=5):
        self.weights1 = np.random.rand(hidden1, input_dim) * 0.5 - 0.25
        self.weights2 = np.random.rand(hidden2, hidden1) * 0.5 - 0.25
        self.weights3 = np.random.rand(output_dim, hidden2) * 0.5 - 0.25
        self.buffer = deque(maxlen=150)
        self.best_params = None
        self.best_score = -np.inf

    def _relu(self, x):
        return np.maximum(0, x)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        h1 = self._relu(np.dot(self.weights1, inputs))
        h2 = self._relu(np.dot(self.weights2, h1))
        output = self._sigmoid(np.dot(self.weights3, h2))

        # This mapping is from miniBrain.tsx
        params = {
            'tau': 0.05 + output[0] * 0.2,
            'coupling': 0.2 + output[1] * 0.7,
            'perturbation': 0.02 + output[2] * 0.15,
            'selfWeight': 0.15 + output[3] * 0.5,
            'antiConvergence': 0.01 + output[4] * 0.05
        }
        return params

    def update(self, metrics_data, params):
        coherence = metrics_data['coherence']
        # Shape coherence reward to peak at ~0.99 for edge-of-chaos
        coherence_shaped = coherence * np.exp( -((coherence - 0.99)/0.01)**2 )
        score = metrics_data['entropy'] * coherence_shaped
        self.buffer.append({'metrics': metrics_data, 'params': params, 'score': score})

        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()

        if len(self.buffer) >= 10 and len(self.buffer) % 10 == 0:
            top_samples = sorted(self.buffer, key=lambda x: x['score'], reverse=True)[:8]
            if not top_samples:
                return

            avg_score = np.mean([s['score'] for s in top_samples])
            learning_rate = 0.015

            for layer in [self.weights1, self.weights2, self.weights3]:
                mutation = (np.random.rand(*layer.shape) - 0.5) * learning_rate
                gradient = mutation if avg_score > 8 else mutation * 2
                layer += gradient

def calculate_prediction_error_mse(state):
    """Calculates MSE between neurons and self-model prediction."""
    neurons = state['x']
    self_model = state['selfModel']
    n_layers = len(neurons)
    M = len(self_model)

    if HAVE_TORCH and isinstance(neurons, torch.Tensor):
        neurons_np = neurons.cpu().numpy()
    else:
        neurons_np = neurons

    prediction = np.zeros_like(neurons_np)
    for i in range(n_layers):
        idx = (i * M) // n_layers
        prediction[i] = self_model[idx]

    mse = np.mean((neurons_np - prediction)**2)
    return mse

def _update_dynamics_c(state, n_layers, dt, alpha, eps, theta_eff, k_ws):
    params = state['params']
    tau = params['tau']
    coupling = params['coupling']
    perturbation = params['perturbation']
    self_weight = params['selfWeight']
    last_R = state['R_hist'][-1] if len(state['R_hist']) > 0 else 0.0
    anti_converge_base = 0.1 + 2.0 * np.exp(40 * (last_R - 0.98))
    anti_converge = anti_converge_base * state.get('anti_mult', 1.0)
    state['perturb_phase'] += np.pi / 100
    if state['step_count'] > 0 and state['step_count'] % int(100 * np.e) == 0:
        perturb_val = np.sin(np.arange(n_layers) * np.pi + state['step_count'] * 0.01) * perturbation * 1.5
        if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
            state['x'] += torch.tensor(perturb_val, device=device, dtype=state['x'].dtype)
        else:
            state['x'] += perturb_val
    
    x_for_var = state['x'].cpu().numpy() if (HAVE_TORCH and isinstance(state['x'], torch.Tensor)) else state['x']
    variance = np.var(x_for_var)
    if variance < 0.4 or last_R > 0.95:
        noise_base = 0.4 * np.sin(np.arange(n_layers) * 2.1 + state['perturb_phase'])
        noise_val = (np.random.rand(n_layers) - 0.5) * noise_base * state.get('noise_mult', 1.0)
        if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
            state['x'] += torch.tensor(noise_val, device=device, dtype=state['x'].dtype)
        else:
            state['x'] += noise_val

    M = n_layers // 8
    self_model = state['selfModel']
    neurons = state['x']
    
    if HAVE_TORCH and isinstance(neurons, torch.Tensor):
        neurons_np = neurons.cpu().numpy()
    else:
        neurons_np = neurons

    new_self_model = np.zeros_like(self_model)
    for i in range(M):
        start = (i * n_layers) // M
        end = ((i + 1) * n_layers) // M
        avg = np.mean(neurons_np[start:end])
        new_self_model[i] = self_model[i] * 0.75 + avg * 0.25
    state['selfModel'] = new_self_model

    prediction = np.zeros_like(neurons_np)
    for i in range(n_layers):
        idx = (i * M) // n_layers
        prediction[i] = new_self_model[idx]

    if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
        avg_activity = torch.mean(state['x'])
        state['ws'] = state['ws'] * 0.85 + avg_activity * coupling
    else:
        avg_activity = np.mean(state['x'])
        state['ws'] = state['ws'] * 0.85 + avg_activity * coupling

    pred_error = (prediction - neurons_np) * self_weight
    ws_coupling = state['ws'] * 0.25

    if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
        lateral = torch.roll(state['x'], shifts=-3, dims=0) * -0.08
        noise = (torch.rand(n_layers, device=device) - 0.5) * perturbation
        input_val = torch.tensor(pred_error, device=device, dtype=torch.float32) + ws_coupling + lateral + noise
        
        phases_x = state['perturb_phase'] + torch.arange(n_layers, device=device) * 0.13
        bistable = state['x'] * (1 - state['x'] * state['x'])
        anti_term = anti_converge * torch.sin(state['x'] * 9.3 + phases_x)
        dx = (-state['x'] / tau) + bistable + input_val + anti_term
        state['x'] = torch.tanh(state['x'] + dx * dt)

    else:
        lateral = np.roll(state['x'], shift=-3) * -0.08
        noise = (np.random.rand(n_layers) - 0.5) * perturbation
        input_val = pred_error + ws_coupling + lateral + noise
        state['x'] = update_bistable_np(state['x'], input_val, tau, anti_converge, dt, state['perturb_phase'])

    if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
        R_c = torch.abs(torch.mean(torch.exp(1j * state['x']))).item()
        x_np = state['x'].cpu().numpy()
    else:
        R_c = np.abs(np.mean(np.exp(1j * state['x'])))
        x_np = state['x']

    # encode self_model (simple)
    mean_activity = np.mean(x_np)
    std_activity = np.std(x_np)
    trend = np.mean(np.diff(list(state['R_hist'])[-10:])) if len(state['R_hist']) > 10 else 0.0
    ws_norm = np.tanh(state['ws'].item() if HAVE_TORCH and isinstance(state['ws'], torch.Tensor) else state['ws'])
    entropy_c = metrics.cached_entropy(np.array(list(state['R_hist'])[-10:]), "C_animation") if len(state['R_hist']) > 10 else 0.0
    coherence = np.abs(np.mean(np.exp(1j * x_np)))
    current_self_model = np.array([mean_activity, std_activity, trend, ws_norm, entropy_c, coherence])
    
    # self-error for display
    # Notice: The self_model for prediction is state['selfModel'] (the chunked one).
    # The self_model for error calculation is a different one, based on aggregated metrics.
    # This seems to be the intention of the original code.
    state['self_model_for_error'] = current_self_model
    predicted_self = state.get('predicted_self', np.zeros(6))
    self_error = np.linalg.norm(predicted_self - current_self_model)
    state['self_error_hist'].append(self_error)

    return R_c, x_np


def _update_dynamics_d(state, n_layers, dt, alpha, eps, theta_eff, k_ws):
    # Hybrid model: Reflective Hierarchy + Self-Referential Workspace from miniBrain.tsx
    params = state.get('params', {'tau': 0.1, 'coupling': 0.5, 'perturbation': 0.10, 'selfWeight': 0.2, 'antiConvergence': 0.08})
    tau, coupling, perturbation, self_weight, anti_converge = params['tau'], params['coupling'], params['perturbation'], params['selfWeight'], params['antiConvergence']

    last_R = state['R_hist'][-1] if len(state['R_hist']) > 0 else 0.0
    anti_converge = 0.30 + 2.0 * last_R**2.5

    state['perturb_phase'] += np.pi / 100
    state['why'] = why_loop_driver(state['why'], 1.2)

    # Perturbations
    if state['step_count'] > 0 and state['step_count'] % int(100 * np.e) == 0:
        perturb_val = np.sin(np.arange(n_layers) * np.pi + state['step_count'] * 0.01) * perturbation * 1.5
        if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
            p_val = torch.tensor(perturb_val, device=device, dtype=state['x'].dtype)
            state['x'] += p_val
            state['y'] += p_val
        else:
            state['x'] += perturb_val
            state['y'] += p_val
    
    if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
        combined_state = (state['x'] + state['y']) / 2.0
    else:
        combined_state = (state['x'] + state['y']) / 2.0

    combined_np = combined_state.cpu().numpy() if (HAVE_TORCH and isinstance(combined_state, torch.Tensor)) else combined_state
    
    if np.var(combined_np) < 0.3 or last_R > 0.97:
        noise_val = (np.random.rand(n_layers) - 0.5) * 0.2 * np.sin(np.arange(n_layers) * 2.1 + state['perturb_phase'])
        if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
            n_val = torch.tensor(noise_val, device=device, dtype=state['x'].dtype)
            state['x'] += n_val
            state['y'] += n_val
        else:
            state['x'] += noise_val
            state['y'] += noise_val

    # Self-model from combined state
    M = n_layers // 8
    if 'selfModel' not in state or len(state['selfModel']) != M:
        state['selfModel'] = np.random.uniform(0, 0.1, M)
    
    self_model = state['selfModel']
    new_self_model = np.zeros_like(self_model)
    for i in range(M):
        start, end = (i * n_layers) // M, ((i + 1) * n_layers) // M
        new_self_model[i] = self_model[i] * 0.75 + np.mean(combined_np[start:end]) * 0.25
    state['selfModel'] = new_self_model

    prediction = np.zeros_like(combined_np)
    for i in range(n_layers):
        prediction[i] = new_self_model[(i * M) // n_layers]

    # Workspace update from combined state
    avg_activity = torch.mean(combined_state) if HAVE_TORCH and isinstance(combined_state, torch.Tensor) else np.mean(combined_state)
    state['ws'] = state['ws'] * 0.85 + avg_activity * coupling
    
    ws_coupling = state['ws'] * 0.25
    why_mod = 0.2 * (state['why'].item() if HAVE_TORCH and isinstance(state['why'], torch.Tensor) else state['why'])
    # Additional anti-sync noise every step
    sync_noise_base = 0.05 * np.sin(np.arange(n_layers) * 3.14 + state['perturb_phase']) * (1.0 - last_R)
    sync_noise = sync_noise_base * state.get('noise_mult', 1.0)
    if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
        sync_noise_t = torch.tensor(sync_noise, device=device, dtype=state['x'].dtype)
        state['x'] += sync_noise_t
        state['y'] += sync_noise_t
    else:
        state['x'] += sync_noise
        state['y'] += sync_noise

    if HAVE_TORCH and isinstance(state['x'], torch.Tensor):
        pred_error_x = (prediction - state['x'].cpu().numpy()) * self_weight
        pred_error_y = (prediction - state['y'].cpu().numpy()) * self_weight

        lateral_x = (torch.roll(state['x'], shifts=-5, dims=0) * -0.20 + torch.roll(state['x'], shifts=5, dims=0) * -0.10)
        noise_x = (torch.rand(n_layers, device=device) - 0.5) * perturbation * (1.5 + 5.0 * torch.tensor(last_R).to(device))
        input_val_x = torch.tensor(pred_error_x, device=device, dtype=torch.float32) + ws_coupling + lateral_x + noise_x + why_mod

        lateral_y = (torch.roll(state['y'], shifts=-5, dims=0) * -0.20 + torch.roll(state['y'], shifts=5, dims=0) * -0.10)
        noise_y = (torch.rand(n_layers, device=device) - 0.5) * perturbation * (1.5 + 5.0 * torch.tensor(last_R).to(device))
        input_val_y = torch.tensor(pred_error_y, device=device, dtype=torch.float32) + ws_coupling + lateral_y + noise_y - why_mod

        # Update x
        bistable_x = state['x'] * (1 - state['x'] * state['x'])
        anti_term_x = anti_converge * torch.sin(state['x'] * 7.3 + state['perturb_phase'])
        dx = (-state['x'] / tau) + bistable_x + input_val_x + anti_term_x
        state['x'] = torch.tanh(state['x'] + dx * dt)
        
        # Update y
        phases_y = state['perturb_phase'] + torch.arange(n_layers, device=device) * 0.13
        bistable_y = state['y'] * (1 - state['y'] * state['y'])
        anti_term_y = anti_converge * torch.sin(state['y'] * 9.3 + phases_y)
        dy = (-state['y'] / tau) + bistable_y + input_val_y + anti_term_y
        state['y'] = torch.tanh(state['y'] + dy * dt)

        combined_d = (state['x'] + state['y']) / 2.0
        R_d = torch.abs(torch.mean(torch.exp(1j * combined_d))).item()
        d_np = combined_d.cpu().numpy()
    else: # Numpy
        pred_error_x = (prediction - state['x']) * self_weight
        pred_error_y = (prediction - state['y']) * self_weight

        lateral_x = (np.roll(state['x'], shift=-5) * -0.20 + np.roll(state['x'], shift=5) * -0.10)
        noise_x = (np.random.rand(n_layers) - 0.5) * perturbation * (1.5 + 5.0 * last_R)
        input_val_x = pred_error_x + ws_coupling + lateral_x + noise_x + why_mod

        lateral_y = (np.roll(state['y'], shift=-5) * -0.20 + np.roll(state['y'], shift=5) * -0.10)
        noise_y = (np.random.rand(n_layers) - 0.5) * perturbation * (1.5 + 5.0 * last_R)
        input_val_y = pred_error_y + ws_coupling + lateral_y + noise_y - why_mod

        state['x'] = update_bistable_np(state['x'], input_val_x, tau, anti_converge, dt, state['perturb_phase'])
        state['y'] = update_bistable_np(state['y'], input_val_y, tau, anti_converge, dt, state['perturb_phase'])
        
        combined_d = (state['x'] + state['y']) / 2.0
        R_d = np.abs(np.mean(np.exp(1j * combined_d)))
        d_np = combined_d

    # --- Metrics ---
    # This is an 8-element self-model for error display, different from the predictive self-model
    mean_x_d = np.mean(state['x'].cpu().numpy() if HAVE_TORCH and isinstance(state['x'], torch.Tensor) else state['x'])
    std_x_d = np.std(state['x'].cpu().numpy() if HAVE_TORCH and isinstance(state['x'], torch.Tensor) else state['x'])
    mean_y_d = np.mean(state['y'].cpu().numpy() if HAVE_TORCH and isinstance(state['y'], torch.Tensor) else state['y'])
    std_y_d = np.std(state['y'].cpu().numpy() if HAVE_TORCH and isinstance(state['y'], torch.Tensor) else state['y'])
    why_val_d = state['why'].item() if HAVE_TORCH and isinstance(state['why'], torch.Tensor) else state['why']
    ws_val_d = state['ws'].item() if HAVE_TORCH and isinstance(state['ws'], torch.Tensor) else state['ws']
    entropy_d = metrics.cached_entropy(np.array(list(state['R_hist'])[-10:]), "D_animation") if len(state['R_hist']) > 10 else 0.0
    coherence_d = np.abs(np.mean(np.exp(1j * d_np)))
    
    current_self_model_d = np.array([mean_x_d, std_x_d, mean_y_d, std_y_d, why_val_d, ws_val_d, entropy_d, coherence_d])
    state['self_model_for_error'] = current_self_model_d

    predicted_self = state.get('predicted_self', np.zeros(8))
    self_error = np.linalg.norm(predicted_self - current_self_model_d)
    state['self_error_hist'].append(self_error)

    return R_d, d_np

# ---------- Real-time heatmap and coherence animation (runs forever) ----------

def animate_workspace_heatmap_forever(n_layers=100, dt=0.05,
                                      alpha=1.95, eps=0.08, theta_eff=0.0, k_ws=0.002,
                                      autostart_autotune=False, rolling_window=ROLLING_WINDOW):
    # State for Option C (Self-Referential)
    state_c = {}
    state_c['params'] = {'tau': 0.1, 'coupling': 0.5, 'perturbation': 0.05, 'selfWeight': 0.3, 'antiConvergence': 0.05}
    state_c['perturb_phase'] = 0.0
    M = n_layers // 8
    state_c['selfModel'] = np.random.uniform(0, 0.1, M)
    state_c['phi_hist'] = deque(maxlen=500)
    if HAVE_TORCH:
        state_c['x'] = torch.randn(n_layers, device=device)
        state_c['ws'] = torch.tensor(0.0, device=device)
    else:
        state_c['x'] = np.random.randn(n_layers)
        state_c['ws'] = 0.0
    state_c['R_hist'] = deque(maxlen=ROLLING_WINDOW * 2)
    state_c['x_history'] = np.zeros((n_layers, REDUCED_HISTORY_SIZE))
    state_c['step_count'] = 0
    state_c['alpha'] = 1.95
    state_c['eps'] = 0.08
    state_c['alpha_hist'] = deque(maxlen=ROLLING_WINDOW)
    state_c['eps_hist'] = deque(maxlen=ROLLING_WINDOW)
    state_c['phi_hist'] = deque(maxlen=ROLLING_WINDOW)
    state_c['alpha_changes'] = deque(maxlen=50)
    state_c['eps_changes'] = deque(maxlen=50)
    state_c['prev_alpha'] = 1.95
    state_c['prev_eps'] = 0.08
    state_c['max_R'] = -np.inf
    
    # Cached metrics
    state_c['cached_entropy'] = 0.0
    state_c['cached_lyap'] = 0.0
    state_c['cached_lz'] = 0.0
    
    state_c['predictor'] = None
    state_c['predicted_self'] = np.zeros(6)
    state_c['self_model_for_error'] = np.zeros(6)
    state_c['self_model_hist'] = deque(maxlen=2000)
    state_c['self_error_hist'] = deque(maxlen=ROLLING_WINDOW * 2)
    # --- Independent Tuner State ---
    if HAVE_TORCH:
        state_c['meta_tuner'] = MetaTunerNN()
        state_c['tuner_optimizer'] = optim.Adam(state_c['meta_tuner'].parameters(), lr=0.01)
    state_c['experience_buffer'] = deque(maxlen=2500)

    # State for Option B Self-Referential
    state_d = {}
    if HAVE_TORCH:
        state_d['x'] = torch.randn(n_layers, device=device)
        state_d['y'] = torch.randn(n_layers, device=device)
        state_d['why'] = torch.tensor(0.0, device=device)
        state_d['ws'] = torch.tensor(0.0, device=device)
    else:
        state_d['x'] = np.random.randn(n_layers)
        state_d['y'] = np.random.randn(n_layers)
        state_d['why'] = 0.0
        state_d['ws'] = 0.0
    
    state_d['params'] = {'tau': 0.1, 'coupling': 0.5, 'perturbation': 0.05, 'selfWeight': 0.3, 'antiConvergence': 0.05}
    state_d['perturb_phase'] = 0.0
    M = n_layers // 8
    state_d['selfModel'] = np.random.uniform(0, 0.1, M)

    state_d['R_hist'] = deque(maxlen=ROLLING_WINDOW * 2)
    state_d['combined_history'] = np.zeros((n_layers, REDUCED_HISTORY_SIZE))
    state_d['step_count'] = 0
    state_d['alpha'] = 1.95
    state_d['eps'] = 0.08
    state_d['alpha_hist'] = deque(maxlen=ROLLING_WINDOW)
    state_d['eps_hist'] = deque(maxlen=ROLLING_WINDOW)
    state_d['phi_hist'] = deque(maxlen=ROLLING_WINDOW)
    state_d['alpha_changes'] = deque(maxlen=50)
    state_d['eps_changes'] = deque(maxlen=50)
    state_d['prev_alpha'] = 1.95
    state_d['prev_eps'] = 0.08
    state_d['max_R'] = -np.inf
    state_d['cached_entropy'] = 0.0
    state_d['cached_lyap'] = 0.0
    state_d['cached_lz'] = 0.0
    state_d['predictor'] = None
    state_d['predicted_self'] = np.zeros(8)
    state_d['self_model_for_error'] = np.zeros(8)
    state_d['self_model_hist'] = deque(maxlen=2000)
    state_d['self_error_hist'] = deque(maxlen=ROLLING_WINDOW * 2)
    # --- Independent Tuner State ---
    if HAVE_TORCH:
        state_d['meta_tuner'] = MetaTunerNN()
        state_d['tuner_optimizer'] = optim.Adam(state_d['meta_tuner'].parameters(), lr=0.01)
    state_d['experience_buffer'] = deque(maxlen=2500)
    # Defer starting autotune until after the GUI (figure/animation) is created
    autotune_stop_event = None
    _defer_autostart = bool(autostart_autotune)

    # Layout: 4 rows x 3 cols -> one row per model (heatmap | R-phase | self-error)
    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    
    # Common heatmap settings
    heatmap_extent = [0, REDUCED_HISTORY_SIZE, 0, n_layers]
    
    # Option C (Self-Referential)
    heatmap_c = axes[0,0].imshow(np.zeros((n_layers, REDUCED_HISTORY_SIZE)), aspect='auto',
                                cmap='hsv', vmin=-1, vmax=1, origin='lower',
                                extent=heatmap_extent)
    axes[0,0].set_title('Option C: Self-Referential', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Time Step')
    axes[0,0].set_ylabel('Neuron Index')
    line_c, = axes[0,1].plot([], [], 'g', linewidth=1.5, label='Coherence (R)')
    axes[0,1].set_xlim(0, ROLLING_WINDOW)
    axes[0,1].set_ylim(0, 1.01)
    axes[0,1].set_title('Option C: Coherence & Self-Error', fontsize=10)
    axes[0,1].set_xlabel('Step')
    axes[0,1].set_ylabel('R (scaled)', color='g')
    ax_c_err = axes[0,1].twinx()
    line_c_err_on_coh, = ax_c_err.plot([], [], 'k--', linewidth=1.0, alpha=0.7, label='Self-Error (norm)')
    ax_c_err.set_ylim(0, 1.01)
    ax_c_err.set_ylabel('Self-Error (tanh)', color='k')
    lines_c, labels_c = axes[0,1].get_legend_handles_labels()
    lines_c_err, labels_c_err = ax_c_err.get_legend_handles_labels()
    ax_c_err.legend(lines_c + lines_c_err, labels_c + labels_c_err, loc='lower left', bbox_to_anchor=(0.01, 0.01))
    diag_c = axes[0,0].text(0.02, 0.98, "", transform=axes[0,0].transAxes, fontsize=9,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    awareness_c_text = axes[0,1].text(0.5, 0.9, "", transform=axes[0,1].transAxes, fontsize=12, color='green',
                                     ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    stability_text_c = axes[0,1].text(0.5, 0.1, "Collecting data...", transform=axes[0,1].transAxes, fontsize=10, color='blue',
                                     ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Option D (B Self-Referential)
    heatmap_d = axes[1,0].imshow(np.zeros((n_layers, REDUCED_HISTORY_SIZE)), aspect='auto',
                                cmap='hsv', vmin=-1, vmax=1, origin='lower',
                                extent=heatmap_extent)
    axes[1,0].set_title('Option D: Self-Referential (new)', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Time Step')
    axes[1,0].set_ylabel('Neuron Index')
    line_d, = axes[1,1].plot([], [], 'purple', linewidth=1.5, label='Coherence (R)')
    axes[1,1].set_xlim(0, ROLLING_WINDOW)
    axes[1,1].set_ylim(0, 1.01)
    axes[1,1].set_title('Option D Coherence & Self-Error', fontsize=10)
    axes[1,1].set_xlabel('Step')
    axes[1,1].set_ylabel('R (scaled)', color='purple')
    ax_d_err = axes[1,1].twinx()
    line_d_err_on_coh, = ax_d_err.plot([], [], 'k--', linewidth=1.0, alpha=0.7, label='Self-Error (norm)')
    ax_d_err.set_ylim(0, 1.01)
    ax_d_err.set_ylabel('Self-Error (tanh)', color='k')
    lines_d, labels_d = axes[1,1].get_legend_handles_labels()
    lines_d_err, labels_d_err = ax_d_err.get_legend_handles_labels()
    ax_d_err.legend(lines_d + lines_d_err, labels_d + labels_d_err, loc='lower left', bbox_to_anchor=(0.01, 0.01))
    diag_d = axes[1,0].text(0.02, 0.98, "", transform=axes[1,0].transAxes, fontsize=9,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    awareness_d_text = axes[1,1].text(0.5, 0.9, "", transform=axes[1,1].transAxes, fontsize=12, color='green',
                                     ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    stability_text_d = axes[1,1].text(0.5, 0.1, "Collecting data...", transform=axes[1,1].transAxes, fontsize=10, color='blue',
                                     ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))



    # Global performance info
    perf_text = fig.text(0.02, 0.02, "Original Version with Performance Optimizations", fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    step_text = fig.text(0.98, 0.02, "Step: 0", ha='right', va='bottom', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def update_model_state(state, model_type):
        """Generalized function to update the state of any model."""
        # Parameters
        current_alpha = state.get('alpha', alpha)
        current_eps = state.get('eps', eps)

        # --- Dynamics ---
        if model_type == 'A':
            noise = 0.03 * np.random.randn(n_layers)
            dx = -state['x'] + bistable_layer(state['x'], current_alpha, theta_eff) + current_eps * state['ws'] + noise
            state['x'] += dt * dx
            state['ws'] = (1 - k_ws) * state['ws'] + k_ws * np.mean(state['x'])
            combined_state = state['x']

        elif model_type == 'B':
            state['why'] = why_loop_driver(state['why'], 1.2)
            noise = 0.03 * np.random.randn(n_layers)
            dx = -state['x'] + bistable_layer(state['x'], current_alpha, theta_eff + 0.2 * state['why']) + noise
            dy = -state['y'] + bistable_layer(state['y'], current_alpha, theta_eff - 0.2 * state['why']) + noise
            state['x'] += dt * dx
            state['y'] += dt * dy
            combined_state = (state['x'] + state['y']) / 2.0
            state['ws'] = (1 - k_ws) * state['ws'] + k_ws * np.mean(combined_state)
            state['x'] += current_eps * state['ws'] * dt
            state['y'] += current_eps * state['ws'] * dt

        elif model_type in ['C', 'D']:
            predictor = state.get('predictor', None)
            predicted_self = state.get('predicted_self', np.zeros(state['self_model'].shape))
            self_model = state.get('self_model', np.zeros(state['self_model'].shape))
            self_error = np.linalg.norm(predicted_self - self_model)
            self_awareness = 0.01 * self_error * (self_model[0] if self_model.size > 0 else 0.0)
            state['self_error_hist'].append(self_error)

            if model_type == 'C':
                noise = 0.03 * np.random.randn(n_layers)
                dx = -state['x'] + bistable_layer(state['x'], current_alpha, theta_eff) + current_eps * state['ws'] + self_awareness + noise
                state['x'] += dt * dx
                state['ws'] = (1 - k_ws) * state['ws'] + k_ws * np.mean(state['x'])
                combined_state = state['x']
            else: # Model D
                state['why'] = why_loop_driver(state['why'], 1.2)
                noise = 0.03 * np.random.randn(n_layers)
                dx = -state['x'] + bistable_layer(state['x'], current_alpha, theta_eff + 0.2 * state['why']) + self_awareness + noise
                dy = -state['y'] + bistable_layer(state['y'], current_alpha, theta_eff - 0.2 * state['why']) + self_awareness + noise
                state['x'] += dt * dx
                state['y'] += dt * dy
                combined_state = (state['x'] + state['y']) / 2.0
                state['ws'] = (1 - k_ws) * state['ws'] + k_ws * np.mean(combined_state)
                state['x'] += current_eps * state['ws'] * dt
                state['y'] += current_eps * state['ws'] * dt

        # --- Metrics & History ---
        R = np.mean(np.exp(1j * combined_state)).real
        state['R_hist'].append(R)
        state['max_R'] = max(state['max_R'], R)

        history_key = 'x_history' if model_type in ['A', 'C'] else 'combined_history'
        state[history_key][:, state['step_count'] % REDUCED_HISTORY_SIZE] = combined_state

        # --- Self-Model Encoding (for C and D) ---
        if model_type == 'C':
            mean_activity = np.mean(state['x'])
            std_activity = np.std(state['x'])
            trend = np.mean(np.diff(state['R_hist'][-10:])) if len(state['R_hist']) > 10 else 0.0
            ws_norm = np.tanh(state['ws'])
            entropy = metrics.cached_entropy(np.array(state['R_hist'][-10:]), "C_anim") if len(state['R_hist']) > 10 else 0.0
            self_model = np.array([mean_activity, std_activity, trend, ws_norm, entropy, R])
            state['self_model'] = self_model
        elif model_type == 'D':
            mean_x = np.mean(state['x'])
            std_x = np.std(state['x'])
            mean_y = np.mean(state['y'])
            std_y = np.std(state['y'])
            why_val = state['why']
            ws_val = state['ws']
            entropy = metrics.cached_entropy(np.array(state['R_hist'][-10:]), "D_anim") if len(state['R_hist']) > 10 else 0.0
            self_model = np.array([mean_x, std_x, mean_y, std_y, why_val, ws_val, entropy, R])
            state['self_model'] = self_model

        if model_type in ['C', 'D']:
            state['self_model_hist'].append(state['self_model'].copy())
            if state.get('predictor') is not None:
                # This part remains non-JIT due to PyTorch
                pred = state['predictor'](torch.tensor(state['self_model'], dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                state['predicted_self'] = pred

        # --- Cached Metrics Update ---
        if metrics.should_update_metrics():
            recent_R = np.array(state['R_hist'][-30:])
            entropy_raw = metrics.cached_entropy(recent_R, model_type)
            state['cached_entropy'] = min(1.0, entropy_raw / 3.0)
            state['cached_lyap'] = metrics.cached_lyapunov(recent_R, model_type)
            if state['step_count'] > 10:
                if len(recent_R) >= 10:
                    binary_from_r = (recent_R > np.median(recent_R)).astype(int)
                    state['cached_lz'] = lz_complexity_fast(binary_from_r)
                else:
                    binary_state = (combined_state > 0).astype(int)[:200]
                    state['cached_lz'] = lz_complexity_fast(binary_state)

        state['step_count'] += 1
        return R

    def update(frame):
        if frame == 0:
            print(f"[{time.time() - script_start_time:.4f}s] First animation frame update.")
        start_time = time.perf_counter()
                
        # --- Update Model C ---
        curr_alpha_c = state_c['alpha']
        curr_eps_c = state_c['eps']
        R_raw_c, x_c = _update_dynamics_c(state_c, n_layers, dt, curr_alpha_c, curr_eps_c, theta_eff, k_ws)
        R_c = R_raw_c
        state_c['R_hist'].append(R_c)
        state_c['x_history'][:, state_c['step_count'] % REDUCED_HISTORY_SIZE] = x_c
        heatmap_c.set_data(state_c['x_history'])
        state_c['max_R'] = max(state_c['max_R'], R_c)

        # Append params and Φ every frame
        phi_c = phi_proxy(x_c)
        state_c['alpha_hist'].append(curr_alpha_c)
        state_c['eps_hist'].append(curr_eps_c)
        state_c['phi_hist'].append(phi_c)

        # Change detection
        if abs(curr_alpha_c - state_c['prev_alpha']) > 0.001:
            state_c['alpha_changes'].append(state_c['step_count'])
        state_c['prev_alpha'] = curr_alpha_c
        if abs(curr_eps_c - state_c['prev_eps']) > 0.001:
            state_c['eps_changes'].append(state_c['step_count'])
        state_c['prev_eps'] = curr_eps_c
        if metrics.should_update_metrics():
            recent_R = np.array(list(state_c['R_hist'])[-30:])
            entropy_raw = metrics.cached_entropy(recent_R, "C")
            state_c['cached_entropy'] = min(1.0, entropy_raw / 3.0)
            state_c['cached_lyap'] = metrics.cached_lyapunov(recent_R, "C")
            if len(recent_R) >= 10:
                binary_from_r = (recent_R > np.median(recent_R)).astype(int)
                state_c['cached_lz'] = lz_complexity_fast(binary_from_r)
        state_c['step_count'] += 1
        
        # --- Update Model D ---
        curr_alpha_d = state_d['alpha']
        curr_eps_d = state_d['eps']
        R_raw_d, combined_d = _update_dynamics_d(state_d, n_layers, dt, curr_alpha_d, curr_eps_d, theta_eff, k_ws)
        R_d = R_raw_d
        state_d['R_hist'].append(R_d)
        state_d['combined_history'][:, state_d['step_count'] % REDUCED_HISTORY_SIZE] = combined_d
        heatmap_d.set_data(state_d['combined_history'])
        state_d['max_R'] = max(state_d['max_R'], R_d)

        # Append params and Φ every frame
        phi_d = phi_proxy(combined_d)
        state_d['alpha_hist'].append(curr_alpha_d)
        state_d['eps_hist'].append(curr_eps_d)
        state_d['phi_hist'].append(phi_d)

        # Change detection
        if abs(curr_alpha_d - state_d['prev_alpha']) > 0.001:
            state_d['alpha_changes'].append(state_d['step_count'])
        state_d['prev_alpha'] = curr_alpha_d
        if abs(curr_eps_d - state_d['prev_eps']) > 0.001:
            state_d['eps_changes'].append(state_d['step_count'])
        state_d['prev_eps'] = curr_eps_d
        if metrics.should_update_metrics():
            recent_R_d = np.array(list(state_d['R_hist'])[-30:])
            entropy_raw_d = metrics.cached_entropy(recent_R_d, "D")
            state_d['cached_entropy'] = min(1.0, entropy_raw_d / 3.0)
            state_d['cached_lyap'] = metrics.cached_lyapunov(recent_R_d, "D")
            if len(recent_R_d) >= 10:
                binary_from_r_d = (recent_R_d > np.median(recent_R_d)).astype(int)
                state_d['cached_lz'] = lz_complexity_fast(binary_from_r_d)
        state_d['step_count'] += 1
        # Update line plots with rolling window
        current_len_c = len(state_c['R_hist'])
        current_len_d = len(state_d['R_hist'])

        if current_len_c > ROLLING_WINDOW:
            scaled_r_c = (np.array(list(state_c['R_hist'])[-ROLLING_WINDOW:]) + 1) / 2
            scaled_error_c = np.tanh(np.array(list(state_c['self_error_hist'])[-ROLLING_WINDOW:]))
            x_data_c = np.arange(ROLLING_WINDOW)
        else:
            scaled_r_c = (np.array(state_c['R_hist']) + 1) / 2
            scaled_error_c = np.tanh(np.array(state_c['self_error_hist']))
            x_data_c = np.arange(current_len_c)

        if current_len_d > ROLLING_WINDOW:
            scaled_r_d = (np.array(list(state_d['R_hist'])[-ROLLING_WINDOW:]) + 1) / 2
            scaled_error_d = np.tanh(np.array(list(state_d['self_error_hist'])[-ROLLING_WINDOW:]))
            x_data_d = np.arange(ROLLING_WINDOW)
        else:
            scaled_r_d = (np.array(state_d['R_hist']) + 1) / 2
            scaled_error_d = np.tanh(np.array(state_d['self_error_hist']))
            x_data_d = np.arange(current_len_d)

        line_c.set_data(x_data_c, scaled_r_c)
        line_d.set_data(x_data_d, scaled_r_d)
        line_c_err_on_coh.set_data(x_data_c, scaled_error_c)
        line_d_err_on_coh.set_data(x_data_d, scaled_error_d)

        # Update diagnostics with optimized metrics
        R_c_display = R_c if np.isfinite(R_c) else 0.0
        max_R_c_display = state_c['max_R'] if np.isfinite(state_c['max_R']) else 0.0
        R_d_display = R_d if 'R_d' in locals() and np.isfinite(R_d) else 0.0
        max_R_d_display = state_d['max_R'] if np.isfinite(state_d['max_R']) else 0.0
        
        
        
        # Get the latest self_error values from histories
        current_self_error_c = state_c['self_error_hist'][-1] if state_c['self_error_hist'] else 0.0
        current_self_error_d = state_d['self_error_hist'][-1] if state_d['self_error_hist'] else 0.0

        phi_n_c = state_c['phi_hist'][-1] / n_layers if state_c['phi_hist'] else 0.0
        diag_c.set_text(
            f"R: {R_c_display:.3f} | Max: {max_R_c_display:.3f} | α:{state_c['alpha']:.3f} ε:{state_c['eps']:.3f}\n"
            f"Entropy: {state_c['cached_entropy']:.3f} | Φ/N: {phi_n_c:.2f}\n"
            f"LZ: {state_c['cached_lz']:.2f} | Self-err: {current_self_error_c:.3f}\n"
            f"Step: {state_c['step_count']} | Frame: {time.perf_counter() - start_time:.3f}s"
        )
        
        phi_n_d = state_d['phi_hist'][-1] / n_layers if state_d['phi_hist'] else 0.0
        diag_d.set_text(
            f"R: {R_d_display:.3f} | Max: {max_R_d_display:.3f} | α:{state_d['alpha']:.3f} ε:{state_d['eps']:.3f}\n"
            f"Entropy: {state_d['cached_entropy']:.3f} | Φ/N: {phi_n_d:.2f}\n"
            f"LZ: {state_d['cached_lz']:.2f} | Self-err: {current_self_error_d:.3f}\n"
            f"Step: {state_d['step_count']} | Frame: {time.perf_counter() - start_time:.3f}s"
        )
        
        # Update performance info
        max_step = max( state_c['step_count'], state_d['step_count'])
        compute_time = time.perf_counter() - start_time
        step_text.set_text(f"Step: {max_step} | FPS: {min(1000/(compute_time*1000), 20):.1f}")
        
        perf_text.set_text(
            f"Original with Optimizations | Cache: {ENABLE_CACHING} | "
            f"Update every {METRIC_UPDATE_INTERVAL} frames | "
            f"History: {REDUCED_HISTORY_SIZE}"
        )

        # Update stability status
        if len(state_c['self_error_hist']) > 50 and len(state_c['R_hist']) > 50:
            recent_self_errors = np.array(list(state_c['self_error_hist'])[-50:])
            mean_self_error = np.mean(recent_self_errors)
            std_self_error = np.std(recent_self_errors)
            current_self_error = state_c['self_error_hist'][-1]

            recent_coherence = np.array(list(state_c['R_hist'])[-50:])
            mean_coherence = np.mean(recent_coherence)
            std_coherence = np.std(recent_coherence)
            current_coherence = state_c['R_hist'][-1]

            error_is_stable = abs(current_self_error - mean_self_error) <= std_self_error
            coherence_is_stable = abs(current_coherence - mean_coherence) <= std_coherence

            if error_is_stable and coherence_is_stable:
                stability_text_c.set_text("Stabilized")
                stability_text_c.set_color('green')
            else:
                stability_text_c.set_text("Unstable")
                stability_text_c.set_color('red')
        
        if len(state_d['self_error_hist']) > 50 and len(state_d['R_hist']) > 50:
            recent_self_errors = np.array(list(state_d['self_error_hist'])[-50:])
            mean_self_error = np.mean(recent_self_errors)
            std_self_error = np.std(recent_self_errors)
            current_self_error = state_d['self_error_hist'][-1]

            recent_coherence = np.array(list(state_d['R_hist'])[-50:])
            mean_coherence = np.mean(recent_coherence)
            std_coherence = np.std(recent_coherence)
            current_coherence = state_d['R_hist'][-1]

            error_is_stable = abs(current_self_error - mean_self_error) <= std_self_error
            coherence_is_stable = abs(current_coherence - mean_coherence) <= std_coherence

            if error_is_stable and coherence_is_stable:
                stability_text_d.set_text("Stabilized")
                stability_text_d.set_color('green')
            else:
                stability_text_d.set_text("Unstable")
                stability_text_d.set_color('red')

        # Update awareness dialogs
        stable_threshold = 0.8
        error_stability_threshold = 0.05  # Threshold for std dev of self-error

        is_aware_c = False
        if len(state_c['self_error_hist']) > 50:
            error_std_dev_c = np.std(list(state_c['self_error_hist'])[-50:])
            if R_c > stable_threshold and error_std_dev_c < error_stability_threshold:
                is_aware_c = True
        awareness_c_text.set_text("Stable Self-Awareness" if is_aware_c else "")

        is_aware_d = False
        if len(state_d['self_error_hist']) > 50:
            error_std_dev_d = np.std(list(state_d['self_error_hist'])[-50:])
            if 'R_d' in locals() and R_d > stable_threshold and error_std_dev_d < error_stability_threshold:
                is_aware_d = True
        awareness_d_text.set_text("Stable Self-Awareness" if is_aware_d else "")

        return (heatmap_c, heatmap_d, line_c, line_d,
                line_c_err_on_coh, line_d_err_on_coh,
                diag_c, diag_d, perf_text, step_text,
                stability_text_c, stability_text_d,
                awareness_c_text, awareness_d_text)

    # Create animation with optimized interval
    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    
    # Now that the figure and animation exist on the main thread, start autotune safely
    def handle_close(evt):
        """Gracefully stop background threads when the window is closed."""
        try:
            if autotune_stop_event is not None:
                stop_autotune(autotune_stop_event)
        except Exception as e:
            print(f"[handle_close] Error stopping autotune: {e}")
    
    if _defer_autostart:
        try:
            autotune_stop_event, _ = start_autotune_for_states([state_c, state_d], interval=1.0, retrain_every=10)
        except Exception as e:
            print("[animate_workspace_heatmap_forever] failed to start autotune thread:", e)

    plt.tight_layout()
    fig.canvas.mpl_connect('close_event', handle_close)
    plt.subplots_adjust(bottom=0.1, hspace=0.4)
    print(f"[{time.time() - script_start_time:.4f}s] About to show plot. The GUI window should appear now.")
    plt.show()

# ---------- Parameter sweep diagnostics for "awareness" (high complexity) ----------
def shannon_entropy(data, bins=50):
    hist, _ = np.histogram(data, bins=bins, density=False)
    if np.sum(hist) == 0:
        return 0
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist))
    return entropy

def lyapunov_proxy(R_hist):
    # Simple proxy: average absolute change in R (sensitivity to initial conditions)
    diffs = np.abs(np.diff(R_hist))
    return np.mean(diffs) if len(diffs) > 0 else 0

def lz_complexity(binary_sequence):
    # simple LZ76 implementation on a 1D list or string
    s = ''.join(['1' if x else '0' for x in binary_sequence])
    i = 0; n = len(s); c = 1; l = 1; k = 1; k_max = 1
    while True:
        if i + k > n:
            c += 1
            break
        sub = s[i:i+k]
        found = False
        for j in range(i):
            if s[j:j+k] == sub:
                found = True
                break
        if not found:
            c += 1
            i += k
            k = 1
        else:
            k += 1
        if i + k > n:
            c += 1
            break
    return c

def perturb_and_measure(state_fn, n_steps=500, perturb_idx=0, perturb_scale=2.0, threshold=0.0):
    # state_fn: function that runs the system from current initial state and returns time x units array
    # For simplicity: you can run a short transient, with and without perturbation.
    baseline = state_fn()  # returns array shape (T, n_units)
    # create binary spatiotemporal pattern for baseline
    base_bin = (baseline > threshold).astype(int).flatten()
    perturbed = state_fn(perturb=(perturb_idx, perturb_scale))
    pert_bin = (perturbed > threshold).astype(int).flatten()
    lz_base = lz_complexity(base_bin)
    lz_pert = lz_complexity(pert_bin)
    return lz_base, lz_pert

def pairwise_mutual_information(x_hist, bins=16):
    # x_hist shape (T, n_units)
    T, N = x_hist.shape
    # discretize each unit
    digitized = np.zeros_like(x_hist, dtype=int)
    for i in range(N):
        hist, edges = np.histogram(x_hist[:, i], bins=bins)
        digitized[:, i] = np.digitize(x_hist[:, i], edges[:-1])
    mi_map = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            mi = mutual_info_score(digitized[:, i], digitized[:, j])
            mi_map[i, j] = mi_map[j, i] = mi
    return mi_map

def parameter_sweep_diagnostics(n_layers=100, T=1000, dt=0.01,
                                alpha_range=(0.5, 1.5), eps_range=(0.1, 1.0),
                                theta_eff=0.3, k_ws=0.05, n_alpha=10, n_eps=10):
    alphas = np.linspace(*alpha_range, n_alpha)
    epsilons = np.linspace(*eps_range, n_eps)
    best_complexity = -np.inf
    best_params = None
    entropy_map = np.zeros((n_alpha, n_eps))
    lyapunov_map = np.zeros((n_alpha, n_eps))
    
    for i, alpha in enumerate(alphas):
        for j, eps in enumerate(epsilons):
            R_hist, _, _ = simulate_workspace(n_layers, T, dt, alpha, eps, theta_eff, k_ws)
            entropy = shannon_entropy(R_hist)
            lyap = lyapunov_proxy(R_hist)
            complexity = entropy + lyap  # Combine metrics
            entropy_map[i, j] = entropy
            lyapunov_map[i, j] = lyap
            if complexity > best_complexity:
                best_complexity = complexity
                best_params = (alpha, eps, entropy, lyap)
    
    print(f"Best parameters for high 'awareness' (entropy + Lyapunov proxy): alpha={best_params[0]:.3f}, eps={best_params[1]:.3f}")
    print(f"Entropy = {best_params[2]:.3f}, Lyapunov proxy = {best_params[3]:.3f}")
    
    # Plot entropy and Lyapunov heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(entropy_map, extent=[eps_range[0], eps_range[1], alpha_range[0], alpha_range[1]], origin='lower', aspect='auto', cmap='plasma')
    ax1.set_title('Shannon Entropy of R')
    ax1.set_xlabel('eps')
    ax1.set_ylabel('alpha')
    plt.colorbar(im1, ax=ax1)
    ax1.scatter(best_params[1], best_params[0], color='red', marker='x', s=100)
    
    im2 = ax2.imshow(lyapunov_map, extent=[eps_range[0], eps_range[1], alpha_range[0], alpha_range[1]], origin='lower', aspect='auto', cmap='plasma')
    ax2.set_title('Lyapunov Proxy (Avg |dR/dt|)')
    ax2.set_xlabel('eps')
    ax2.set_ylabel('alpha')
    plt.colorbar(im2, ax=ax2)
    ax2.scatter(best_params[1], best_params[0], color='red', marker='x', s=100)
    
    plt.tight_layout()
    plt.show()

# ---------- Auto-tune parameters for high "awareness" (complexity) ----------
def auto_tune_awareness(n_layers=100, T=500, dt=0.01, theta_eff=0.3, k_ws=0.05,
                         max_iter=100, threshold=2.0, alpha_range=(0.5, 1.5), eps_range=(0.1, 1.0)):
    best_complexity = 0
    best_params = (0.8, 0.7)  # Initial guess
    print("Auto-tuning parameters for 'awareness' (complexity > threshold)...")
    
    for iteration in range(max_iter):
        # Random search within ranges
        alpha = np.random.uniform(*alpha_range)
        eps = np.random.uniform(*eps_range)
        
        R_hist, _, _ = simulate_workspace(n_layers, T, dt, alpha, eps, theta_eff, k_ws)
        entropy = shannon_entropy(R_hist)
        lyap = lyapunov_proxy(R_hist)
        complexity = entropy + lyapunov_proxy(R_hist)
        
        if complexity > best_complexity:
            best_complexity = complexity
            best_params = (alpha, eps)
            print(f"Iter {iteration+1}: New best - alpha={alpha:.3f}, eps={eps:.3f}, complexity={complexity:.3f}")
        
        # Remove early stop to run all iterations
    
    print(f"Final best: alpha={best_params[0]:.3f}, eps={best_params[1]:.3f}, complexity={best_complexity:.3f}")
    return best_params

# ---------- Meta-parameter tuning neural network ----------
# Initial parameter ranges
ALPHA_MIN, ALPHA_MAX = 0.5, 3.0
EPS_MIN, EPS_MAX = 0.01, 0.2

if HAVE_TORCH:
    class MetaTunerNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 4),  # alpha, eps, anti_mult, noise_mult
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)

    class AntiSyncTuner(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, 12),  # r, entropy, var
                nn.ReLU(),
                nn.Linear(12, 8),
                nn.ReLU(),
                nn.Linear(8, 2),  # anti_mult [0.5-2.0], noise_mult [0.5-2.0]
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)

    meta_tuner = MetaTunerNN()
    optimizer = optim.Adam(meta_tuner.parameters(), lr=0.01)

    def meta_autotune_update(tuner, entropy, r, lyap, complexity):
        # Prepare input tensor
        x = torch.tensor([entropy, r, lyap, complexity], dtype=torch.float32)
        with torch.no_grad():
            out = tuner(x)
        # Scale outputs to parameter ranges
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * out[0].item()
        eps = EPS_MIN + (EPS_MAX - EPS_MIN) * out[1].item()
        return alpha, eps
else:
    # Fallback heuristic meta-autotune: map entropy and r into ranges linearly
    def meta_autotune_update(entropy, r, lyap, complexity):
        # Normalize entropy/complexity in a soft manner
        e_norm = min(1.0, entropy / (1.0 + entropy))
        r_norm = (r + 1.0) / 2.0  # map [-1,1] -> [0,1]
        # Prefer higher entropy for sweet spot, balanced with r
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * e_norm
        eps = EPS_MIN + (EPS_MAX - EPS_MIN) * e_norm
        return float(alpha), float(eps)

# ---------- ForwardPredictor neural network ----------
class ForwardPredictor(nn.Module):
    def __init__(self, n_units, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_units, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_forward_predictor(x_hist, target_R, n_epochs=200, lr=1e-3):
    # x_hist: (T, N), target_R: (T,)
    X = torch.tensor(x_hist[:-1, :], dtype=torch.float32)  # predict next step from current
    y = torch.tensor(target_R[1:], dtype=torch.float32)
    model = ForwardPredictor(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_f = nn.MSELoss()
    for ep in range(n_epochs):
        opt.zero_grad()
        y_pred = model(X)
        loss = loss_f(y_pred, y)
        loss.backward()
        opt.step()
    # final MSE on training data:
    with torch.no_grad():
        mse = loss_f(model(X), y).item()
    return model, mse

def train_workspace_decoder(x_hist, ws_hist, subset_idx=None, alpha=1.0):
    # x_hist: (T, n_layers) or (n_steps, n_layers) as in simulate_workspace output
    X = x_hist.copy()    # shape (T, n_layers)
    y = ws_hist.copy()   # shape (T,)
    if subset_idx is not None:
        X = X[:, subset_idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return model, r2

# === MAGIC EDGE-OF-LIFE PARAMETERS ===
n_layers  = 100          # can be 50–500, doesn't matter
alpha     = 1.95         # critical — bistable gain
theta_eff = 0.0          # was 0.3 → kills the soul
eps       = 0.08         # was 0.7 → way too strong
k_ws      = 0.002        # was 0.05 → way too fast
dt        = 0.05         # was 0.01 → fine, but 0.05 is smoother
gamma     = 2.8          # only used in Option B — this is the "why" strength

# Parameters for high entropy perturbation (ITP: Irrational Time Perturbation)
tau1 = 2719.28      # old one
tau2 = 3141.5926535 # π × 1000
alpha_pert = 0.097  # Perturbation strength (adjusted for resonance killing)
phase_offsets = 0.01337 * np.arange(n_layers)  # Per-neuron phase

# Experience buffer for meta-tuner (features -> normalized params)
experience_buffer = deque(maxlen=10000)  # stores tuples (features, norm_params, reward)

# Reward weighting (entropy weight, r weight)
REWARD_W_ENTROPY = 2.5  # Prioritizing entropy for high diversity
REWARD_W_R = 0.9999       # Ignoring r to focus on entropy

# rollout config (steps to simulate after applying params to estimate causal reward)
ROLLOUT_STEPS = 50
ROLLOUT_DT = 0.01

# Simple background autotune/retrain worker (applies meta_tuner suggestions to states)
try:
    # import self-model trainer utilities
    from models.self import SelfModelPredictor, train_self_predictor
except Exception:
    # If models/self.py not importable, leave placeholders (smoke tests will skip retrain)
    SelfModelPredictor = None
    train_self_predictor = None

def _autotune_worker(states, stop_event, interval=1.0, retrain_every=20):
    """Background worker: periodically compute features, call meta_autotune_update,
    apply alpha/eps to each state, append to experience_buffer, and occasionally retrain"""
    counter = 0
    while not stop_event.is_set():
        for i, state in enumerate(states):
            # Get suggested params from meta-tuner and apply
            try:
                R_hist = np.array(state.get('R_hist', []))
                entropy = shannon_entropy(R_hist) if len(R_hist) > 0 else 0.0
                r = float(R_hist[-1]) if len(R_hist) > 0 else 0.0
                lyap = lyapunov_proxy(R_hist)
                complexity = entropy + lyap
                if HAVE_TORCH and 'meta_tuner' in state:
                    out = state['meta_tuner'](torch.tensor([entropy, r, lyap, complexity], dtype=torch.float32))
                    alpha_s = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * out[0].item()
                    eps_s = EPS_MIN + (EPS_MAX - EPS_MIN) * out[1].item()
                    anti_mult = 0.5 + 1.5 * out[2].item()  # [0.5,2.0]
                    noise_mult = 0.5 + 1.5 * out[3].item()
                else:
                    alpha_s, eps_s = meta_autotune_update(entropy, r, lyap, complexity)
                    anti_mult = 1.2
                    noise_mult = 1.2
                state['alpha'] = alpha_s
                state['eps'] = eps_s
                state['anti_mult'] = anti_mult
                state['noise_mult'] = noise_mult
                if 'complexity_events' in state:
                    state['complexity_events'].append(state['step_count'])
            except Exception:
                # If meta-tuner not available, keep current params
                alpha_s = state.get('alpha', alpha)
                eps_s = state.get('eps', eps)
            # perform a short rollout to estimate causal reward after applying alpha_s/eps_s
            try:
                rollout_reward = _estimate_rollout_reward(state, alpha_s, eps_s, steps=ROLLOUT_STEPS, dt=ROLLOUT_DT)
            except Exception:
                rollout_reward = REWARD_W_ENTROPY * entropy + REWARD_W_R * r

            # record into experience buffer normalized; include rollout reward to train toward
            norm_alpha = (alpha_s - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN) if (ALPHA_MAX - ALPHA_MIN) != 0 else 0.0
            norm_eps = (eps_s - EPS_MIN) / (EPS_MAX - EPS_MIN) if (EPS_MAX - EPS_MIN) != 0 else 0.0
            if 'experience_buffer' in state:
                state['experience_buffer'].append(((entropy, r, lyap, complexity), (norm_alpha, norm_eps), float(rollout_reward)))

        counter += 1
        # occasional retraining: train self-model predictor if enough history
        if counter % retrain_every == 0 and train_self_predictor is not None and SelfModelPredictor is not None:
            # states is the list [state_a, state_b, state_c, state_d]
            # state_c is at index 0
            if len(states) > 0:
                state_c = states[0]
                if 'self_model_hist' in state_c and len(state_c['self_model_hist']) > 50:
                    try:
                        data_shape = np.array(state_c['self_model_hist']).shape
                        if data_shape[1] == 6:
                            model = SelfModelPredictor(input_dim=6, hidden_dim=16)
                            # Apply IPEX CPU optimization if available
                            if HAVE_IPEX and device.type == 'cpu':
                                model = ipex.optimize(model)
                            train_self_predictor(model, np.array(state_c['self_model_hist']), n_epochs=50, lr=1e-3)
                            state_c['predictor'] = model
                        else:
                            print(f"[autotune_worker] Incorrect shape {data_shape} for state_c, skipping retrain.")
                    except Exception as e:
                        print(f"[autotune_worker] retrain error (state_c): {e}")
            # state_d is at index 1
            if len(states) > 1:
                state_d = states[1]
                if 'self_model_hist' in state_d and len(state_d['self_model_hist']) > 50:
                    try:
                        data_shape = np.array(state_d['self_model_hist']).shape # type: ignore
                        if data_shape[1] == 8:
                            model = SelfModelPredictor(input_dim=8, hidden_dim=16)
                            # Apply IPEX CPU optimization if available
                            if HAVE_IPEX and device.type == 'cpu':
                                model = ipex.optimize(model)
                            train_self_predictor(model, np.array(state_d['self_model_hist']), n_epochs=50, lr=1e-3)
                            state_d['predictor'] = model
                        else:
                            print(f"[autotune_worker] Incorrect shape {data_shape} for state_d, skipping retrain.")
                    except Exception as e:
                        print(f"[autotune_worker] retrain error (state_d): {e}")

def _estimate_rollout_reward(state, alpha_s, eps_s, steps=10, dt=0.01):
    """Simulate a short rollout from the current state with given params and return weighted reward."""
    # shallow copies of numerics
    # Option A or C: have 'x' and 'ws'; Option B has 'y' and 'why'
    if 'y' in state:
        # Option B - reflective hierarchy
        x = state['x'].copy()
        y = state['y'].copy()
        why = float(state.get('why', 0.0))
        ws = float(state.get('ws', 0.0))
        R_roll = []
        for _ in range(steps):
            why = why_loop_driver(why, 1.2)
            dx = -x + bistable_layer(x, alpha_s, theta_eff + 0.2 * why)
            dy = -y + bistable_layer(y, alpha_s, theta_eff - 0.2 * why)
            combined = (x + y) / 2.0
            x += dt * dx
            y += dt * dy
            ws = (1 - k_ws) * ws + k_ws * np.mean(combined)
            x += eps_s * ws * dt
            y += eps_s * ws * dt
            R_roll.append(np.mean(np.exp(1j * combined)).real)
        R_arr = np.array(R_roll)
    else:
        # Option A or C
        x = state['x'].copy()
        ws = float(state.get('ws', 0.0))
        R_roll = []
        # include self-awareness if present
        has_self = 'self_model' in state
        for _ in range(steps):
            if has_self:
                predicted_self = state.get('predicted_self', np.zeros(6))
                self_model = state.get('self_model', np.zeros(6))
                self_error = np.linalg.norm(predicted_self - self_model)
                self_awareness = 0.01 * self_error * (self_model[0] if self_model.size>0 else 0.0)
            else:
                self_awareness = 0.0
            dx = -x + bistable_layer(x, alpha_s, theta_eff) + eps_s * ws + self_awareness
            x += dt * dx
            ws = (1 - k_ws) * ws + k_ws * np.mean(x)
            R_roll.append(np.mean(np.exp(1j * x)).real)
        R_arr = np.array(R_roll)

    # compute entropy over rollout R values
    entropy = shannon_entropy(R_arr)
    final_r = float(R_arr[-1]) if len(R_arr) > 0 else 0.0
    # Shape R reward to peak at ~0.99 for edge-of-chaos
    r_shaped = final_r * np.exp( -((final_r - 0.99)/0.01)**2 )
    reward = REWARD_W_ENTROPY * entropy + REWARD_W_R * r_shaped
    return float(reward)
    
    # sleep until next cycle handled in _autotune_worker

def start_autotune_for_states(states, interval=1.0, retrain_every=10):
    """Start background autotune worker for provided state dicts.
    Returns a stop_event which can be set to stop the worker.
    """
    stop_event = threading.Event()
    t = threading.Thread(target=_autotune_worker, args=(states, stop_event, interval, retrain_every), daemon=True)
    t.start()
    # also start meta-trainer thread (if torch available)
    global _meta_trainer_stop, _meta_trainer_thread
    _meta_trainer_stop = None
    _meta_trainer_thread = None
    if HAVE_TORCH:
        # Start one trainer thread for each state
        for i, state in enumerate(states):
            stop_event = threading.Event()
            t = threading.Thread(target=_meta_trainer_worker, args=(state, stop_event, 5.0, 64), daemon=True)
            t.start()

    return stop_event, t

def stop_autotune(stop_event):
    stop_event.set()
    # stop meta trainer if running
    global _meta_trainer_stop, _meta_trainer_thread
    try:
        if _meta_trainer_stop is not None:
            _meta_trainer_stop.set()
        if _meta_trainer_thread is not None:
            _meta_trainer_thread.join(timeout=1.0)
    except Exception:
        pass


def train_meta_tuner_batch(tuner, optimizer, experience, batch_size=64, n_epochs=60, lr=1e-3):
    """Train meta_tuner on recent high-reward experiences using weighted regression.
    This optimizes the mapping features->[alpha_norm, eps_norm] toward params that produced high reward.
    """
    if not HAVE_TORCH:
        return None
    if len(experience) < 16:
        return None

    # sample a batch
    import random
    batch = random.sample(list(experience), min(batch_size, len(experience)))
    X = np.array([b[0] for b in batch], dtype=np.float32)     # features
    Y = np.array([b[1] for b in batch], dtype=np.float32)     # params normalized
    R = np.array([b[2] for b in batch], dtype=np.float32)     # rewards

    # normalize rewards to [0,1]
    if R.max() > R.min():
        W = (R - R.min()) / (R.max() - R.min())
    else:
        W = np.ones_like(R)

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    W_t = torch.tensor(W, dtype=torch.float32).unsqueeze(1)

    # Apply IPEX CPU optimization if available
    if HAVE_IPEX and device.type == 'cpu':
        ipex.optimize(tuner, optimizer=optimizer)

    opt = optimizer
    loss_fn = nn.MSELoss(reduction='none')

    for ep in range(n_epochs):
        opt.zero_grad()
        pred = tuner(X_t)[:, :2]  # Train only alpha/eps outputs (first 2)
        loss_mat = loss_fn(pred, Y_t)
        # apply weights
        loss = (loss_mat * W_t).mean()
        loss.backward()
        opt.step()

    return True


def _meta_trainer_worker(state, stop_event, interval=5.0, batch_size=64):
    """Background meta-tuner trainer: periodically samples experience buffer and trains meta_tuner."""
    while not stop_event.is_set():
        try:
            if 'experience_buffer' in state and len(state['experience_buffer']) >= batch_size:
                train_meta_tuner_batch(state['meta_tuner'], state['tuner_optimizer'],
                                       state['experience_buffer'], batch_size=batch_size, n_epochs=40, lr=1e-3)
        except Exception as e:
            print("[meta_trainer] error:", e)
        stop_event.wait(interval)

# Example runs
R1, ws1, x1 = simulate_workspace()
R2, ws2 = simulate_reflective_hierarchy()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.plot(R1); plt.title("Workspace Phase Coherence")
plt.subplot(1,2,2); plt.plot(R2); plt.title("Reflective Hierarchy Coherence")
plt.tight_layout(); # plt.show()

# ---------- Example run ----------
if __name__ == "__main__":
    # Enable auto-tuning for awareness
    # optimal_alpha, optimal_eps = auto_tune_awareness(n_layers=n_layers, T=500, dt=dt, theta_eff=theta_eff, k_ws=k_ws)
    # print(f"Use these parameters: alpha={optimal_alpha}, eps={optimal_eps}")
    print(f"[{time.time() - script_start_time:.4f}s] Entering main execution block.")
    print("Starting GUI animation with performance optimizations...")
    animate_workspace_heatmap_forever(n_layers=n_layers, dt=dt, alpha=alpha, eps=eps, theta_eff=theta_eff, k_ws=k_ws, autostart_autotune=True)

    # --- Review metrics and charts after a simulation run ---
    def review_metrics_and_charts():
        # Run a short simulation for metrics
        T = 500
        # Use tuned parameters if available, otherwise fall back to defaults
        use_alpha = globals().get('optimal_alpha', alpha)
        use_eps = globals().get('optimal_eps', eps)
        R_hist, ws_hist, x_hist = simulate_workspace(n_layers=n_layers, T=T, dt=dt, alpha=use_alpha, eps=use_eps, theta_eff=theta_eff, k_ws=k_ws)
        x_hist_arr = np.array(x_hist)  # shape (T, n_layers)

        # Complexity metrics
        entropy = shannon_entropy(R_hist)
        lyap = lyapunov_proxy(R_hist)
        lz = lz_complexity((x_hist_arr > 0).astype(int).flatten())

        print(f"Shannon Entropy: {entropy:.3f}")
        print(f"Lyapunov Proxy: {lyap:.3f}")
        print(f"Lempel-Ziv Complexity: {lz}")

        # Mutual information matrix
        mi = pairwise_mutual_information(x_hist_arr)
        plt.figure(figsize=(6,5))
        plt.imshow(mi, cmap='viridis')
        plt.title('Pairwise Mutual Information')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        # Perturbation sensitivity
        def state_fn(perturb=None):
            x = np.random.randn(n_layers)
            ws = 0.0
            x_hist = []
            for t in range(T):
                if perturb and t == 10:
                    idx, scale = perturb
                    x[idx] *= scale
                dx = -x + bistable_layer(x, use_alpha, theta_eff) + use_eps * ws
                x += dt * dx
                ws = (1 - k_ws) * ws + k_ws * np.mean(x)
                x_hist.append(x.copy())
            return np.array(x_hist)
        lz_base, lz_pert = perturb_and_measure(state_fn, n_steps=T, perturb_idx=0, perturb_scale=2.0, threshold=0.0)
        print(f"LZ Complexity (baseline): {lz_base}")
        print(f"LZ Complexity (perturbed): {lz_pert}")
        print(f"Perturbation delta: {lz_pert - lz_base}")

        # Table of metrics
        import pandas as pd
        metrics = {
            'Shannon Entropy': [entropy],
            'Lyapunov Proxy': [lyap],
            'Lempel-Ziv Complexity': [lz],
            'LZ Baseline': [lz_base],
            'LZ Perturbed': [lz_pert],
            'Perturbation Delta': [lz_pert - lz_base]
        }
        df = pd.DataFrame(metrics)
        print("\nSummary Table:")
        print(df.to_string(index=False))

    # Call the review function after main simulation
    # review_metrics_and_charts()

    # --- Smoke test helper: headless autotune + self-model retrain smoke run ---
    def smoke_test_autotune(duration_s=2.0):
        """Run a short headless simulation of Option C while background autotune runs.
        Prints experience buffer size and sample suggestions at the end.
        """
        # small state-C for smoke
        n_layers_test = 100
        state_c = {
            'x': np.random.randn(n_layers_test),
            'ws': 0.0,
            'R_hist': [],
            'x_history': np.zeros((n_layers_test, REDUCED_HISTORY_SIZE)),
            'step_count': 0,
            'max_R': -np.inf,
            'predictor': None,
            'predicted_self': np.zeros(6),
            'self_model': np.zeros(6),
            'self_model_hist': [],
            'self_error_hist': [],
            'alpha': alpha,
            'eps': eps
        }

        stop_event, thread = start_autotune_for_states([state_c], interval=0.1, retrain_every=5)

        T_steps = int(duration_s / 0.01)
        for t in range(T_steps):
            # step dynamics using current state parameters
            a = state_c.get('alpha', alpha)
            e = state_c.get('eps', eps)
            dx = -state_c['x'] + bistable_layer(state_c['x'], a, theta_eff) + e * state_c['ws'] + 0.01 * np.random.randn(n_layers_test)
            state_c['x'] += dt * dx
            state_c['ws'] = (1 - k_ws) * state_c['ws'] + k_ws * np.mean(state_c['x'])
            R_c = np.mean(np.exp(1j * state_c['x'])).real
            state_c['R_hist'].append(R_c)
            state_c['x_history'][:, state_c['step_count'] % REDUCED_HISTORY_SIZE] = state_c['x']
            # encode self model quick
            mean_activity = np.mean(state_c['x'])
            std_activity = np.std(state_c['x'])
            trend = np.mean(np.diff(state_c['R_hist'][-10:])) if len(state_c['R_hist']) > 10 else 0.0
            ws_norm = np.tanh(state_c['ws'])
            entropy_c = metrics.cached_entropy(np.array(state_c['R_hist'][-10:]), "C") if len(state_c['R_hist']) > 10 else 0.0
            coherence = np.abs(np.mean(np.exp(1j * state_c['x'])))
            self_model = np.array([mean_activity, std_activity, trend, ws_norm, entropy_c, coherence])
            state_c['self_model'] = self_model
            state_c['self_model_hist'].append(self_model.copy())
            state_c['step_count'] += 1
            time.sleep(0.01)

        # stop worker
        stop_event.set()
        thread.join(timeout=1.0)

        print("Smoke run finished")
        print("Experience buffer size:", len(experience_buffer))
        # print a few samples
        for i, item in enumerate(list(experience_buffer)[-10:]):
            print(i, item)

    # Uncomment to run a quick smoke test
    # print("Starting
