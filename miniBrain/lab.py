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
        try:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                device = torch.device('xpu')
                print("Using device: Intel GPU (XPU)")
            else:
                device = torch.device('cpu')
                print("Using device: CPU")
        except ImportError:
            device = torch.device('cpu')
            print("Using device: CPU")
    print(f"Selected device: {device}")
except Exception:
    torch = None
    nn = None
    optim = None
    HAVE_TORCH = False
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

# Performance optimization configuration
ENABLE_CACHING = True
CACHE_SIZE = 30
METRIC_UPDATE_INTERVAL = 1  # Update metrics every frame for immediate display
REDUCED_HISTORY_SIZE = 800  # Reduced from 2000
ROLLING_WINDOW = 1000  # Rolling window of 1k steps

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
        """Cache Lyapunov proxy calculations"""
        if not ENABLE_CACHING or len(data) < 5:
            return self._compute_lyapunov_fast(data)
            
        key = f"{len(data)}_{key_suffix}"
        if key in self.lyap_cache:
            return self.lyap_cache[key]
            
        result = self._compute_lyapunov_fast(data)
        self.lyap_cache[key] = result
        
        if len(self.lyap_cache) > CACHE_SIZE:
            self.lyap_cache.clear()
            
        return result
    
    def _compute_lyapunov_fast(self, data):
        """Fast Lyapunov proxy on recent data"""
        if len(data) < 3:
            return 0.0
        # Use last 12 values only
        recent_data = data[-12:] if len(data) > 12 else data
        diffs = np.abs(np.diff(recent_data))
        return np.mean(diffs)

# Fast LZ complexity
@nb.njit(cache=True)
def lz_complexity_fast(binary_sequence, max_length=300):
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
    
    while i + l <= n and c < 50:  # Limit iterations for performance
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
def bistable_layer(x, alpha, theta_eff):
    # This function is polymorphic for numpy/torch, so we don't use njit here
    if HAVE_TORCH and isinstance(x, torch.Tensor):
        return torch.tanh(alpha * x - theta_eff)
    else:
        return np.tanh(alpha * x - theta_eff)

# ---------- Why-loop driver ----------
def why_loop_driver(y, gamma):
    # Polymorphic for numpy/torch
    if HAVE_TORCH and isinstance(y, torch.Tensor):
        return torch.tanh(gamma * y)  # reflective recursion term
    else:
        return np.tanh(gamma * y)  # reflective recursion term

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
            R = torch.mean(torch.exp(1j * x)).real
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
            R_hist.append(np.mean(np.exp(1j * x)).real)
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

            R = torch.mean(torch.exp(1j * combined)).real
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

            R_hist.append(np.mean(np.exp(1j * combined)).real)
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

            R = torch.mean(torch.exp(1j * combined)).real
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

            R_hist.append(np.mean(np.exp(1j * combined)).real)
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

# ---------- Real-time heatmap and coherence animation (runs forever) ----------
def animate_workspace_heatmap_forever(n_layers=100, dt=0.05,
                                      alpha=1.95, eps=0.08, theta_eff=0.0, k_ws=0.002,
                                      autostart_autotune=False, rolling_window=ROLLING_WINDOW):
    # State for Option A
    state_a = {}
    if HAVE_TORCH:
        state_a['x'] = torch.randn(n_layers, device=device)
        state_a['ws'] = torch.tensor(0.0, device=device)
    else:
        state_a['x'] = np.random.randn(n_layers)
        state_a['ws'] = 0.0
    state_a['R_hist'] = []
    state_a['x_history'] = np.zeros((n_layers, REDUCED_HISTORY_SIZE))
    state_a['step_count'] = 0
    state_a['max_R'] = -np.inf
    
    # Cached metrics
    state_a['cached_entropy'] = 0.0
    state_a['cached_lyap'] = 0.0
    state_a['cached_lz'] = 0.0
    # --- Independent Tuner State ---
    if HAVE_TORCH:
        state_a['meta_tuner'] = MetaTunerNN()
        state_a['tuner_optimizer'] = optim.Adam(state_a['meta_tuner'].parameters(), lr=0.01)
    state_a['experience_buffer'] = deque(maxlen=2500) # Smaller independent buffer

    # State for Option B
    state_b = {}
    if HAVE_TORCH:
        state_b['x'] = torch.randn(n_layers, device=device)
        state_b['y'] = torch.randn(n_layers, device=device)
        state_b['why'] = torch.tensor(0.0, device=device)
        state_b['ws'] = torch.tensor(0.0, device=device)
    else:
        state_b['x'] = np.random.randn(n_layers)
        state_b['y'] = np.random.randn(n_layers)
        state_b['why'] = 0.0
        state_b['ws'] = 0.0
    state_b['R_hist'] = []
    state_b['combined_history'] = np.zeros((n_layers, REDUCED_HISTORY_SIZE))
    state_b['step_count'] = 0
    state_b['max_R'] = -np.inf
    
    # Cached metrics
    state_b['cached_entropy'] = 0.0
    state_b['cached_lyap'] = 0.0
    state_b['cached_lz'] = 0.0
    # --- Independent Tuner State ---
    if HAVE_TORCH:
        state_b['meta_tuner'] = MetaTunerNN()
        state_b['tuner_optimizer'] = optim.Adam(state_b['meta_tuner'].parameters(), lr=0.01)
    state_b['experience_buffer'] = deque(maxlen=2500)

    # State for Option C (Self-Referential)
    state_c = {}
    if HAVE_TORCH:
        state_c['x'] = torch.randn(n_layers, device=device)
        state_c['ws'] = torch.tensor(0.0, device=device)
    else:
        state_c['x'] = np.random.randn(n_layers)
        state_c['ws'] = 0.0
    state_c['R_hist'] = []
    state_c['x_history'] = np.zeros((n_layers, REDUCED_HISTORY_SIZE))
    state_c['step_count'] = 0
    state_c['max_R'] = -np.inf
    
    # Cached metrics
    state_c['cached_entropy'] = 0.0
    state_c['cached_lyap'] = 0.0
    state_c['cached_lz'] = 0.0
    
    state_c['predictor'] = None
    state_c['predicted_self'] = np.zeros(6)
    state_c['self_model'] = np.zeros(6)
    state_c['self_model_hist'] = []
    state_c['self_error_hist'] = []
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
    state_d['R_hist'] = []
    state_d['combined_history'] = np.zeros((n_layers, REDUCED_HISTORY_SIZE))
    state_d['step_count'] = 0
    state_d['max_R'] = -np.inf
    state_d['cached_entropy'] = 0.0
    state_d['cached_lyap'] = 0.0
    state_d['cached_lz'] = 0.0
    state_d['predictor'] = None
    state_d['predicted_self'] = np.zeros(8)
    state_d['self_model'] = np.zeros(8)
    state_d['self_model_hist'] = []
    state_d['self_error_hist'] = []
    # --- Independent Tuner State ---
    if HAVE_TORCH:
        state_d['meta_tuner'] = MetaTunerNN()
        state_d['tuner_optimizer'] = optim.Adam(state_d['meta_tuner'].parameters(), lr=0.01)
    state_d['experience_buffer'] = deque(maxlen=2500)
    # Defer starting autotune until after the GUI (figure/animation) is created
    autotune_stop_event = None
    _defer_autostart = bool(autostart_autotune)

    # Layout: 3 rows x 2 cols -> one row per model (heatmap | R-phase)
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    
    # Common heatmap settings
    heatmap_extent = [0, REDUCED_HISTORY_SIZE, 0, n_layers]
    
    # Option A
    heatmap_a = axes[0,0].imshow(np.zeros((n_layers, REDUCED_HISTORY_SIZE)), aspect='auto', 
                                cmap='hsv', vmin=0, vmax=2*np.pi, origin='lower', 
                                extent=heatmap_extent)
    axes[0,0].set_title('Option A: Global Workspace', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Time Step')
    axes[0,0].set_ylabel('Neuron Index')
    line_a, = axes[0,1].plot([], [], 'r', linewidth=1.5)
    axes[0,1].set_xlim(0, ROLLING_WINDOW)
    axes[0,1].set_ylim(-1.01, 1.01)
    axes[0,1].set_title('Option A: Coherence R', fontsize=10)
    axes[0,1].set_xlabel('Step')
    axes[0,1].set_ylabel('R')
    diag_a = axes[0,0].text(0.02, 0.98, "", transform=axes[0,0].transAxes, fontsize=9,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    awareness_a_text = axes[0,1].text(0.5, 0.5, "", transform=axes[0,1].transAxes, fontsize=12, color='green',
                                     ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Option B
    heatmap_b = axes[1,0].imshow(np.zeros((n_layers, REDUCED_HISTORY_SIZE)), aspect='auto',
                                cmap='hsv', vmin=0, vmax=2*np.pi, origin='lower',
                                extent=heatmap_extent)
    axes[1,0].set_title('Option B: Reflective Hierarchy', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Time Step')
    axes[1,0].set_ylabel('Neuron Index')
    line_b, = axes[1,1].plot([], [], 'b', linewidth=1.5)
    axes[1,1].set_xlim(0, ROLLING_WINDOW)
    axes[1,1].set_ylim(-1.01, 1.01)
    axes[1,1].set_title('Option B: Coherence R', fontsize=10)
    axes[1,1].set_xlabel('Step')
    axes[1,1].set_ylabel('R')
    diag_b = axes[1,0].text(0.02, 0.98, "", transform=axes[1,0].transAxes, fontsize=9,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    awareness_b_text = axes[1,1].text(0.5, 0.5, "", transform=axes[1,1].transAxes, fontsize=12, color='green',
                                     ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Option C (Self-Referential)
    heatmap_c = axes[2,0].imshow(np.zeros((n_layers, REDUCED_HISTORY_SIZE)), aspect='auto',
                                cmap='hsv', vmin=0, vmax=2*np.pi, origin='lower',
                                extent=heatmap_extent)
    axes[2,0].set_title('Option C: Self-Referential', fontsize=12, fontweight='bold')
    axes[2,0].set_xlabel('Time Step')
    axes[2,0].set_ylabel('Neuron Index')
    line_c, = axes[2,1].plot([], [], 'g', linewidth=1.5)
    axes[2,1].set_xlim(0, ROLLING_WINDOW)
    axes[2,1].set_ylim(-1.01, 1.01)
    axes[2,1].set_title('Option C: Coherence R', fontsize=10)
    axes[2,1].set_xlabel('Step')
    axes[2,1].set_ylabel('R')
    diag_c = axes[2,0].text(0.02, 0.98, "", transform=axes[2,0].transAxes, fontsize=9,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    awareness_c_text = axes[2,1].text(0.5, 0.5, "", transform=axes[2,1].transAxes, fontsize=12, color='green',
                                     ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Option D (B Self-Referential)
    heatmap_d = axes[3,0].imshow(np.zeros((n_layers, REDUCED_HISTORY_SIZE)), aspect='auto',
                                cmap='hsv', vmin=0, vmax=2*np.pi, origin='lower',
                                extent=heatmap_extent)
    axes[3,0].set_title('Option B Self-Referential', fontsize=12, fontweight='bold')
    axes[3,0].set_xlabel('Time Step')
    axes[3,0].set_ylabel('Neuron Index')
    line_d, = axes[3,1].plot([], [], 'purple', linewidth=1.5)
    axes[3,1].set_xlim(0, ROLLING_WINDOW)
    axes[3,1].set_ylim(-1.01, 1.01)
    axes[3,1].set_title('Option B Self-Ref Coherence R', fontsize=10)
    axes[3,1].set_xlabel('Step')
    axes[3,1].set_ylabel('R')
    diag_d = axes[3,0].text(0.02, 0.98, "", transform=axes[3,0].transAxes, fontsize=9,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    awareness_d_text = axes[3,1].text(0.5, 0.5, "", transform=axes[3,1].transAxes, fontsize=12, color='green',
                                     ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

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
        start_time = time.perf_counter()
        
        # Update Option A
        if HAVE_TORCH and isinstance(state_a['x'], torch.Tensor):
            noise = 0.03 * torch.randn(n_layers, device=device)
            dx_a = -state_a['x'] + bistable_layer(state_a['x'], alpha, theta_eff) + eps * state_a['ws'] + noise
            state_a['x'] += dt * dx_a
            state_a['ws'] = (1 - k_ws) * state_a['ws'] + k_ws * torch.mean(state_a['x'])
            R_a = torch.mean(torch.exp(1j * state_a['x'])).real.item()
            state_a['R_hist'].append(R_a)
            state_a['x_history'][:, state_a['step_count'] % REDUCED_HISTORY_SIZE] = state_a['x'].cpu().numpy()
            heatmap_a.set_data(state_a['x_history'] % (2*np.pi))
        else:
            noise = 0.03 * np.random.randn(n_layers)
            dx_a = -state_a['x'] + bistable_layer(state_a['x'], alpha, theta_eff) + eps * state_a['ws'] + noise
            state_a['x'] += dt * dx_a
            state_a['ws'] = (1 - k_ws) * state_a['ws'] + k_ws * np.mean(state_a['x'])
            R_a = np.mean(np.exp(1j * state_a['x'])).real
            state_a['R_hist'].append(R_a)
            state_a['x_history'][:, state_a['step_count'] % REDUCED_HISTORY_SIZE] = state_a['x']
            heatmap_a.set_data(state_a['x_history'] % (2*np.pi))
        state_a['max_R'] = max(state_a['max_R'], R_a)

        # Update cached metrics if needed
        if metrics.should_update_metrics():
            recent_R = np.array(state_a['R_hist'][-30:])
            entropy_raw = metrics.cached_entropy(recent_R, "A")
            # Scale entropy to 0-1 range for better display
            state_a['cached_entropy'] = min(1.0, entropy_raw / 3.0)  # Scale by typical max entropy
            state_a['cached_lyap'] = metrics.cached_lyapunov(recent_R, "A")
            
            # LZ complexity on current state (limited scope)
            if state_a['step_count'] > 10:
                if HAVE_TORCH and isinstance(state_a['x'], torch.Tensor):
                    binary_state = (state_a['x'] > 0).type(torch.int64).cpu().numpy()[:200]  # Limit to first 200 neurons
                else:
                    binary_state = (state_a['x'] > 0).astype(int)[:200]  # Limit to first 200 neurons
                # Use dynamic binary sequence from recent R values instead of static neuron states
                if len(recent_R) >= 10:
                    binary_from_r = (recent_R > np.median(recent_R)).astype(int)
                    state_a['cached_lz'] = lz_complexity_fast(binary_from_r)
                else:
                    state_a['cached_lz'] = lz_complexity_fast(binary_state)
        
        state_a['step_count'] += 1

        # Update Option B
        if HAVE_TORCH and isinstance(state_b['x'], torch.Tensor):
            state_b['why'] = why_loop_driver(state_b['why'], 1.2)
            dx_b = -state_b['x'] + bistable_layer(state_b['x'], alpha, theta_eff + 0.2 * state_b['why'])
            dy_b = -state_b['y'] + bistable_layer(state_b['y'], alpha, theta_eff - 0.2 * state_b['why'])
            noise = 0.03 * torch.randn(n_layers, device=device)
            dx_b += noise
            dy_b += noise
            combined = (state_b['x'] + state_b['y']) / 2.0
            state_b['x'] += dt * dx_b
            state_b['y'] += dt * dy_b
            state_b['ws'] = (1 - k_ws) * state_b['ws'] + k_ws * torch.mean(combined)
            state_b['x'] += eps * state_b['ws'] * dt
            state_b['y'] += eps * state_b['ws'] * dt
            R_b = torch.mean(torch.exp(1j * combined)).real.item()
            state_b['R_hist'].append(R_b)
            state_b['combined_history'][:, state_b['step_count'] % REDUCED_HISTORY_SIZE] = combined.cpu().numpy()
            heatmap_b.set_data(state_b['combined_history'] % (2*np.pi))
        else:
            state_b['why'] = why_loop_driver(state_b['why'], 1.2)
            dx_b = -state_b['x'] + bistable_layer(state_b['x'], alpha, theta_eff + 0.2 * state_b['why'])
            dy_b = -state_b['y'] + bistable_layer(state_b['y'], alpha, theta_eff - 0.2 * state_b['why'])
            noise = 0.03 * np.random.randn(n_layers)
            dx_b += noise
            dy_b += noise
            combined = (state_b['x'] + state_b['y']) / 2.0
            state_b['x'] += dt * dx_b
            state_b['y'] += dt * dy_b
            state_b['ws'] = (1 - k_ws) * state_b['ws'] + k_ws * np.mean(combined)
            state_b['x'] += eps * state_b['ws'] * dt
            state_b['y'] += eps * state_b['ws'] * dt
            R_b = np.mean(np.exp(1j * combined)).real
            state_b['R_hist'].append(R_b)
            state_b['combined_history'][:, state_b['step_count'] % REDUCED_HISTORY_SIZE] = combined
            heatmap_b.set_data(state_b['combined_history'] % (2*np.pi))
        state_b['max_R'] = max(state_b['max_R'], R_b)
        
        # Update cached metrics if needed
        if metrics.should_update_metrics():
            recent_R = np.array(state_b['R_hist'][-30:])
            entropy_raw = metrics.cached_entropy(recent_R, "B")
            # Scale entropy to 0-1 range for better display
            state_b['cached_entropy'] = min(1.0, entropy_raw / 3.0)  # Scale by typical max entropy
            state_b['cached_lyap'] = metrics.cached_lyapunov(recent_R, "B")
            
            if state_b['step_count'] > 10:
                if HAVE_TORCH and isinstance(combined, torch.Tensor):
                    binary_state = (combined > 0).type(torch.int64).cpu().numpy()[:200]
                else:
                    binary_state = (combined > 0).astype(int)[:200]
                # Use dynamic binary sequence from recent R values instead of static neuron states
                if len(recent_R) >= 10:
                    binary_from_r = (recent_R > np.median(recent_R)).astype(int)
                    state_b['cached_lz'] = lz_complexity_fast(binary_from_r)
                else:
                    state_b['cached_lz'] = lz_complexity_fast(binary_state)
        
        state_b['step_count'] += 1
        
        # Option C: Self-referential dynamics
        predictor = state_c.get('predictor', None)
        predicted_self = state_c.get('predicted_self', np.zeros(6))
        self_model = state_c.get('self_model', np.zeros(6))
        self_error = np.linalg.norm(predicted_self - self_model)
        self_awareness = 0.01 * self_error * (self_model[0] if self_model.size>0 else 0.0)
        
        if HAVE_TORCH and isinstance(state_c['x'], torch.Tensor):
            dx_c = -state_c['x'] + bistable_layer(state_c['x'], alpha, theta_eff) + eps * state_c['ws'] + self_awareness + 0.03 * torch.randn(n_layers, device=device)
            state_c['x'] += dt * dx_c
            state_c['ws'] = (1 - k_ws) * state_c['ws'] + k_ws * torch.mean(state_c['x'])
            R_c = torch.mean(torch.exp(1j * state_c['x'])).real.item()
            state_c['R_hist'].append(R_c)
            state_c['x_history'][:, state_c['step_count'] % REDUCED_HISTORY_SIZE] = state_c['x'].cpu().numpy()
            heatmap_c.set_data(state_c['x_history'] % (2*np.pi))
            state_c['max_R'] = max(state_c['max_R'], R_c)
            # encode self_model (simple)
            mean_activity = torch.mean(state_c['x']).item()
            std_activity = torch.std(state_c['x']).item()
            trend = np.mean(np.diff(state_c['R_hist'][-10:])) if len(state_c['R_hist']) > 10 else 0.0
            ws_norm = torch.tanh(state_c['ws']).item()
            entropy_c = metrics.cached_entropy(np.array(state_c['R_hist'][-10:]), "C_animation") if len(state_c['R_hist']) > 10 else 0.0
            coherence = torch.abs(torch.mean(torch.exp(1j * state_c['x']))).item()
            self_model = np.array([mean_activity, std_activity, trend, ws_norm, entropy_c, coherence])
            state_c['self_model'] = self_model
            # predictor prediction
            if predictor is not None:
                with torch.no_grad():
                    pred = predictor(torch.tensor(self_model, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
                    state_c['predicted_self'] = pred
            state_c['self_model_hist'].append(self_model.copy())
            state_c['self_error_hist'].append(self_error)
        else:
            dx_c = -state_c['x'] + bistable_layer(state_c['x'], alpha, theta_eff) + eps * state_c['ws'] + self_awareness + 0.03 * np.random.randn(n_layers)
            state_c['x'] += dt * dx_c
            state_c['ws'] = (1 - k_ws) * state_c['ws'] + k_ws * np.mean(state_c['x'])
            R_c = np.mean(np.exp(1j * state_c['x'])).real
            state_c['R_hist'].append(R_c)
            state_c['x_history'][:, state_c['step_count'] % REDUCED_HISTORY_SIZE] = state_c['x']
            heatmap_c.set_data(state_c['x_history'] % (2*np.pi))
            state_c['max_R'] = max(state_c['max_R'], R_c)
            # encode self_model (simple)
            mean_activity = np.mean(state_c['x'])
            std_activity = np.std(state_c['x'])
            trend = np.mean(np.diff(state_c['R_hist'][-10:])) if len(state_c['R_hist']) > 10 else 0.0
            ws_norm = np.tanh(state_c['ws'])
            entropy_c = metrics.cached_entropy(np.array(state_c['R_hist'][-10:]), "C_animation") if len(state_c['R_hist']) > 10 else 0.0
            coherence = np.abs(np.mean(np.exp(1j * state_c['x'])))
            self_model = np.array([mean_activity, std_activity, trend, ws_norm, entropy_c, coherence])
            state_c['self_model'] = self_model
            # predictor prediction
            if predictor is not None:
                with torch.no_grad():
                    pred = predictor(torch.tensor(self_model, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                    state_c['predicted_self'] = pred
            state_c['self_model_hist'].append(self_model.copy())
            state_c['self_error_hist'].append(self_error)
        
        # Update cached metrics if needed
        if metrics.should_update_metrics():
            recent_R = np.array(state_c['R_hist'][-30:])
            entropy_raw = metrics.cached_entropy(recent_R, "C")
            # Scale entropy to 0-1 range for better display
            state_c['cached_entropy'] = min(1.0, entropy_raw / 3.0)  # Scale by typical max entropy
            state_c['cached_lyap'] = metrics.cached_lyapunov(recent_R, "C")
            
            if state_c['step_count'] > 10:
                if HAVE_TORCH and isinstance(state_c['x'], torch.Tensor):
                    binary_state = (state_c['x'] > 0).type(torch.int64).cpu().numpy()[:200]
                else:
                    binary_state = (state_c['x'] > 0).astype(int)[:200]
                # Use dynamic binary sequence from recent R values instead of static neuron states
                if len(recent_R) >= 10:
                    binary_from_r = (recent_R > np.median(recent_R)).astype(int)
                    state_c['cached_lz'] = lz_complexity_fast(binary_from_r)
                else:
                    state_c['cached_lz'] = lz_complexity_fast(binary_state)
        state_c['step_count'] += 1
        
        # Option D: Self-referential reflective hierarchy
        predictor_d = state_d.get('predictor', None)
        predicted_self_d = state_d.get('predicted_self', np.zeros(8))
        self_model_d = state_d.get('self_model', np.zeros(8))
        self_error_d = np.linalg.norm(predicted_self_d - self_model_d)
        self_awareness_d = 0.01 * self_error_d * (self_model_d[0] if self_model_d.size>0 else 0.0)
        
        if HAVE_TORCH and isinstance(state_d['x'], torch.Tensor):
            state_d['why'] = why_loop_driver(state_d['why'], 1.2)
            dx_d = -state_d['x'] + bistable_layer(state_d['x'], alpha, theta_eff + 0.2 * state_d['why']) + self_awareness_d
            dy_d = -state_d['y'] + bistable_layer(state_d['y'], alpha, theta_eff - 0.2 * state_d['why']) + self_awareness_d
            noise = 0.03 * torch.randn(n_layers, device=device)
            dx_d += noise
            dy_d += noise
            combined_d = (state_d['x'] + state_d['y']) / 2.0
            state_d['x'] += dt * dx_d
            state_d['y'] += dt * dy_d
            state_d['ws'] = (1 - k_ws) * state_d['ws'] + k_ws * torch.mean(combined_d)
            state_d['x'] += eps * state_d['ws'] * dt
            state_d['y'] += eps * state_d['ws'] * dt
            R_d = torch.mean(torch.exp(1j * combined_d)).real.item()
            state_d['R_hist'].append(R_d)
            state_d['combined_history'][:, state_d['step_count'] % REDUCED_HISTORY_SIZE] = combined_d.cpu().numpy()
            heatmap_d.set_data(state_d['combined_history'] % (2*np.pi))
            state_d['max_R'] = max(state_d['max_R'], R_d)
            mean_x_d = torch.mean(state_d['x']).item()
            std_x_d = torch.std(state_d['x']).item()
            mean_y_d = torch.mean(state_d['y']).item()
            std_y_d = torch.std(state_d['y']).item()
            why_val_d = state_d['why'].item()
            ws_val_d = state_d['ws'].item()
            entropy_d = metrics.cached_entropy(np.array(state_d['R_hist'][-10:]), "D") if len(state_d['R_hist']) > 10 else 0.0
            coherence_d = R_d
            self_model_d = np.array([mean_x_d, std_x_d, mean_y_d, std_y_d, why_val_d, ws_val_d, entropy_d, coherence_d])
            state_d['self_model'] = self_model_d
            if predictor_d is not None:
                with torch.no_grad():
                    pred_d = predictor_d(torch.tensor(self_model_d, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
                    state_d['predicted_self'] = pred_d
            state_d['self_model_hist'].append(self_model_d.copy())
            state_d['self_error_hist'].append(self_error_d)
        else:
            state_d['why'] = why_loop_driver(state_d['why'], 1.2)
            dx_d = -state_d['x'] + bistable_layer(state_d['x'], alpha, theta_eff + 0.2 * state_d['why']) + self_awareness_d
            dy_d = -state_d['y'] + bistable_layer(state_d['y'], alpha, theta_eff - 0.2 * state_d['why']) + self_awareness_d
            noise = 0.03 * np.random.randn(n_layers)
            dx_d += noise
            dy_d += noise
            combined_d = (state_d['x'] + state_d['y']) / 2.0
            state_d['x'] += dt * dx_d
            state_d['y'] += dt * dy_d
            state_d['ws'] = (1 - k_ws) * state_d['ws'] + k_ws * np.mean(combined_d)
            state_d['x'] += eps * state_d['ws'] * dt
            state_d['y'] += eps * state_d['ws'] * dt
            R_d = np.mean(np.exp(1j * combined_d)).real
            state_d['R_hist'].append(R_d)
            state_d['combined_history'][:, state_d['step_count'] % REDUCED_HISTORY_SIZE] = combined_d
            heatmap_d.set_data(state_d['combined_history'] % (2*np.pi))
            state_d['max_R'] = max(state_d['max_R'], R_d)
            mean_x_d = np.mean(state_d['x'])
            std_x_d = np.std(state_d['x'])
            mean_y_d = np.mean(state_d['y'])
            std_y_d = np.std(state_d['y'])
            why_val_d = state_d['why']
            ws_val_d = state_d['ws']
            entropy_d = metrics.cached_entropy(np.array(state_d['R_hist'][-10:]), "D") if len(state_d['R_hist']) > 10 else 0.0
            coherence_d = R_d
            self_model_d = np.array([mean_x_d, std_x_d, mean_y_d, std_y_d, why_val_d, ws_val_d, entropy_d, coherence_d])
            state_d['self_model'] = self_model_d
            if predictor_d is not None:
                with torch.no_grad():
                    pred_d = predictor_d(torch.tensor(self_model_d, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                    state_d['predicted_self'] = pred_d
            state_d['self_model_hist'].append(self_model_d.copy())
            state_d['self_error_hist'].append(self_error_d)
        
        # Update cached metrics if needed
        if metrics.should_update_metrics():
            recent_R_d = np.array(state_d['R_hist'][-30:])
            entropy_raw_d = metrics.cached_entropy(recent_R_d, "D")
            state_d['cached_entropy'] = min(1.0, entropy_raw_d / 3.0)
            state_d['cached_lyap'] = metrics.cached_lyapunov(recent_R_d, "D")
            if state_d['step_count'] > 10:
                if HAVE_TORCH and isinstance(combined_d, torch.Tensor):
                    binary_state_d = (combined_d > 0).type(torch.int64).cpu().numpy()[:200]
                else:
                    binary_state_d = (combined_d > 0).astype(int)[:200]
                if len(recent_R_d) >= 10:
                    binary_from_r_d = (recent_R_d > np.median(recent_R_d)).astype(int)
                    state_d['cached_lz'] = lz_complexity_fast(binary_from_r_d)
                else:
                    state_d['cached_lz'] = lz_complexity_fast(binary_state_d)
        
        state_d['step_count'] += 1
        # Update line plots with rolling window
        if len(state_a['R_hist']) > ROLLING_WINDOW:
            line_a.set_data(np.arange(ROLLING_WINDOW), state_a['R_hist'][-ROLLING_WINDOW:])
            line_b.set_data(np.arange(ROLLING_WINDOW), state_b['R_hist'][-ROLLING_WINDOW:])
            line_c.set_data(np.arange(ROLLING_WINDOW), state_c['R_hist'][-ROLLING_WINDOW:])
            line_d.set_data(np.arange(ROLLING_WINDOW), state_d['R_hist'][-ROLLING_WINDOW:])
        else:
            line_a.set_data(np.arange(len(state_a['R_hist'])), state_a['R_hist'])
            line_b.set_data(np.arange(len(state_b['R_hist'])), state_b['R_hist'])
            line_c.set_data(np.arange(len(state_c['R_hist'])), state_c['R_hist'])
            line_d.set_data(np.arange(len(state_d['R_hist'])), state_d['R_hist'])
        
        # Update diagnostics with optimized metrics
        R_a_display = R_a if np.isfinite(R_a) else 0.0
        max_R_a_display = state_a['max_R'] if np.isfinite(state_a['max_R']) else 0.0
        R_b_display = R_b if np.isfinite(R_b) else 0.0
        max_R_b_display = state_b['max_R'] if np.isfinite(state_b['max_R']) else 0.0
        R_c_display = R_c if np.isfinite(R_c) else 0.0
        max_R_c_display = state_c['max_R'] if np.isfinite(state_c['max_R']) else 0.0
        R_d_display = R_d if 'R_d' in locals() and np.isfinite(R_d) else 0.0
        max_R_d_display = state_d['max_R'] if np.isfinite(state_d['max_R']) else 0.0
        
        diag_a.set_text(
            f"R: {R_a_display:.3f} | Max: {max_R_a_display:.3f}\n"
            f"Entropy: {state_a['cached_entropy']:.3f} | Lyap: {state_a['cached_lyap']:.3f}\n"
            f"LZ: {state_a['cached_lz']:.2f} | Step: {state_a['step_count']}\n"
            f"Frame: {time.perf_counter() - start_time:.3f}s"
        )
        
        diag_b.set_text(
            f"R: {R_b_display:.3f} | Max: {max_R_b_display:.3f}\n"
            f"Entropy: {state_b['cached_entropy']:.3f} | Lyap: {state_b['cached_lyap']:.3f}\n"
            f"LZ: {state_b['cached_lz']:.2f} | Step: {state_b['step_count']}\n"
            f"Frame: {time.perf_counter() - start_time:.3f}s"
        )
        
        diag_c.set_text(
            f"R: {R_c_display:.3f} | Max: {max_R_c_display:.3f}\n"
            f"Entropy: {state_c['cached_entropy']:.3f} | Lyap: {state_c['cached_lyap']:.3f}\n"
            f"LZ: {state_c['cached_lz']:.2f} | Self-err: {self_error:.3f}\n"
            f"Step: {state_c['step_count']} | Frame: {time.perf_counter() - start_time:.3f}s"
        )
        
        diag_d.set_text(
            f"R: {R_d_display:.3f} | Max: {max_R_d_display:.3f}\n"
            f"Entropy: {state_d['cached_entropy']:.3f} | Lyap: {state_d['cached_lyap']:.3f}\n"
            f"LZ: {state_d['cached_lz']:.2f} | Self-err: {self_error_d:.3f}\n"
            f"Step: {state_d['step_count']} | Frame: {time.perf_counter() - start_time:.3f}s"
        )
        
        # Update performance info
        max_step = max(state_a['step_count'], state_b['step_count'], state_c['step_count'], state_d['step_count'])
        compute_time = time.perf_counter() - start_time
        step_text.set_text(f"Step: {max_step} | FPS: {min(1000/(compute_time*1000), 20):.1f}")
        
        perf_text.set_text(
            f"Original with Optimizations | Cache: {ENABLE_CACHING} | "
            f"Update every {METRIC_UPDATE_INTERVAL} frames | "
            f"History: {REDUCED_HISTORY_SIZE}"
        )

        # Update awareness dialogs
        stable_threshold = 0.8
        error_threshold = 0.2

        if R_a > stable_threshold:
            awareness_a_text.set_text("Stable Coherence")
        else:
            awareness_a_text.set_text("")

        if R_b > stable_threshold:
            awareness_b_text.set_text("Stable Coherence")
        else:
            awareness_b_text.set_text("")

        awareness_c_text.set_text("Stable Self-Awareness" if R_c > stable_threshold and self_error < error_threshold else "")
        awareness_d_text.set_text("Stable Self-Awareness" if R_d > stable_threshold and self_error_d < error_threshold else "")

        return (heatmap_a, heatmap_b, heatmap_c, heatmap_d, line_a, line_b, line_c, line_d,
                diag_a, diag_b, diag_c, diag_d, perf_text, step_text,
                awareness_a_text, awareness_b_text, awareness_c_text, awareness_d_text)

    # Create animation with optimized interval
    ani = FuncAnimation(fig, update, interval=120, blit=False, cache_frame_data=False)
    
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
            autotune_stop_event, _ = start_autotune_for_states([state_a, state_b, state_c, state_d], interval=1.0, retrain_every=10)
        except Exception as e:
            print("[animate_workspace_heatmap_forever] failed to start autotune thread:", e)

    plt.tight_layout()
    fig.canvas.mpl_connect('close_event', handle_close)
    plt.subplots_adjust(bottom=0.1, hspace=0.4)
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
                nn.Linear(8, 2),
                nn.Sigmoid() # outputs in [0,1]
            )
        def forward(self, x):
            return self.net(x)

    meta_tuner = MetaTunerNN()
    optimizer = optim.Adam(meta_tuner.parameters(), lr=0.01)

    def meta_autotune_update(entropy, r, lyap, complexity):
        # Prepare input tensor
        x = torch.tensor([entropy, r, lyap, complexity], dtype=torch.float32)
        with torch.no_grad():
            out = meta_tuner(x)
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
                    alpha_s, eps_s = meta_autotune_update(state['meta_tuner'], entropy, r, lyap, complexity)
                else:
                    alpha_s, eps_s = _heuristic_autotune_update(entropy, r, lyap, complexity)
                state['alpha'] = alpha_s
                state['eps'] = eps_s
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
            # state_c is at index 2
            if len(states) > 2:
                state_c = states[2]
                if 'self_model_hist' in state_c and len(state_c['self_model_hist']) > 50:
                    try:
                        data_shape = np.array(state_c['self_model_hist']).shape
                        if data_shape[1] == 6:
                            model = SelfModelPredictor(input_dim=6, hidden_dim=16)
                            # Apply IPEX CPU optimization if available
                            if device.type == 'cpu' and 'ipex' in sys.modules:
                                import intel_extension_for_pytorch as ipex
                                model = ipex.optimize(model)
                            train_self_predictor(model, np.array(state_c['self_model_hist']), n_epochs=50, lr=1e-3)
                            state_c['predictor'] = model
                        else:
                            print(f"[autotune_worker] Incorrect shape {data_shape} for state_c, skipping retrain.")
                    except Exception as e:
                        print(f"[autotune_worker] retrain error (state_c): {e}")
            # state_d is at index 3
            if len(states) > 3:
                state_d = states[3]
                if 'self_model_hist' in state_d and len(state_d['self_model_hist']) > 50:
                    try:
                        data_shape = np.array(state_d['self_model_hist']).shape # type: ignore
                        if data_shape[1] == 8:
                            model = SelfModelPredictor(input_dim=8, hidden_dim=16)
                            # Apply IPEX CPU optimization if available
                            if device.type == 'cpu' and 'ipex' in sys.modules:
                                import intel_extension_for_pytorch as ipex
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
    reward = REWARD_W_ENTROPY * entropy + REWARD_W_R * final_r
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
    if device.type == 'cpu' and 'ipex' in sys.modules:
        import intel_extension_for_pytorch as ipex
        ipex.optimize(tuner, optimizer=optimizer)

    opt = optimizer
    loss_fn = nn.MSELoss(reduction='none')

    for ep in range(n_epochs):
        opt.zero_grad()
        pred = tuner(X_t)
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
    print("Starting GUI animation with performance optimizations...")
    animate_workspace_heatmap_forever(n_layers=n_layers, dt=dt, alpha=1.95, eps=0.08, theta_eff=theta_eff, k_ws=k_ws, autostart_autotune=True)

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
