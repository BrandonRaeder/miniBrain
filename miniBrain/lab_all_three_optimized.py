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
import time
from collections import deque

# Performance optimization configuration
ENABLE_CACHING = True
CACHE_SIZE = 30
METRIC_UPDATE_INTERVAL = 3  # Update heavy metrics every 3 frames
REDUCED_HISTORY_SIZE = 800  # Reduced from 2000
ROLLING_WINDOW = 150  # Reduced from 500

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

# Global optimizer
metrics = OptimizedMetrics()

# ---------- Optimized bistable layer ----------
def bistable_layer(x, alpha, theta_eff):
    """Vectorized bistable activation"""
    return np.tanh(alpha * x - theta_eff)

# ---------- Optimized why-loop driver ----------
def why_loop_driver(y, gamma):
    """Optimized reflective recursion"""
    return np.tanh(gamma * y)

# ---------- Fast LZ complexity ----------
def lz_complexity_fast(binary_sequence, max_length=300):
    """Simplified LZ complexity for performance"""
    if len(binary_sequence) == 0:
        return 0
    
    # Convert to string and limit length for performance
    s = ''.join(['1' if x else '0' for x in binary_sequence[:max_length]])
    n = len(s)
    
    if n == 0:
        return 0
    
    # Very simplified LZ algorithm
    i = 0
    c = 1
    l = 1
    
    while i + l <= n and c < 50:  # Limit iterations for performance
        found = False
        # Simple substring matching (optimized)
        for j in range(max(0, i - l + 1), i):
            if j + l <= n and s[j:j+l] == s[i:i+l]:
                found = True
                break
        
        if found:
            l += 1
        else:
            c += 1
            i += l
            l = 1
    
    return c

# ---------- OPTION A: Optimized Global Workspace ----------
class OptimizedWorkspaceSim:
    def __init__(self, n_layers=100, dt=0.05, alpha=1.95, eps=0.08, theta_eff=0.0, k_ws=0.002):
        self.n_layers = n_layers
        self.dt = dt
        self.alpha = alpha
        self.eps = eps
        self.theta_eff = theta_eff
        self.k_ws = k_ws
        
        # State
        self.x = np.random.randn(n_layers)
        self.ws = 0.0
        self.R_hist = []
        self.x_history = np.zeros((n_layers, REDUCED_HISTORY_SIZE))
        self.step_count = 0
        self.max_R = -np.inf
        
        # Cached metrics
        self.cached_entropy = 0.0
        self.cached_lyap = 0.0
        self.cached_lz = 0.0
    
    def step(self):
        """Single simulation step"""
        # Add controlled noise
        noise = 0.03 * np.random.randn(self.n_layers)
        
        # Dynamics
        dx = -self.x + bistable_layer(self.x, self.alpha, self.theta_eff) + self.eps * self.ws + noise
        self.x += self.dt * dx
        
        # Workspace update
        self.ws = (1 - self.k_ws) * self.ws + self.k_ws * np.mean(self.x)
        
        # Record
        R = np.mean(np.exp(1j * self.x)).real
        self.R_hist.append(R)
        self.max_R = max(self.max_R, R)
        
        # Update history (circular buffer)
        self.x_history[:, self.step_count % REDUCED_HISTORY_SIZE] = self.x
        self.step_count += 1
        
        # Update cached metrics if needed
        if metrics.should_update_metrics():
            recent_R = np.array(self.R_hist[-30:])
            self.cached_entropy = metrics.cached_entropy(recent_R, "A")
            self.cached_lyap = metrics.cached_lyapunov(recent_R, "A")
            
            # LZ complexity on current state (limited scope)
            if self.step_count > 10:
                binary_state = (self.x > 0).astype(int)[:200]  # Limit to first 200 neurons
                self.cached_lz = lz_complexity_fast(binary_state)
        
        return R, self.x.copy()
    
    def get_diagnostics(self):
        """Get diagnostic info for display"""
        current_R = self.R_hist[-1] if self.R_hist else 0.0
        max_R = self.max_R if np.isfinite(self.max_R) else 0.0
        
        return {
            'current_R': current_R,
            'max_R': max_R,
            'entropy': self.cached_entropy,
            'lyapunov': self.cached_lyap,
            'lz': self.cached_lz,
            'step': self.step_count
        }

# ---------- OPTION B: Optimized Reflective Hierarchy ----------
class OptimizedReflectiveSim:
    def __init__(self, n_layers=100, dt=0.05, alpha=1.95, eps=0.08, theta_eff=0.0, k_ws=0.002, gamma=1.2):
        self.n_layers = n_layers
        self.dt = dt
        self.alpha = alpha
        self.eps = eps
        self.theta_eff = theta_eff
        self.k_ws = k_ws
        self.gamma = gamma
        
        # State
        self.x = np.random.randn(n_layers)
        self.y = np.random.randn(n_layers)
        self.why = 0.0
        self.ws = 0.0
        self.R_hist = []
        self.combined_history = np.zeros((n_layers, REDUCED_HISTORY_SIZE))
        self.step_count = 0
        self.max_R = -np.inf
        
        # Cached metrics
        self.cached_entropy = 0.0
        self.cached_lyap = 0.0
        self.cached_lz = 0.0
    
    def step(self):
        """Single simulation step with reflective dynamics"""
        # Why-loop driver
        self.why = why_loop_driver(self.why, self.gamma)
        
        # Dual system dynamics
        dx = -self.x + bistable_layer(self.x, self.alpha, self.theta_eff + 0.2 * self.why)
        dy = -self.y + bistable_layer(self.y, self.alpha, self.theta_eff - 0.2 * self.why)
        
        # Add controlled noise
        noise = 0.03 * np.random.randn(self.n_layers)
        dx += noise
        dy += noise
        
        # Combine systems
        combined = (self.x + self.y) / 2.0
        
        # Update
        self.x += self.dt * dx
        self.y += self.dt * dy
        self.ws = (1 - self.k_ws) * self.ws + self.k_ws * np.mean(combined)
        
        # Workspace feedback
        self.x += self.eps * self.ws * self.dt
        self.y += self.eps * self.ws * self.dt
        
        # Record
        R = np.mean(np.exp(1j * combined)).real
        self.R_hist.append(R)
        self.max_R = max(self.max_R, R)
        
        # Update history
        self.combined_history[:, self.step_count % REDUCED_HISTORY_SIZE] = combined
        self.step_count += 1
        
        # Update cached metrics
        if metrics.should_update_metrics():
            recent_R = np.array(self.R_hist[-30:])
            self.cached_entropy = metrics.cached_entropy(recent_R, "B")
            self.cached_lyap = metrics.cached_lyapunov(recent_R, "B")
            
            if self.step_count > 10:
                binary_state = (combined > 0).astype(int)[:200]
                self.cached_lz = lz_complexity_fast(binary_state)
        
        return R, combined.copy()
    
    def get_diagnostics(self):
        """Get diagnostic info for display"""
        current_R = self.R_hist[-1] if self.R_hist else 0.0
        max_R = self.max_R if np.isfinite(self.max_R) else 0.0
        
        return {
            'current_R': current_R,
            'max_R': max_R,
            'entropy': self.cached_entropy,
            'lyapunov': self.cached_lyap,
            'lz': self.cached_lz,
            'step': self.step_count,
            'why': self.why
        }

# ---------- OPTION C: Optimized Self-Referential ----------
class OptimizedSelfReferentialSim:
    def __init__(self, n_layers=100, dt=0.05, alpha=1.95, eps=0.08, theta_eff=0.0, k_ws=0.002, meta_lr=0.01):
        self.n_layers = n_layers
        self.dt = dt
        self.alpha = alpha
        self.eps = eps
        self.theta_eff = theta_eff
        self.k_ws = k_ws
        self.meta_lr = meta_lr
        
        # State
        self.x = np.random.randn(n_layers)
        self.ws = 0.0
        self.R_hist = []
        self.x_history = np.zeros((n_layers, REDUCED_HISTORY_SIZE))
        self.step_count = 0
        self.max_R = -np.inf
        
        # Self-model components
        self.self_model = np.zeros(6)
        self.predicted_self = np.zeros(6)
        self.self_model_hist = []
        self.self_error_hist = []
        
        # Cached metrics
        self.cached_entropy = 0.0
        self.cached_lyap = 0.0
        self.cached_lz = 0.0
    
    def step(self):
        """Single simulation step with self-referential dynamics"""
        # Self-reference computation (simplified)
        self_error = np.linalg.norm(self.predicted_self - self.self_model)
        self_awareness = self.meta_lr * self_error * (self.self_model[0] if len(self.self_model) > 0 else 0.0)
        
        # Dynamics with self-awareness
        noise = 0.03 * np.random.randn(self.n_layers)
        dx = -self.x + bistable_layer(self.x, self.alpha, self.theta_eff) + self.eps * self.ws + self_awareness + noise
        self.x += self.dt * dx
        
        # Workspace update
        self.ws = (1 - self.k_ws) * self.ws + self.k_ws * np.mean(self.x)
        
        # Record
        R = np.mean(np.exp(1j * self.x)).real
        self.R_hist.append(R)
        self.max_R = max(self.max_R, R)
        
        # Update history
        self.x_history[:, self.step_count % REDUCED_HISTORY_SIZE] = self.x
        self.step_count += 1
        
        # Self-model encoding (simplified)
        mean_activity = np.mean(self.x)
        std_activity = np.std(self.x)
        trend = np.mean(np.diff(self.R_hist[-10:])) if len(self.R_hist) > 10 else 0.0
        ws_norm = np.tanh(self.ws)
        entropy_simple = -np.sum([r * np.log(r + 1e-10) for r in self.R_hist[-10:] if r > 0]) if len(self.R_hist) > 10 else 0.0
        coherence = np.abs(np.mean(np.exp(1j * self.x)))
        
        self.self_model = np.array([mean_activity, std_activity, trend, ws_norm, entropy_simple, coherence])
        self.self_model_hist.append(self.self_model.copy())
        self.self_error_hist.append(self_error)
        
        # Simplified prediction (no actual neural net, just trend)
        if len(self.self_model_hist) > 5:
            recent_models = np.array(self.self_model_hist[-5:])
            self.predicted_self = recent_models.mean(axis=0) + 0.1 * np.random.randn(6)
        
        # Update cached metrics
        if metrics.should_update_metrics():
            recent_R = np.array(self.R_hist[-30:])
            self.cached_entropy = metrics.cached_entropy(recent_R, "C")
            self.cached_lyap = metrics.cached_lyapunov(recent_R, "C")
            
            if self.step_count > 10:
                binary_state = (self.x > 0).astype(int)[:200]
                self.cached_lz = lz_complexity_fast(binary_state)
        
        return R, self.x.copy()
    
    def get_diagnostics(self):
        """Get diagnostic info for display"""
        current_R = self.R_hist[-1] if self.R_hist else 0.0
        max_R = self.max_R if np.isfinite(self.max_R) else 0.0
        self_error = self.self_error_hist[-1] if self.self_error_hist else 0.0
        
        return {
            'current_R': current_R,
            'max_R': max_R,
            'entropy': self.cached_entropy,
            'lyapunov': self.cached_lyap,
            'lz': self.cached_lz,
            'step': self.step_count,
            'self_error': self_error
        }

# ---------- UNIFIED OPTIMIZED ANIMATION ----------
def animate_all_three_optimized(n_layers=100, dt=0.08,
                               alpha=1.95, eps=0.08, theta_eff=0.0, k_ws=0.002,
                               autostart_autotune=False):
    """Optimized animation of all three workspace models"""
    
    # Initialize all three optimized simulations
    sim_a = OptimizedWorkspaceSim(n_layers, dt, alpha, eps, theta_eff, k_ws)
    sim_b = OptimizedReflectiveSim(n_layers, dt, alpha, eps, theta_eff, k_ws)
    sim_c = OptimizedSelfReferentialSim(n_layers, dt, alpha, eps, theta_eff, k_ws)
    
    # Setup figure with optimized layout
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Common heatmap settings
    heatmap_extent = [0, REDUCED_HISTORY_SIZE, 0, n_layers]
    
    # Option A setup
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
    
    # Option B setup
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
    
    # Option C setup
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
    
    # Global performance info
    perf_text = fig.text(0.02, 0.02, "Optimized Performance Mode", fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    step_text = fig.text(0.98, 0.02, "Step: 0", ha='right', va='bottom', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def update(frame):
        start_time = time.perf_counter()
        
        # Step all three simulations
        R_a, x_a = sim_a.step()
        R_b, x_b = sim_b.step()
        R_c, x_c = sim_c.step()
        
        # Update heatmaps
        heatmap_a.set_data(sim_a.x_history % (2*np.pi))
        heatmap_b.set_data(sim_b.combined_history % (2*np.pi))
        heatmap_c.set_data(sim_c.x_history % (2*np.pi))
        
        # Update line plots with rolling window
        if len(sim_a.R_hist) > ROLLING_WINDOW:
            line_a.set_data(np.arange(ROLLING_WINDOW), sim_a.R_hist[-ROLLING_WINDOW:])
            line_b.set_data(np.arange(ROLLING_WINDOW), sim_b.R_hist[-ROLLING_WINDOW:])
            line_c.set_data(np.arange(ROLLING_WINDOW), sim_c.R_hist[-ROLLING_WINDOW:])
        else:
            line_a.set_data(np.arange(len(sim_a.R_hist)), sim_a.R_hist)
            line_b.set_data(np.arange(len(sim_b.R_hist)), sim_b.R_hist)
            line_c.set_data(np.arange(len(sim_c.R_hist)), sim_c.R_hist)
        
        # Update diagnostics
        diag_a_data = sim_a.get_diagnostics()
        diag_b_data = sim_b.get_diagnostics()
        diag_c_data = sim_c.get_diagnostics()
        
        compute_time = time.perf_counter() - start_time
        
        diag_a.set_text(
            f"R: {diag_a_data['current_R']:.3f} | Max: {diag_a_data['max_R']:.3f}\n"
            f"Entropy: {diag_a_data['entropy']:.3f} | Lyap: {diag_a_data['lyapunov']:.3f}\n"
            f"LZ: {diag_a_data['lz']:.2f} | Step: {diag_a_data['step']}\n"
            f"Frame: {compute_time:.3f}s"
        )
        
        diag_b.set_text(
            f"R: {diag_b_data['current_R']:.3f} | Max: {diag_b_data['max_R']:.3f}\n"
            f"Entropy: {diag_b_data['entropy']:.3f} | Lyap: {diag_b_data['lyapunov']:.3f}\n"
            f"LZ: {diag_b_data['lz']:.2f} | Why: {diag_b_data['why']:.3f}\n"
            f"Step: {diag_b_data['step']} | Frame: {compute_time:.3f}s"
        )
        
        diag_c.set_text(
            f"R: {diag_c_data['current_R']:.3f} | Max: {diag_c_data['max_R']:.3f}\n"
            f"Entropy: {diag_c_data['entropy']:.3f} | Lyap: {diag_c_data['lyapunov']:.3f}\n"
            f"LZ: {diag_c_data['lz']:.2f} | Self-err: {diag_c_data['self_error']:.3f}\n"
            f"Step: {diag_c_data['step']} | Frame: {compute_time:.3f}s"
        )
        
        # Update performance info
        max_step = max(diag_a_data['step'], diag_b_data['step'], diag_c_data['step'])
        step_text.set_text(f"Step: {max_step} | FPS: {min(1000/(compute_time*1000), 20):.1f}")
        
        perf_text.set_text(
            f"3-Model Optimized | Cache: {ENABLE_CACHING} | "
            f"Update every {METRIC_UPDATE_INTERVAL} frames | "
            f"History: {REDUCED_HISTORY_SIZE}"
        )
        
        return (heatmap_a, heatmap_b, heatmap_c, line_a, line_b, line_c, 
                diag_a, diag_b, diag_c, perf_text, step_text)
    
    # Create animation with optimized interval
    ani = FuncAnimation(fig, update, interval=120, blit=False, cache_frame_data=False)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, hspace=0.4)
    plt.show()

# ---------- Performance comparison function ----------
def compare_all_three_performance():
    """Compare performance of all three optimized models"""
    print("=== THREE-MODEL PERFORMANCE COMPARISON ===\n")
    
    n_layers = 50
    T = 500
    dt = 0.05
    
    print("Running performance test for 500 steps...")
    
    # Test each model individually
    models = [
        ("Option A (Global Workspace)", OptimizedWorkspaceSim),
        ("Option B (Reflective Hierarchy)", OptimizedReflectiveSim),
        ("Option C (Self-Referential)", OptimizedSelfReferentialSim)
    ]
    
    for name, model_class in models:
        print(f"\nTesting {name}:")
        
        start_time = time.perf_counter()
        sim = model_class(n_layers, dt)
        
        for _ in range(T):
            sim.step()
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        per_step = (duration / T) * 1000
        
        print(f"  Total time: {duration:.3f}s")
        print(f"  Per step: {per_step:.2f}ms")
        print(f"  Estimated FPS: {1000/per_step:.1f}")
        
        # Show some diagnostics
        diag = sim.get_diagnostics()
        print(f"  Final R: {diag['current_R']:.3f}")
        print(f"  Max R: {diag['max_R']:.3f}")
    
    print(f"\n=== OPTIMIZATION SUMMARY ===")
    print("Key improvements applied to all three models:")
    print(f"• Reduced history buffer: 2000 -> {REDUCED_HISTORY_SIZE} steps")
    print(f"• Smaller rolling window: 500 -> {ROLLING_WINDOW}")
    print(f"• Cached metrics: {ENABLE_CACHING}")
    print(f"• Update interval: every {METRIC_UPDATE_INTERVAL} frames")
    print(f"• Optimized LZ complexity (limited scope)")
    print(f"• Reduced noise and simplified dynamics")
    print(f"• Efficient circular buffer updates")
    print(f"• Streamlined diagnostic computations")
    print(f"\nExpected result: Smooth animation at ~8-10 FPS")

if __name__ == "__main__":
    print("Starting optimized 3-model neural workspace simulation...")
    print("Performance optimizations:")
    print(f"- Caching enabled: {ENABLE_CACHING}")
    print(f"- Metric update interval: {METRIC_UPDATE_INTERVAL}")
    print(f"- Reduced history size: {REDUCED_HISTORY_SIZE}")
    print(f"- Rolling window: {ROLLING_WINDOW}")
    print(f"- Animation interval: 120ms (~8 FPS)")
    
    # Run performance comparison
    compare_all_three_performance()
    
    print(f"\nStarting real-time animation...")
    animate_all_three_optimized(
        n_layers=100,
        dt=0.08,  # Slightly slower for stability
        alpha=1.95,
        eps=0.08,
        theta_eff=0.0,
        k_ws=0.002
    )