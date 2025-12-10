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

# Performance optimization flags
ENABLE_CACHING = True
CACHE_SIZE = 50  # Cache entropy/LZ calculations to avoid recomputation
METRIC_UPDATE_INTERVAL = 5  # Update heavy metrics every N frames instead of every frame

class PerformanceOptimizer:
    def __init__(self):
        self.entropy_cache = {}
        self.lyap_cache = {}
        self.lz_cache = {}
        self.frame_counter = 0
        
    def cached_entropy(self, data, key=None):
        """Cache entropy calculations to avoid recomputation"""
        if not ENABLE_CACHING:
            return self._compute_entropy(data)
        
        if key is None:
            key = str(len(data)) + "_" + str(hash(str(data[-10:])))  # Use last 10 values as key
        
        if key in self.entropy_cache:
            return self.entropy_cache[key]
        
        result = self._compute_entropy(data)
        self.entropy_cache[key] = result
        
        # Limit cache size
        if len(self.entropy_cache) > CACHE_SIZE:
            self.entropy_cache.clear()
            
        return result
    
    def _compute_entropy(self, data, bins=30):  # Reduced bins for speed
        """Optimized entropy computation"""
        if len(data) < 5:
            return 0.0
        hist, _ = np.histogram(data[-50:], bins=bins, density=False)  # Use last 50 values only
        if np.sum(hist) == 0:
            return 0.0
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist + 1e-10))
    
    def cached_lyapunov(self, data):
        """Cache Lyapunov proxy calculations"""
        if not ENABLE_CACHING:
            return self._compute_lyapunov(data)
            
        key = str(len(data))
        if key in self.lyap_cache:
            return self.lyap_cache[key]
            
        result = self._compute_lyapunov(data)
        self.lyap_cache[key] = result
        
        if len(self.lyap_cache) > CACHE_SIZE:
            self.lyap_cache.clear()
            
        return result
    
    def _compute_lyapunov(self, data):
        """Optimized Lyapunov proxy computation"""
        if len(data) < 2:
            return 0.0
        diffs = np.abs(np.diff(data[-20:]))  # Use last 20 values only
        return np.mean(diffs)
    
    def should_update_metrics(self):
        """Decide whether to update heavy metrics this frame"""
        self.frame_counter += 1
        return self.frame_counter % METRIC_UPDATE_INTERVAL == 0

# Global optimizer instance
optimizer = PerformanceOptimizer()

# ---------- Optimized bistable layer ----------
def bistable_layer(x, alpha, theta_eff):
    """Vectorized bistable activation"""
    return np.tanh(alpha * x - theta_eff)

# ---------- Simplified why-loop driver ----------
def why_loop_driver(y, gamma):
    """Simplified reflective recursion"""
    return np.tanh(gamma * y)

# ---------- Streamlined simulation functions ----------
def simulate_workspace_optimized(n_layers=100, T=1000, dt=0.01,
                                alpha=0.8, eps=0.7, theta_eff=0.3, k_ws=0.05):
    """Optimized workspace simulation with reduced computations"""
    x = np.random.randn(n_layers)
    ws = 0.0
    R_hist, ws_hist = [], []
    x_hist = []
    
    for t in range(T):
        # Local dynamics (vectorized)
        dx = -x + bistable_layer(x, alpha, theta_eff) + eps * ws
        x += dt * dx
        
        # Workspace update
        ws = (1 - k_ws) * ws + k_ws * np.mean(x)
        
        # Record (simplified)
        R_hist.append(np.mean(np.exp(1j * x)).real)
        ws_hist.append(ws)
        x_hist.append(x.copy())
    
    return np.array(R_hist), np.array(ws_hist), np.array(x_hist)

# ---------- Optimized real-time animation ----------
def animate_workspace_heatmap_optimized(n_layers=100, dt=0.1,  # Increased dt for stability
                                      alpha=1.95, eps=0.08, theta_eff=0.0, k_ws=0.002,
                                      autostart_autotune=False, rolling_window=200):  # Reduced rolling window
    
    # Optimized state management
    state = {
        'x': np.random.randn(n_layers),
        'ws': 0.0,
        'R_hist': [],
        'x_history': np.zeros((n_layers, 1000)),  # Reduced history size
        'step': 0,
        'max_R': -np.inf,
        'last_entropy': 0.0,
        'last_lyap': 0.0,
        'last_lz': 0.0
    }
    
    # Simplified figure layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Heatmap setup
    heatmap = ax1.imshow(np.zeros((n_layers, 1000)), aspect='auto', 
                        cmap='hsv', vmin=0, vmax=2*np.pi, origin='lower')
    ax1.set_title('Neural Phase Heatmap (Optimized)')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Neuron Index')
    
    # Line plot setup
    line, = ax2.plot([], [], 'r')
    ax2.set_xlim(0, rolling_window)
    ax2.set_ylim(-1.01, 1.01)
    ax2.set_title('Global Coherence R (Rolling Window)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('R')
    
    # Performance text
    perf_text = ax1.text(0.02, 0.98, "", transform=ax1.transAxes, fontsize=10, 
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Step counter
    step_text = fig.text(0.92, 0.96, "Step: 0", ha='right', va='top', fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    def update(frame):
        start_time = time.time()
        
        # Simulation step
        noise = 0.05 * np.random.randn(n_layers)  # Reduced noise
        dx = -state['x'] + bistable_layer(state['x'], alpha, theta_eff) + eps * state['ws'] + noise
        state['x'] += dt * dx
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * np.mean(state['x'])
        
        # Compute coherence
        R = np.mean(np.exp(1j * state['x'])).real
        state['R_hist'].append(R)
        state['max_R'] = max(state['max_R'], R)
        
        # Update history (circular buffer)
        state['x_history'][:, state['step'] % 1000] = state['x']
        
        # Update visualizations (every frame)
        heatmap.set_data(state['x_history'] % (2*np.pi))
        
        # Rolling window for line plot
        if len(state['R_hist']) > rolling_window:
            line.set_data(np.arange(rolling_window), state['R_hist'][-rolling_window:])
            ax2.set_xlim(0, rolling_window)
        else:
            line.set_data(np.arange(len(state['R_hist'])), state['R_hist'])
            ax2.set_xlim(0, rolling_window)
        
        # Heavy computations (cached and less frequent)
        if optimizer.should_update_metrics():
            # Use cached computations on shorter data segments
            recent_R = np.array(state['R_hist'][-100:])  # Last 100 values only
            state['last_entropy'] = optimizer.cached_entropy(recent_R)
            state['last_lyap'] = optimizer.cached_lyapunov(recent_R)
            
            # Simplified LZ complexity (binary only, shorter sequence)
            if len(state['R_hist']) > 10:
                binary_seq = (state['x_history'][:, state['step'] % 1000] > 0).astype(int)
                state['last_lz'] = lz_complexity_fast(binary_seq.flatten()[:500])  # First 500 elements
            else:
                state['last_lz'] = 0.0
        
        # Update performance text
        compute_time = time.time() - start_time
        perf_text.set_text(
            f"Current R: {R:.4f}\n"
            f"Highest R: {state['max_R']:.4f}\n"
            f"Entropy: {state['last_entropy']:.3f}\n"
            f"Lyapunov: {state['last_lyap']:.4f}\n"
            f"LZ Complexity: {state['last_lz']:.3f}\n"
            f"Compute Time: {compute_time:.3f}s"
        )
        
        # Update step counter
        step_text.set_text(f"Step: {state['step']}")
        state['step'] += 1
        
        return heatmap, line, perf_text, step_text
    
    # Create animation with longer interval for stability
    ani = FuncAnimation(fig, update, interval=200, blit=False, cache_frame_data=False)  # 200ms = 5 FPS
    
    plt.tight_layout()
    plt.show()

# ---------- Fast LZ complexity (simplified) ----------
def lz_complexity_fast(binary_sequence, max_length=1000):
    """Simplified LZ complexity for performance"""
    if len(binary_sequence) == 0:
        return 0
    
    # Convert to string and limit length
    s = ''.join(['1' if x else '0' for x in binary_sequence[:max_length]])
    n = len(s)
    
    if n == 0:
        return 0
    
    # Simplified LZ76 - much faster
    i = 0
    c = 1
    l = 1
    
    while i + l <= n:
        if s[i:i+l] == s[i-l:i] if i >= l else False:
            l += 1
        else:
            c += 1
            i += l
            l = 1
    
    return c

# ---------- Simplified diagnostics ----------
def shannon_entropy_fast(data, bins=20):  # Fewer bins for speed
    """Optimized entropy computation"""
    if len(data) < 5:
        return 0.0
    # Use last 30 values only for speed
    recent_data = data[-30:] if len(data) > 30 else data
    hist, _ = np.histogram(recent_data, bins=bins, density=False)
    if np.sum(hist) == 0:
        return 0.0
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist + 1e-10))

def lyapunov_proxy_fast(data):
    """Optimized Lyapunov proxy"""
    if len(data) < 2:
        return 0.0
    # Use last 15 values only
    recent_data = data[-15:] if len(data) > 15 else data
    diffs = np.abs(np.diff(recent_data))
    return np.mean(diffs)

# ---------- Quick performance test ----------
def performance_test():
    """Test the performance improvements"""
    print("Running performance test...")
    
    start_time = time.time()
    
    # Test the optimized simulation
    R_hist, ws_hist, x_hist = simulate_workspace_optimized(n_layers=50, T=1000, dt=0.1)
    
    simulation_time = time.time() - start_time
    print(f"Simulation time: {simulation_time:.3f}s for 1000 steps")
    print(f"Time per step: {simulation_time/1000*1000:.2f}ms")
    
    # Test metric computations
    start_time = time.time()
    entropy = shannon_entropy_fast(R_hist)
    lyap = lyapunov_proxy_fast(R_hist)
    lz = lz_complexity_fast((x_hist[-1] > 0).astype(int))
    metrics_time = time.time() - start_time
    
    print(f"Metrics computation time: {metrics_time:.3f}s")
    print(f"Entropy: {entropy:.3f}, Lyapunov: {lyap:.3f}, LZ: {lz}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(R_hist)
    plt.title('Coherence R (Optimized)')
    plt.xlabel('Step')
    plt.ylabel('R')
    
    plt.subplot(1, 2, 2)
    plt.imshow(x_hist.T, aspect='auto', cmap='hsv', vmin=0, vmax=2*np.pi)
    plt.title('Neural Activity Heatmap (Optimized)')
    plt.xlabel('Neuron Index')
    plt.ylabel('Time Step')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Starting optimized neural workspace simulation...")
    print("Performance optimizations enabled:")
    print(f"- Caching: {ENABLE_CACHING}")
    print(f"- Metric update interval: {METRIC_UPDATE_INTERVAL}")
    print(f"- Reduced animation rate: 200ms intervals")
    print(f"- Simplified computations")
    
    # Run performance test first
    performance_test()
    
    # Then run the real-time animation
    animate_workspace_heatmap_optimized(
        n_layers=100, 
        dt=0.1,  # Slower, more stable dynamics
        alpha=1.95, 
        eps=0.08, 
        theta_eff=0.0, 
        k_ws=0.002
    )