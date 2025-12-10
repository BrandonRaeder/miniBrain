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

# ---------- Performance test function ----------
def test_performance():
    """Test performance of the fixed optimized models"""
    print("=== FIXED OPTIMIZED MODEL PERFORMANCE TEST ===\n")
    
    n_layers = 50
    n_steps = 1000
    dt = 0.05
    
    print(f"Test Configuration:")
    print(f"- Layers: {n_layers}")
    print(f"- Steps: {n_steps}")
    print(f"- Time step: {dt}")
    print()
    
    # Test each model
    models = [
        ("Option A: Global Workspace", OptimizedWorkspaceSim),
        ("Option B: Reflective Hierarchy", OptimizedReflectiveSim), 
        ("Option C: Self-Referential", OptimizedSelfReferentialSim)
    ]
    
    results = []
    
    for name, model_class in models:
        print(f"Testing {name}...")
        
        # Initialize simulation
        sim = model_class(n_layers=n_layers, dt=dt)
        
        # Warm up (let it stabilize)
        for _ in range(10):
            sim.step()
        
        # Clear history for clean test
        sim.R_hist = []
        sim.step_count = 0
        
        # Performance test
        start_time = time.perf_counter()
        
        for step in range(n_steps):
            R, x = sim.step()
            
            # Show progress every 200 steps
            if step % 200 == 0 and step > 0:
                elapsed = time.perf_counter() - start_time
                estimated_fps = (step / elapsed) if elapsed > 0 else 0
                print(f"  Step {step}: {estimated_fps:.1f} FPS")
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        per_step_time = (total_time / n_steps) * 1000
        fps = n_steps / total_time if total_time > 0 else 0
        
        # Get final diagnostics
        diag = sim.get_diagnostics()
        
        results.append({
            'name': name,
            'total_time': total_time,
            'per_step_ms': per_step_time,
            'fps': fps,
            'final_R': diag['current_R'],
            'max_R': diag['max_R'],
            'steps': diag['step']
        })
        
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Per step: {per_step_time:.2f}ms")
        print(f"  FPS: {fps:.1f}")
        print(f"  Final R: {diag['current_R']:.3f}")
        print()
    
    # Summary
    print("=== PERFORMANCE SUMMARY ===")
    print(f"{'Model':<30} {'Time (s)':<10} {'ms/step':<10} {'FPS':<8} {'Final R':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['name']:<30} {result['total_time']:<10.3f} "
              f"{result['per_step_ms']:<10.2f} {result['fps']:<8.1f} {result['final_R']:<10.3f}")
    
    # Animation feasibility
    print(f"\n=== ANIMATION FEASIBILITY ===")
    for result in results:
        target_fps = 10  # Target FPS for smooth animation
        required_ms = 1000 / target_fps
        feasible = "✅ YES" if result['per_step_ms'] < required_ms else "❌ NO"
        print(f"  {result['name']}: {result['per_step_ms']:.1f}ms < {required_ms:.1f}ms → {feasible}")
    
    return results

if __name__ == "__main__":
    print("Neural Workspace Fixed Optimized Performance Test")
    print("=" * 60)
    
    # Test simulations
    results = test_performance()
    
    print(f"\n=== CONCLUSION ===")
    if results:
        avg_per_step = np.mean([r['per_step_ms'] for r in results])
        print(f"Average per-step time: {avg_per_step:.2f}ms")
        print(f"Estimated animation FPS: {1000/avg_per_step:.1f}")
        
        if avg_per_step < 50:  # Target for 20 FPS
            print("✅ Performance is sufficient for smooth animation!")
        elif avg_per_step < 100:  # Target for 10 FPS
            print("⚠️  Performance is acceptable but may be choppy")
        else:
            print("❌ Performance needs further optimization")
    
    print("\nKey fixes applied:")
    print("- Fixed naming conflict (step vs step_count)")
    print("- Maintained all optimization strategies")
    print("- Proper method calling without conflicts")