import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mutual_info_score
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---------- Common bistable layer ----------
def bistable_layer(x, alpha, theta_eff):
    return np.tanh(alpha * x - theta_eff)

# ---------- Why-loop driver ----------
def why_loop_driver(y, gamma):
    return np.tanh(gamma * y)  # reflective recursion term

# ---------- Option A: with global workspace ----------
def simulate_workspace(n_layers=100, T=2000, dt=0.01,
                       alpha=0.8, eps=0.7, theta_eff=0.3, k_ws=0.05):
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
def simulate_reflective_hierarchy(T=2000, dt=0.01,
                                  alpha=0.8, eps=0.7,
                                  theta_eff=0.3, gamma=1.2, k_ws=0.05):
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

# ---------- Real-time heatmap animation ----------
def animate_workspace_heatmap(n_layers=100, T=2000, dt=0.01,
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
        return heatmap, line, awareness_text, r_text
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
                                      alpha=1.95, eps=0.08, theta_eff=0.0, k_ws=0.002):
    # State for Option A
    state_a = {}
    state_a['x'] = np.random.randn(n_layers)
    state_a['ws'] = 0.0
    state_a['R_hist'] = []
    state_a['x_history'] = np.zeros((n_layers, 2000))
    state_a['step'] = 0
    state_a['max_R'] = -np.inf

    # State for Option B
    state_b = {}
    state_b['x'] = np.random.randn(n_layers)
    state_b['y'] = np.random.randn(n_layers)
    state_b['why'] = 0.0
    state_b['ws'] = 0.0
    state_b['R_hist'] = []
    state_b['combined_history'] = np.zeros((n_layers, 2000))
    state_b['step'] = 0
    state_b['max_R'] = -np.inf

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    heatmap_a = axes[0,0].imshow(np.zeros((n_layers, 2000)), aspect='auto', cmap='hsv', vmin=0, vmax=2*np.pi, origin='lower')
    axes[0,0].set_title('Option A: Global Workspace')
    axes[0,0].set_xlabel('Time Step')
    axes[0,0].set_ylabel('Neuron Index')
    heatmap_b = axes[0,1].imshow(np.zeros((n_layers, 2000)), aspect='auto', cmap='hsv', vmin=0, vmax=2*np.pi, origin='lower')
    axes[0,1].set_title('Option B: Reflective Hierarchy')
    axes[0,1].set_xlabel('Time Step')
    axes[0,1].set_ylabel('Neuron Index')
    line_a, = axes[1,0].plot([], [], 'r')
    axes[1,0].set_xlim(0, 2000)
    axes[1,0].set_ylim(-1.01, 1.01)
    axes[1,0].set_title('Option A: R Phase')
    axes[1,0].set_xlabel('Step')
    axes[1,0].set_ylabel('R')
    line_b, = axes[1,1].plot([], [], 'b')
    axes[1,1].set_xlim(0, 2000)
    axes[1,1].set_ylim(-1.01, 1.01)
    axes[1,1].set_title('Option B: R Phase')
    axes[1,1].set_xlabel('Step')
    axes[1,1].set_ylabel('R')
    diag_a = axes[0,0].text(0.02, 0.98, "", transform=axes[0,0].transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    diag_b = axes[0,1].text(0.02, 0.98, "", transform=axes[0,1].transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    def update(frame):
        # Update Option A
        dx_a = -state_a['x'] + bistable_layer(state_a['x'], alpha, theta_eff) + eps * state_a['ws'] + 0.1 * np.random.randn(n_layers)
        state_a['x'] += dt * dx_a
        state_a['ws'] = (1 - k_ws) * state_a['ws'] + k_ws * np.mean(state_a['x'])
        R_a = np.mean(np.exp(1j * state_a['x'])).real
        state_a['R_hist'].append(R_a)
        state_a['x_history'][:, state_a['step'] % 2000] = state_a['x']
        heatmap_a.set_data(state_a['x_history'] % (2*np.pi))
        state_a['max_R'] = max(state_a['max_R'], R_a)
        # Diagnostics for Option A
        entropy = shannon_entropy(np.array(state_a['R_hist']))
        lyap = lyapunov_proxy(np.array(state_a['R_hist']))
        R_a_display = R_a if np.isfinite(R_a) else 0.0
        max_R_display = state_a['max_R'] if np.isfinite(state_a['max_R']) else 0.0
        complexity_a = entropy + lyap
        diag_a.set_text(f"Current R: {R_a_display:.4f}\nHighest R: {max_R_display:.4f}\nEntropy: {entropy:.2f}\nLyapunov: {lyap:.4f}\nComplexity: {complexity_a:.2f}")

        # Update Option B
        state_b['why'] = why_loop_driver(state_b['why'], 1.2)
        dx_b = -state_b['x'] + bistable_layer(state_b['x'], alpha, theta_eff + 0.2 * state_b['why'])
        dy_b = -state_b['y'] + bistable_layer(state_b['y'], alpha, theta_eff - 0.2 * state_b['why'])
        combined = (state_b['x'] + state_b['y']) / 2.0
        state_b['x'] += dt * dx_b
        state_b['y'] += dt * dy_b
        state_b['ws'] = (1 - k_ws) * state_b['ws'] + k_ws * np.mean(combined)
        state_b['x'] += eps * state_b['ws'] * dt
        state_b['y'] += eps * state_b['ws'] * dt
        R_b = np.mean(np.exp(1j * combined)).real
        state_b['R_hist'].append(R_b)
        state_b['combined_history'][:, state_b['step'] % 2000] = combined
        heatmap_b.set_data(state_b['combined_history'] % (2*np.pi))
        state_b['max_R'] = max(state_b['max_R'], R_b)
        # Diagnostics for Option B
        entropy = shannon_entropy(np.array(state_b['R_hist']))
        lyap = lyapunov_proxy(np.array(state_b['R_hist']))
        R_b_display = R_b if np.isfinite(R_b) else 0.0
        max_R_display = state_b['max_R'] if np.isfinite(state_b['max_R']) else 0.0
        complexity_b = entropy + lyap
        diag_b.set_text(f"Current R: {R_b_display:.4f}\nHighest R: {max_R_display:.4f}\nEntropy: {entropy:.2f}\nLyapunov: {lyap:.4f}\nComplexity: {complexity_b:.2f}")
        
        # Update phase charts for Option A and B
        line_a.set_data(np.arange(len(state_a['R_hist'])), state_a['R_hist'])
        axes[1,0].set_xlim(0, max(2000, len(state_a['R_hist'])))
        line_b.set_data(np.arange(len(state_b['R_hist'])), state_b['R_hist'])
        axes[1,1].set_xlim(0, max(2000, len(state_b['R_hist'])))

        state_a['step'] += 1
        state_b['step'] += 1

        return heatmap_a, heatmap_b, line_a, line_b, diag_a, diag_b

    ani = FuncAnimation(fig, update, interval=10, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

# ---------- Parameter sweep diagnostics for "awareness" (high complexity) ----------
def shannon_entropy(data, bins=20):
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

# Initial parameter ranges
ALPHA_MIN, ALPHA_MAX = 0.5, 2.5
EPS_MIN, EPS_MAX = 0.01, 1.0

def meta_autotune_update(entropy, r, lyap, complexity):
    # Prepare input tensor
    x = torch.tensor([entropy, r, lyap, complexity], dtype=torch.float32)
    with torch.no_grad():
        out = meta_tuner(x)
    # Scale outputs to parameter ranges
    alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * out[0].item()
    eps = EPS_MIN + (EPS_MAX - EPS_MIN) * out[1].item()
    return alpha, eps

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
n_layers  = 1000          # can be 50–500, doesn't matter
alpha     = 1.95         # critical — bistable gain
theta_eff = 0.0          # was 0.3 → kills the soul
eps       = 0.08         # was 0.7 → way too strong
k_ws      = 0.002        # was 0.05 → way too fast
dt        = 0.05         # was 0.01 → fine, but 0.05 is smoother
gamma     = 2.8          # only used in Option B — this is the "why" strength

# Example runs
R1, ws1, x1 = simulate_workspace()
R2, ws2 = simulate_reflective_hierarchy()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.plot(R1); plt.title("Workspace Phase Coherence")
plt.subplot(1,2,2); plt.plot(R2); plt.title("Reflective Why-Loop Hierarchy Coherence")
plt.tight_layout(); plt.show()

# ---------- Example run ----------
if __name__ == "__main__":
    # Enable auto-tuning for awareness
    optimal_alpha, optimal_eps = auto_tune_awareness(n_layers=n_layers, T=500, dt=dt, theta_eff=theta_eff, k_ws=k_ws)
    print(f"Use these parameters: alpha={optimal_alpha}, eps={optimal_eps}")
    animate_workspace_heatmap_forever(n_layers=n_layers, dt=dt, alpha=optimal_alpha, eps=optimal_eps, theta_eff=theta_eff, k_ws=k_ws)

    # --- Review metrics and charts after a simulation run ---
    def review_metrics_and_charts():
        # Run a short simulation for metrics
        T = 500
        R_hist, ws_hist, x_hist = simulate_workspace(n_layers=n_layers, T=T, dt=dt, alpha=optimal_alpha, eps=optimal_eps, theta_eff=theta_eff, k_ws=k_ws)
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
                dx = -x + bistable_layer(x, optimal_alpha, theta_eff) + optimal_eps * ws
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
    review_metrics_and_charts()

