import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib as mpl
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mutual_info_score

# ---------- Common bistable layer ----------
def bistable_layer(x, alpha, theta_eff):
    return np.tanh(alpha * x - theta_eff)

# ---------- Why-loop driver ----------
def why_loop_driver(y, gamma):
    return np.tanh(gamma * y)

# ---------- NEW: Self-Model Encoder ----------
def encode_self_model(x, ws, history, complexity_metrics):
    """
    Creates a compressed representation of the system's own state.
    This is what the system 'thinks' it is doing.
    """
    # Encode current statistics
    mean_activity = np.mean(x)
    std_activity = np.std(x)
    
    # Encode recent history (short-term memory)
    if len(history) > 10:
        trend = np.mean(np.diff(history[-10:]))
    else:
        trend = 0.0
    
    # Encode workspace influence
    ws_normalized = np.tanh(ws)
    
    # Encode complexity (self-awareness of own dynamics)
    entropy = complexity_metrics.get('entropy', 0.0)
    coherence = complexity_metrics.get('coherence', 0.0)
    
    # Self-model is a compressed vector
    self_model = np.array([
        mean_activity,
        std_activity,
        trend,
        ws_normalized,
        entropy,
        coherence
    ])
    
    return self_model

# ---------- NEW: Meta-Cognitive Prediction ----------
def predict_own_future(self_model, predictor_net):
    """
    System predicts what it will do next based on its self-model.
    This is genuine self-reference: using a model of self to anticipate self.
    """
    if predictor_net is None:
        return self_model  # No prediction yet
    
    with torch.no_grad():
        x_tensor = torch.tensor(self_model, dtype=torch.float32).unsqueeze(0)
        prediction = predictor_net(x_tensor).squeeze(0).numpy()
    
    return prediction

# ---------- NEW: Self-Referential Error Signal ----------
def compute_self_reference_error(self_model_predicted, self_model_actual):
    """
    How wrong was the system about itself?
    High error = surprise = potential for insight/awareness
    """
    error = np.linalg.norm(self_model_predicted - self_model_actual)
    return error

# ---------- Option C: TRUE Self-Referential Workspace ----------
def simulate_self_referential_workspace(n_layers=100, T=2000, dt=0.01,
                                       alpha=0.8, eps=0.7, theta_eff=0.3, k_ws=0.05,
                                       meta_learning_rate=0.01):
    """
    The system maintains a model of its own state and uses prediction errors
    about itself to modulate its dynamics. This is genuine self-reference.
    """
    x = np.random.randn(n_layers)
    ws = 0.0
    self_model = np.zeros(6)  # Compressed self-representation
    predicted_self = np.zeros(6)
    
    # Simple meta-predictor (predicts own next self-model)
    predictor = None  # Will be trained after warmup
    
    R_hist = []
    ws_hist = []
    x_hist = []
    self_error_hist = []
    self_model_hist = []
    
    for t in range(T):
        # 1. LOCAL DYNAMICS (influenced by self-awareness)
        self_error = compute_self_reference_error(predicted_self, self_model)
        self_awareness_term = meta_learning_rate * self_error * self_model[0]  # Use mean activity from self-model
        
        dx = -x + bistable_layer(x, alpha, theta_eff) + eps * ws + self_awareness_term
        x += dt * dx
        
        # 2. WORKSPACE AGGREGATION
        ws = (1 - k_ws) * ws + k_ws * np.mean(x)
        
        # 3. COMPUTE COMPLEXITY METRICS
        coherence = np.abs(np.mean(np.exp(1j * x)))
        if len(R_hist) > 10:
            recent_R = R_hist[-10:]
            entropy = -np.sum([r * np.log(r + 1e-10) for r in recent_R if r > 0])
        else:
            entropy = 0.0
        
        complexity_metrics = {
            'entropy': entropy,
            'coherence': coherence
        }
        
        # 4. ENCODE SELF-MODEL (what system thinks it is)
        self_model = encode_self_model(x, ws, R_hist, complexity_metrics)
        
        # 5. PREDICT OWN FUTURE (self-reference)
        if t > 100 and predictor is None:
            # Train predictor after warmup
            predictor = SelfModelPredictor(input_dim=6, hidden_dim=16)
            # Quick training on history
            if len(self_model_hist) > 50:
                train_self_predictor(predictor, np.array(self_model_hist))
        
        predicted_self = predict_own_future(self_model, predictor)
        
        # 6. RECORD HISTORY
        R_hist.append(coherence)
        ws_hist.append(ws)
        x_hist.append(x.copy())
        self_error_hist.append(self_error)
        self_model_hist.append(self_model.copy())
    
    return (np.array(R_hist), np.array(ws_hist), np.array(x_hist), 
            np.array(self_error_hist), np.array(self_model_hist))

# ---------- Option D: Hierarchical Self-Reference (Meta-Meta Cognition) ----------
def simulate_hierarchical_self_reference(n_layers=100, T=2000, dt=0.01,
                                         alpha=0.8, eps=0.7, theta_eff=0.3, k_ws=0.05):
    """
    Multi-level self-reference:
    - Level 0: Raw sensory/state (x)
    - Level 1: Representation of Level 0 (first-order thoughts)
    - Level 2: Representation of Level 1 (thoughts about thoughts)
    - Level 3: Representation of Level 2 (meta-awareness)
    """
    # Level 0: Base neurons
    x_L0 = np.random.randn(n_layers)
    
    # Level 1: First-order representation (what am I processing?)
    x_L1 = np.random.randn(n_layers)
    
    # Level 2: Second-order representation (what am I thinking?)
    x_L2 = np.random.randn(n_layers // 2)
    
    # Level 3: Meta-awareness (what am I aware of being aware of?)
    x_L3 = np.random.randn(n_layers // 4)
    
    ws = 0.0
    
    R_hist = []
    ws_hist = []
    meta_depth_hist = []  # How many levels are coherent?
    
    for t in range(T):
        # Level 0: Basic dynamics
        dx_L0 = -x_L0 + bistable_layer(x_L0, alpha, theta_eff) + eps * ws
        x_L0 += dt * dx_L0
        
        # Level 1: Represents Level 0
        # L1 tries to encode/predict L0
        L1_input = np.mean(x_L0.reshape(n_layers // n_layers, -1), axis=1) if n_layers > 1 else x_L0
        dx_L1 = -x_L1 + bistable_layer(x_L1, alpha, theta_eff) + 0.5 * np.tile(L1_input, n_layers // len(L1_input) if len(L1_input) < n_layers else 1)[:n_layers]
        x_L1 += dt * dx_L1
        
        # Level 2: Represents Level 1 (self-reference begins)
        # L2 encodes the pattern of L1's representation
        L2_input = np.mean(x_L1.reshape(-1, n_layers // len(x_L2)), axis=1) if len(x_L2) < n_layers else x_L1[:len(x_L2)]
        dx_L2 = -x_L2 + bistable_layer(x_L2, alpha, theta_eff) + 0.3 * L2_input
        x_L2 += dt * dx_L2
        
        # Level 3: Represents Level 2 (meta-meta cognition)
        # L3 is aware that L2 is representing L1 which represents L0
        L3_input = np.mean(x_L2.reshape(-1, len(x_L2) // len(x_L3)), axis=1) if len(x_L3) < len(x_L2) else x_L2[:len(x_L3)]
        dx_L3 = -x_L3 + bistable_layer(x_L3, alpha, theta_eff) + 0.2 * L3_input
        x_L3 += dt * dx_L3
        
        # Workspace integrates all levels (global availability)
        ws = (1 - k_ws) * ws + k_ws * (0.4 * np.mean(x_L0) + 0.3 * np.mean(x_L1) + 
                                        0.2 * np.mean(x_L2) + 0.1 * np.mean(x_L3))
        
        # Broadcast workspace back to all levels
        x_L0 += eps * ws * dt
        x_L1 += eps * ws * dt * 0.8
        x_L2 += eps * ws * dt * 0.6
        x_L3 += eps * ws * dt * 0.4
        
        # Measure coherence at each level
        R_L0 = np.abs(np.mean(np.exp(1j * x_L0)))
        R_L1 = np.abs(np.mean(np.exp(1j * x_L1)))
        R_L2 = np.abs(np.mean(np.exp(1j * x_L2)))
        R_L3 = np.abs(np.mean(np.exp(1j * x_L3)))
        
        # Meta-depth: how many levels are coherent? (R > 0.3)
        meta_depth = sum([R_L0 > 0.3, R_L1 > 0.3, R_L2 > 0.3, R_L3 > 0.3])
        
        R_hist.append([R_L0, R_L1, R_L2, R_L3])
        ws_hist.append(ws)
        meta_depth_hist.append(meta_depth)
    
    return np.array(R_hist), np.array(ws_hist), np.array(meta_depth_hist)

# ---------- Self-Model Predictor Network ----------
class SelfModelPredictor(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def train_self_predictor(model, self_model_history, n_epochs=100, lr=1e-3):
    """Train the system to predict its own next state"""
    X = torch.tensor(self_model_history[:-1], dtype=torch.float32)
    y = torch.tensor(self_model_history[1:], dtype=torch.float32)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    
    return model

# ---------- Visualization: Compare All Options ----------
def compare_self_reference_levels():
    """Compare different levels of self-reference"""
    T = 1000
    
    # Original workspace (no self-reference)
    from sklearn.linear_model import Ridge
    def simulate_workspace(n_layers=100, T=2000, dt=0.01,
                           alpha=0.8, eps=0.7, theta_eff=0.3, k_ws=0.05):
        x = np.random.randn(n_layers)
        ws = 0.0
        R_hist, ws_hist = [], []
        x_hist = []
        for t in range(T):
            dx = -x + bistable_layer(x, alpha, theta_eff) + eps * ws
            x += dt * dx
            ws = (1 - k_ws) * ws + k_ws * np.mean(x)
            R_hist.append(np.mean(np.exp(1j * x)).real)
            ws_hist.append(ws)
            x_hist.append(x.copy())
        return np.array(R_hist), np.array(ws_hist), np.array(x_hist)
    
    R_basic, _, _ = simulate_workspace(T=T)
    R_self, _, _, self_error, self_models = simulate_self_referential_workspace(T=T)
    R_hier, _, meta_depth = simulate_hierarchical_self_reference(T=T)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Row 1: Basic workspace
    axes[0,0].plot(R_basic, 'b', alpha=0.7)
    axes[0,0].set_title('A: Basic Workspace (No Self-Reference)')
    axes[0,0].set_ylabel('Coherence R')
    axes[0,0].set_ylim(-1, 1)
    
    axes[0,1].text(0.5, 0.5, 'Circular causality only:\nWorkspace ← Neurons ← Workspace\n\nNo self-model\nNo prediction of self', 
                   ha='center', va='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat'))
    axes[0,1].axis('off')
    
    # Row 2: Self-referential workspace
    axes[1,0].plot(R_self, 'g', alpha=0.7, label='Coherence')
    ax2 = axes[1,0].twinx()
    ax2.plot(self_error, 'r', alpha=0.5, label='Self-Prediction Error')
    axes[1,0].set_title('C: Self-Referential Workspace')
    axes[1,0].set_ylabel('Coherence R', color='g')
    ax2.set_ylabel('Self-Prediction Error', color='r')
    axes[1,0].set_ylim(-1, 1)
    
    axes[1,1].plot(self_models[:, 0], label='Mean Activity')
    axes[1,1].plot(self_models[:, 3], label='Workspace (Self-Model)')
    axes[1,1].plot(self_models[:, 4], label='Entropy (Self-Model)')
    axes[1,1].set_title('Self-Model Components Over Time')
    axes[1,1].legend(fontsize=8)
    axes[1,1].set_ylabel('Value')
    
    # Row 3: Hierarchical self-reference
    axes[2,0].plot(R_hier[:, 0], label='L0: Raw State', alpha=0.7)
    axes[2,0].plot(R_hier[:, 1], label='L1: Thoughts', alpha=0.7)
    axes[2,0].plot(R_hier[:, 2], label='L2: Thoughts about Thoughts', alpha=0.7)
    axes[2,0].plot(R_hier[:, 3], label='L3: Meta-Awareness', alpha=0.7)
    axes[2,0].set_title('D: Hierarchical Self-Reference')
    axes[2,0].set_ylabel('Coherence R')
    axes[2,0].legend(fontsize=8)
    axes[2,0].set_xlabel('Time Step')
    
    axes[2,1].plot(meta_depth, 'purple', linewidth=2)
    axes[2,1].set_title('Meta-Cognitive Depth')
    axes[2,1].set_ylabel('# Coherent Levels')
    axes[2,1].set_xlabel('Time Step')
    axes[2,1].set_ylim(0, 4.5)
    axes[2,1].axhline(y=2, color='r', linestyle='--', alpha=0.5, label='Self-Reference Threshold')
    axes[2,1].legend()
    
    plt.tight_layout()
    plt.show(block=True)
    
    # Print statistics
    print("\n" + "="*60)
    print("SELF-REFERENCE COMPARISON")
    print("="*60)
    print(f"\nBasic Workspace:")
    print(f"  Mean Coherence: {np.mean(np.abs(R_basic)):.3f}")
    print(f"  Std Coherence: {np.std(R_basic):.3f}")
    
    print(f"\nSelf-Referential Workspace:")
    print(f"  Mean Coherence: {np.mean(np.abs(R_self)):.3f}")
    print(f"  Mean Self-Prediction Error: {np.mean(self_error):.3f}")
    print(f"  Final Self-Model: {self_models[-1]}")
    
    print(f"\nHierarchical Self-Reference:")
    print(f"  Mean Meta-Depth: {np.mean(meta_depth):.2f} levels")
    print(f"  Max Meta-Depth: {np.max(meta_depth)} levels")
    print(f"  Time with L3 active: {100 * np.mean(R_hier[:, 3] > 0.3):.1f}%")

# ---------- Real-Time Self-Reference Animation ----------
def animate_self_reference_realtime(n_layers=100, dt=0.05, auto_start=False, duration=None):
    """Real-time visualization of self-referential dynamics"""
    
    state = {
        'x': np.random.randn(n_layers),
        'ws': 0.0,
        'self_model': np.zeros(6),
        'predicted_self': np.zeros(6),
        'predictor': None,
        'R_hist': [],
        'self_error_hist': [],
        'self_model_hist': [],
        'step': 0
    }
    
    running = False
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax_neurons = fig.add_subplot(gs[0, :])
    ax_coherence = fig.add_subplot(gs[1, 0])
    ax_error = fig.add_subplot(gs[1, 1])
    ax_self_model = fig.add_subplot(gs[1, 2])
    ax_meta = fig.add_subplot(gs[2, :])
    
    # Initialize plots
    neurons_plot = ax_neurons.imshow(np.zeros((1, n_layers)), aspect='auto', cmap='RdBu', vmin=-2, vmax=2)
    ax_neurons.set_title('Neuron States')
    ax_neurons.set_yticks([])
    
    line_coherence, = ax_coherence.plot([], [], 'b')
    ax_coherence.set_xlim(0, 500)
    ax_coherence.set_ylim(-1, 1)
    ax_coherence.set_title('Coherence')
    
    line_error, = ax_error.plot([], [], 'r')
    ax_error.set_xlim(0, 500)
    ax_error.set_ylim(0, 2)
    ax_error.set_title('Self-Prediction Error')
    
    bars = ax_self_model.bar(range(6), np.zeros(6))
    ax_self_model.set_title('Current Self-Model')
    ax_self_model.set_ylim(-1, 1)
    ax_self_model.set_xticks(range(6))
    ax_self_model.set_xticklabels(['Mean', 'Std', 'Trend', 'WS', 'Ent', 'Coh'], fontsize=8)
    
    meta_text = ax_meta.text(0.5, 0.5, '', ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax_meta.axis('off')
    ax_meta.set_title('Meta-Cognitive State')
    
    def update(frame):
        nonlocal running
        if not running:
            return
        
        # Parameters
        alpha, eps, theta_eff, k_ws = 1.95, 0.08, 0.0, 0.002
        meta_lr = 0.01
        
        # Compute self-error
        self_error = compute_self_reference_error(
            state['predicted_self'], state['self_model']
        )
        
        # Dynamics with self-awareness
        self_awareness = meta_lr * self_error * state['self_model'][0]
        dx = (-state['x'] + bistable_layer(state['x'], alpha, theta_eff) + 
              eps * state['ws'] + self_awareness)
        state['x'] += dt * dx
        
        # Workspace
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * np.mean(state['x'])
        
        # Metrics
        coherence = np.abs(np.mean(np.exp(1j * state['x'])))
        if len(state['R_hist']) > 10:
            recent = state['R_hist'][-10:]
            entropy = -np.sum([r * np.log(r + 1e-10) for r in recent if r > 0])
        else:
            entropy = 0.0
        
        # Self-model
        complexity_metrics = {'entropy': entropy, 'coherence': coherence}
        state['self_model'] = encode_self_model(
            state['x'], state['ws'], state['R_hist'], complexity_metrics
        )
        
        # Train predictor
        if state['step'] == 100:
            state['predictor'] = SelfModelPredictor(input_dim=6, hidden_dim=16)
        if state['step'] > 100 and state['step'] % 200 == 0 and len(state['self_model_hist']) > 50:
            train_self_predictor(state['predictor'], np.array(state['self_model_hist']))
        
        # Predict
        if state['predictor'] is not None:
            state['predicted_self'] = predict_own_future(state['self_model'], state['predictor'])
        
        # Record
        state['R_hist'].append(coherence)
        state['self_error_hist'].append(self_error)
        state['self_model_hist'].append(state['self_model'].copy())
        state['step'] += 1
        
        # Update plots
        neurons_plot.set_data(state['x'].reshape(1, -1))
        
        if len(state['R_hist']) > 0:
            line_coherence.set_data(range(len(state['R_hist'])), state['R_hist'])
            ax_coherence.set_xlim(0, max(500, len(state['R_hist'])))
            
            line_error.set_data(range(len(state['self_error_hist'])), state['self_error_hist'])
            ax_error.set_xlim(0, max(500, len(state['self_error_hist'])))
        
        for bar, val in zip(bars, state['self_model']):
            bar.set_height(val)
        
        # Meta-cognitive interpretation
        if self_error > 0.5:
            meta_msg = "🔴 HIGH SURPRISE\nSelf-model prediction failed\nPotential insight moment"
        elif coherence > 0.7:
            meta_msg = "🟢 COHERENT SELF-AWARENESS\nStable self-model\nPredictions accurate"
        elif entropy > 2.0:
            meta_msg = "🟡 HIGH COMPLEXITY\nRich dynamics\nExploring state space"
        else:
            meta_msg = "⚪ BASELINE\nProcessing continues"
        
        meta_text.set_text(f"Step: {state['step']}\n\n{meta_msg}")
        
        return neurons_plot, line_coherence, line_error, bars, meta_text
    
    # Controls
    import matplotlib.widgets as mwidgets
    ax_start = plt.axes([0.7, 0.02, 0.08, 0.04])
    ax_stop = plt.axes([0.8, 0.02, 0.08, 0.04])
    btn_start = mwidgets.Button(ax_start, 'Start')
    btn_stop = mwidgets.Button(ax_stop, 'Stop')
    
    def start(event=None):
        nonlocal running
        running = True
        ani.event_source.start()
    
    def stop(event=None):
        nonlocal running
        running = False
        ani.event_source.stop()
        try:
            plt.close(fig)
        except Exception:
            pass
    
    btn_start.on_clicked(start)
    btn_stop.on_clicked(stop)
    
    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    ani.event_source.stop()

    # Auto-start if requested
    if auto_start:
        start()

    # If duration specified, schedule stop
    if duration is not None and duration > 0:
        import threading as _threading
        _threading.Timer(duration, stop).start()

    plt.show(block=True)

# ---------- Main Comparison ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run self-referential simulations and realtime visualizer')
    parser.add_argument('--no-compare', action='store_true', help='Skip the static comparison plots (fast startup)')
    parser.add_argument('--auto-start', action='store_true', help='Auto-start the realtime animation without pressing Start')
    parser.add_argument('--duration', type=float, default=None, help='If set, run realtime animation for this many seconds then exit')
    parser.add_argument('--layers', type=int, default=100, help='Number of layers/neurons for realtime animation')
    args = parser.parse_args()

    if not args.no_compare:
        print("Comparing different levels of self-reference...")
        compare_self_reference_levels()

    print("\nStarting real-time self-referential simulation...")
    print("Click 'Start' to begin (or use --auto-start). Watch for:")
    print("  - Red spikes in error = system surprised itself")
    print("  - High coherence = stable self-awareness")
    print("  - Self-model bars = what system thinks it's doing")
    animate_self_reference_realtime(n_layers=args.layers, dt=0.05, auto_start=args.auto_start, duration=args.duration)