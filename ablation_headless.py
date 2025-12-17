import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import deque
from lab import bistable_layer

# --- Parameters (frozen for ablation) ---
n_layers = 100
dt = 0.05
alpha = 1.95
eps = 0.08
theta_eff = 0.0
k_ws = 0.002
steps = 20000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model runner utility ---
def run_model_A():
    state = {}
    state['x'] = torch.randn(n_layers, device=device)
    state['ws'] = torch.tensor(0.0, device=device)
    state['R_hist'] = []
def run_model_A_meta(lambda_decouple=1.0):
    state = {}
    state['x'] = torch.randn(n_layers, device=device)
    state['ws'] = torch.tensor(0.0, device=device)
    state['R_hist'] = []
    from lab import create_meta_tuner_and_optimizer, meta_autotune_update, train_meta_tuner_batch, ALPHA_MIN, ALPHA_MAX, EPS_MIN, EPS_MAX
    meta_tuner, tuner_optimizer = create_meta_tuner_and_optimizer()
    experience_buffer = deque(maxlen=1000)
    for step in range(steps):
        x_np = state['x'].detach().cpu().numpy()
        hist = np.histogram(x_np, bins=32, density=True)[0]
        hist = hist / (np.sum(hist) + 1e-12)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))
        r = torch.abs(torch.mean(torch.exp(1j * state['x']))).item()
        lyap = 0.0
        complexity = float(np.std(x_np))
        alpha_dyn, eps_dyn = meta_autotune_update(meta_tuner, entropy, r, lyap, complexity)
        alpha_dyn = lambda_decouple * alpha_dyn + (1 - lambda_decouple) * alpha
        eps_dyn = lambda_decouple * eps_dyn + (1 - lambda_decouple) * eps
        dx = -state['x'] + bistable_layer(state['x'], alpha_dyn, theta_eff) + eps_dyn * state['ws'] + 0.03 * torch.randn(n_layers, device=device)
        state['x'] += dt * dx
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(state['x'])
        R = torch.abs(torch.mean(torch.exp(1j * state['x']))).item()
        state['R_hist'].append(R)
        alpha_norm = (alpha_dyn - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)
        eps_norm = (eps_dyn - EPS_MIN) / (EPS_MAX - EPS_MIN)
        reward = entropy + r
        experience_buffer.append(([entropy, r, lyap, complexity], [alpha_norm, eps_norm], reward))
        if len(experience_buffer) > 16 and step % 20 == 0:
            train_meta_tuner_batch(meta_tuner, tuner_optimizer, experience_buffer, batch_size=32, n_epochs=5, lr=1e-3)
    return state['R_hist']

def run_model_B():
    state = {}
    state['x'] = torch.randn(n_layers, device=device)
    state['y'] = torch.randn(n_layers, device=device)
    state['why'] = torch.tensor(0.0, device=device)
    state['ws'] = torch.tensor(0.0, device=device)
    state['R_hist'] = []
def run_model_B_meta(lambda_decouple=1.0):
    state = {}
    state['x'] = torch.randn(n_layers, device=device)
    state['y'] = torch.randn(n_layers, device=device)
    state['why'] = torch.tensor(0.0, device=device)
    state['ws'] = torch.tensor(0.0, device=device)
    state['R_hist'] = []
    for step in range(steps):
        state['why'] = torch.tanh(1.2 * state['why'])
        meta_alpha = alpha + 0.2 * np.sin(2 * np.pi * step / 5000)
        meta_eps = eps + 0.02 * np.cos(2 * np.pi * step / 7000)
        alpha_dyn = lambda_decouple * meta_alpha + (1 - lambda_decouple) * alpha
        eps_dyn = lambda_decouple * meta_eps + (1 - lambda_decouple) * eps
        dx = -state['x'] + bistable_layer(state['x'], alpha_dyn, theta_eff + 0.2 * state['why'])
        dy = -state['y'] + bistable_layer(state['y'], alpha_dyn, theta_eff - 0.2 * state['why'])
        dx += 0.03 * torch.randn(n_layers, device=device)
        dy += 0.03 * torch.randn(n_layers, device=device)
        state['x'] += dt * dx
        state['y'] += dt * dy
        combined = (state['x'] + state['y']) / 2.0
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(combined)
        state['x'] += eps_dyn * state['ws'] * dt
        state['y'] += eps_dyn * state['ws'] * dt
        R = torch.abs(torch.mean(torch.exp(1j * combined))).item()
        state['R_hist'].append(R)
    return state['R_hist']


def run_model_C():
    state = {}
    state['x'] = torch.randn(n_layers, device=device)
    state['ws'] = torch.tensor(0.0, device=device)
    state['R_hist'] = []
    state['self_error_hist'] = []
    state['goal_pattern'] = np.sin(np.linspace(0, 2 * np.pi, n_layers))
    self_model = np.zeros(6)
    predicted_self = np.zeros(6)
    for step in range(steps):
        self_error = np.linalg.norm(predicted_self - self_model)
        self_awareness = 0.01 * self_error * (self_model[0] if self_model.size > 0 else 0.0)
        dx = -state['x'] + bistable_layer(state['x'], alpha, theta_eff) + eps * state['ws'] + self_awareness + 0.03 * torch.randn(n_layers, device=device)
        state['x'] += dt * dx
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(state['x'])

    # --- Metaoptimizer model runners (moved up for early use) ---
        R = torch.abs(torch.mean(torch.exp(1j * state['x']))).item()
        state['R_hist'].append(R)
        # Update self_model (simple moving average)
        x_np = state['x'].cpu().numpy()
        M = len(self_model)
        new_self_model = np.zeros_like(self_model)
        for i in range(M):
            start = (i * len(x_np)) // M
            end = ((i + 1) * len(x_np)) // M
            avg = np.mean(x_np[start:end])
            new_self_model[i] = self_model[i] * 0.75 + avg * 0.25
        self_model = new_self_model
        state['self_error_hist'].append(self_error)
    return state['R_hist']

def run_model_D():
    state = {}
    state['x'] = torch.randn(n_layers, device=device)
    state['y'] = torch.randn(n_layers, device=device)
    state['why'] = torch.tensor(0.0, device=device)
    state['ws'] = torch.tensor(0.0, device=device)
    state['R_hist'] = []
    state['self_error_hist'] = []
    state['goal_pattern'] = np.sin(np.linspace(0, 2 * np.pi, n_layers))
    self_model = np.zeros(8)
    predicted_self = np.zeros(8)
    for step in range(steps):
        state['why'] = torch.tanh(1.2 * state['why'])
        self_error = np.linalg.norm(predicted_self - self_model)
        self_awareness = 0.01 * self_error * (self_model[0] if self_model.size > 0 else 0.0)
        dx = -state['x'] + bistable_layer(state['x'], alpha, theta_eff + 0.2 * state['why']) + self_awareness + 0.03 * torch.randn(n_layers, device=device)
        dy = -state['y'] + bistable_layer(state['y'], alpha, theta_eff - 0.2 * state['why']) + self_awareness + 0.03 * torch.randn(n_layers, device=device)
        state['x'] += dt * dx
        state['y'] += dt * dy
        combined = (state['x'] + state['y']) / 2.0
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(combined)
        state['x'] += eps * state['ws'] * dt
        state['y'] += eps * state['ws'] * dt
        R = torch.abs(torch.mean(torch.exp(1j * combined))).item()
        state['R_hist'].append(R)
        # Update self_model (simple moving average)
        x_np = state['x'].cpu().numpy()
        y_np = state['y'].cpu().numpy()
        M = len(self_model) // 2
        new_self_model = np.zeros_like(self_model)
        for i in range(M):
            start = (i * len(x_np)) // M
            end = ((i + 1) * len(x_np)) // M
            avg_x = np.mean(x_np[start:end])
            avg_y = np.mean(y_np[start:end])
            new_self_model[i] = self_model[i] * 0.75 + avg_x * 0.25
            new_self_model[i+M] = self_model[i+M] * 0.75 + avg_y * 0.25
        self_model = new_self_model
        state['self_error_hist'].append(self_error)
    return state['R_hist']

def run_model_E():
    # Model E: D + awareness modulation
    state = {}
    state['x'] = torch.randn(n_layers, device=device)
    state['y'] = torch.randn(n_layers, device=device)
    state['why'] = torch.tensor(0.0, device=device)
    state['ws'] = torch.tensor(0.0, device=device)
    state['R_hist'] = []
    state['self_error_hist'] = []
    state['goal_pattern'] = np.sin(np.linspace(0, 2 * np.pi, n_layers))
    self_model = np.zeros(8)
    predicted_self = np.zeros(8)
    for step in range(steps):
        state['why'] = torch.tanh(1.2 * state['why'])
        self_error = np.linalg.norm(predicted_self - self_model)
        self_awareness = 0.01 * self_error * (self_model[0] if self_model.size > 0 else 0.0)
        dx = -state['x'] + bistable_layer(state['x'], alpha, theta_eff + 0.2 * state['why']) + self_awareness + 0.03 * torch.randn(n_layers, device=device)
        dy = -state['y'] + bistable_layer(state['y'], alpha, theta_eff - 0.2 * state['why']) + self_awareness + 0.03 * torch.randn(n_layers, device=device)
        state['x'] += dt * dx
        state['y'] += dt * dy
        combined = (state['x'] + state['y']) / 2.0
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(combined)
        state['x'] += eps * state['ws'] * dt
        state['y'] += eps * state['ws'] * dt
        # Awareness perturbation
        awareness = 0.0
        if len(state['self_error_hist']) >= 20:
            recent_self_error = np.array(state['self_error_hist'][-20:])
            std_self_error = np.std(recent_self_error)
            awareness = max(0.0, 1.0 - std_self_error / 0.1)
        pert_strength = 0.05 * awareness
        awareness_pert = pert_strength * np.sin(np.arange(n_layers) * 0.2)
        state['x'] += torch.tensor(awareness_pert, dtype=state['x'].dtype, device=device)
        state['y'] += torch.tensor(awareness_pert, dtype=state['y'].dtype, device=device)
        R = torch.abs(torch.mean(torch.exp(1j * combined))).item()
        state['R_hist'].append(R)
        # Update self_model (simple moving average)
        x_np = state['x'].cpu().numpy()
        y_np = state['y'].cpu().numpy()
        M = len(self_model) // 2
        new_self_model = np.zeros_like(self_model)
        for i in range(M):
            start = (i * len(x_np)) // M
            end = ((i + 1) * len(x_np)) // M
            avg_x = np.mean(x_np[start:end])
            avg_y = np.mean(y_np[start:end])
            new_self_model[i] = self_model[i] * 0.75 + avg_x * 0.25
            new_self_model[i+M] = self_model[i+M] * 0.75 + avg_y * 0.25
        self_model = new_self_model
        state['self_error_hist'].append(self_error)
    return state['R_hist']

# --- Run ablation for all models ---

# --- 1. Standalone models report ---
standalone_results = {
    'A': run_model_A(),
    'B': run_model_B(),
    'C': run_model_C(),
    'D': run_model_D(),
    'E': run_model_E(),
}
# Debug print for model runner return values
for label, R_hist in standalone_results.items():
    print(f"Standalone {label}: type={type(R_hist)}, len={len(R_hist) if R_hist is not None else 'None'}")
plt.figure(figsize=(12, 6))
for label, R_hist in standalone_results.items():
    plt.plot(R_hist, label=f'Model {label}')
plt.xlabel('Step')
plt.ylabel('Global Coherence R')
plt.title('Standalone Models: Global Coherence R over 20K Steps')
plt.legend()
plt.tight_layout()
plt.savefig('ablation_report_standalone.png')
print('Saved ablation_report_standalone.png')

# --- 2. Progressive ablation (frozen) ---
progressive_labels = ['A', 'A+B', 'A+B+C', 'A+B+C+D', 'A+B+C+D+E']
progressive_results = []
R_A = standalone_results['A']
R_B = standalone_results['B']
R_C = standalone_results['C']
R_D = standalone_results['D']
R_E = standalone_results['E']
progressive_results.append(R_A)
progressive_results.append(np.mean([R_A, R_B], axis=0))
progressive_results.append(np.mean([R_A, R_B, R_C], axis=0))
progressive_results.append(np.mean([R_A, R_B, R_C, R_D], axis=0))
progressive_results.append(np.mean([R_A, R_B, R_C, R_D, R_E], axis=0))
plt.figure(figsize=(12, 6))
for label, R_hist in zip(progressive_labels, progressive_results):
    plt.plot(R_hist, label=label)
plt.xlabel('Step')
plt.ylabel('Average Global Coherence R')
plt.title('Progressive Ablation (Frozen): Average R over 20K Steps')
plt.legend()
plt.tight_layout()
plt.savefig('ablation_report_progressive_frozen.png')
print('Saved ablation_report_progressive_frozen.png')

# --- 3. Progressive ablation (optimizer on) ---
R_A_meta = run_model_A_meta()
R_B_meta = run_model_B_meta()
R_C_meta = run_model_C_meta()
R_D_meta = run_model_D_meta()
R_E_meta = run_model_E_meta()
# Debug print for meta model runner return values
for label, R_hist in zip(['A','B','C','D','E'], [R_A_meta, R_B_meta, R_C_meta, R_D_meta, R_E_meta]):
    print(f"Meta {label}: type={type(R_hist)}, len={len(R_hist) if R_hist is not None else 'None'}")
progressive_results_meta = []
progressive_results_meta.append(R_A_meta)
progressive_results_meta.append(np.mean([R_A_meta, R_B_meta], axis=0))
progressive_results_meta.append(np.mean([R_A_meta, R_B_meta, R_C_meta], axis=0))
progressive_results_meta.append(np.mean([R_A_meta, R_B_meta, R_C_meta, R_D_meta], axis=0))
progressive_results_meta.append(np.mean([R_A_meta, R_B_meta, R_C_meta, R_D_meta, R_E_meta], axis=0))
plt.figure(figsize=(12, 6))
for label, R_hist in zip(progressive_labels, progressive_results_meta):
    plt.plot(R_hist, label=label)
plt.xlabel('Step')
plt.ylabel('Average Global Coherence R')
plt.title('Progressive Ablation (Optimizer On): Average R over 20K Steps')
plt.legend()
plt.tight_layout()
plt.savefig('ablation_report_progressive_meta.png')
print('Saved ablation_report_progressive_meta.png')

# --- 4. Progressive ablation (consensus/decoupling) ---
def run_progressive_consensus():
    # Run consensus for each progressive set
    results = []
    for n in range(1, 6):
        # Only use first n models in consensus
        consensus_results = run_all_models_meta_with_consensus()
        keys = list(consensus_results.keys())[:n]
        avg_R = np.mean([consensus_results[k] for k in keys], axis=0)
        results.append(avg_R)
    return results
progressive_results_consensus = run_progressive_consensus()
plt.figure(figsize=(12, 6))
for label, R_hist in zip(progressive_labels, progressive_results_consensus):
    plt.plot(R_hist, label=label)
plt.xlabel('Step')
plt.ylabel('Average Global Coherence R')
plt.title('Progressive Ablation (Consensus/Decoupling): Average R over 20K Steps')
plt.legend()
plt.tight_layout()
plt.savefig('ablation_report_progressive_consensus.png')
print('Saved ablation_report_progressive_consensus.png')

# --- Plot and save report ---
plt.figure(figsize=(12, 6))
for label, R_hist in results_meta.items():
    plt.plot(R_hist, label=f'Model {label} (meta, consensus)')
plt.xlabel('Step')
plt.ylabel('Global Coherence R')
plt.title('Ablation: Global Coherence R over 20K Steps (MetaOptimizer Consensus)')
plt.legend()
plt.tight_layout()
plt.savefig('ablation_report_meta.png')
print('Saved ablation_report_meta.png')


# --- Metaoptimizer-enabled runners ---
def run_all_models_meta_with_consensus():
    # Each model proposes on/off, consensus determines lambda_decouple for all
    state_A = {'x': torch.randn(n_layers, device=device), 'ws': torch.tensor(0.0, device=device), 'R_hist': []}
    state_B = {'x': torch.randn(n_layers, device=device), 'y': torch.randn(n_layers, device=device), 'why': torch.tensor(0.0, device=device), 'ws': torch.tensor(0.0, device=device), 'R_hist': []}
    state_C = {'x': torch.randn(n_layers, device=device), 'ws': torch.tensor(0.0, device=device), 'R_hist': [], 'self_error_hist': [], 'goal_pattern': np.sin(np.linspace(0, 2 * np.pi, n_layers))}
    state_D = {'x': torch.randn(n_layers, device=device), 'y': torch.randn(n_layers, device=device), 'why': torch.tensor(0.0, device=device), 'ws': torch.tensor(0.0, device=device), 'R_hist': [], 'self_error_hist': [], 'goal_pattern': np.sin(np.linspace(0, 2 * np.pi, n_layers))}
    state_E = {'x': torch.randn(n_layers, device=device), 'y': torch.randn(n_layers, device=device), 'why': torch.tensor(0.0, device=device), 'ws': torch.tensor(0.0, device=device), 'R_hist': [], 'self_error_hist': [], 'goal_pattern': np.sin(np.linspace(0, 2 * np.pi, n_layers))}
    self_model_C = np.zeros(6)
    predicted_self_C = np.zeros(6)
    self_model_D = np.zeros(8)
    predicted_self_D = np.zeros(8)
    self_model_E = np.zeros(8)
    predicted_self_E = np.zeros(8)
    steps_range = range(steps)
    for step in steps_range:
        # Each model proposes on/off (simple: on if R < 0.7, else off)
        # Model A
        meta_alpha_A = alpha + 0.2 * np.sin(2 * np.pi * step / 5000)
        meta_eps_A = eps + 0.02 * np.cos(2 * np.pi * step / 7000)
        R_A = torch.abs(torch.mean(torch.exp(1j * state_A['x']))).item() if step > 0 else 0.0
        propose_A = 1.0 if R_A < 0.7 else 0.0
        # Model B
        meta_alpha_B = alpha + 0.2 * np.sin(2 * np.pi * step / 5000)
        meta_eps_B = eps + 0.02 * np.cos(2 * np.pi * step / 7000)
        combined_B = (state_B['x'] + state_B['y']) / 2.0
        R_B = torch.abs(torch.mean(torch.exp(1j * combined_B))).item() if step > 0 else 0.0
        propose_B = 1.0 if R_B < 0.7 else 0.0
        # Model C
        self_error_C = np.linalg.norm(predicted_self_C - self_model_C)
        propose_C = 1.0 if self_error_C > 0.5 else 0.0
        # Model D
        self_error_D = np.linalg.norm(predicted_self_D - self_model_D)
        propose_D = 1.0 if self_error_D > 0.5 else 0.0
        # Model E
        self_error_E = np.linalg.norm(predicted_self_E - self_model_E)
        propose_E = 1.0 if self_error_E > 0.5 else 0.0
        # Consensus: majority vote
        votes = [propose_A, propose_B, propose_C, propose_D, propose_E]
        lambda_decouple = 1.0 if sum(votes) >= 3 else 0.0
        # Model A step
        alpha_dyn_A = lambda_decouple * meta_alpha_A + (1 - lambda_decouple) * alpha
        eps_dyn_A = lambda_decouple * meta_eps_A + (1 - lambda_decouple) * eps
        dx_A = -state_A['x'] + bistable_layer(state_A['x'], alpha_dyn_A, theta_eff) + eps_dyn_A * state_A['ws'] + 0.03 * torch.randn(n_layers, device=device)
        state_A['x'] += dt * dx_A
        state_A['ws'] = (1 - k_ws) * state_A['ws'] + k_ws * torch.mean(state_A['x'])
        state_A['R_hist'].append(torch.abs(torch.mean(torch.exp(1j * state_A['x']))).item())
        # Model B step
        state_B['why'] = torch.tanh(1.2 * state_B['why'])
        dx_B = -state_B['x'] + bistable_layer(state_B['x'], alpha_dyn_A, theta_eff + 0.2 * state_B['why'])
        dy_B = -state_B['y'] + bistable_layer(state_B['y'], alpha_dyn_A, theta_eff - 0.2 * state_B['why'])
        dx_B += 0.03 * torch.randn(n_layers, device=device)
        dy_B += 0.03 * torch.randn(n_layers, device=device)
        state_B['x'] += dt * dx_B
        state_B['y'] += dt * dy_B
        combined_B = (state_B['x'] + state_B['y']) / 2.0
        state_B['ws'] = (1 - k_ws) * state_B['ws'] + k_ws * torch.mean(combined_B)
        state_B['x'] += eps_dyn_A * state_B['ws'] * dt
        state_B['y'] += eps_dyn_A * state_B['ws'] * dt
        state_B['R_hist'].append(torch.abs(torch.mean(torch.exp(1j * combined_B))).item())
        # Model C step
        meta_alpha_C = meta_alpha_A
        meta_eps_C = meta_eps_A
        alpha_dyn_C = lambda_decouple * meta_alpha_C + (1 - lambda_decouple) * alpha
        eps_dyn_C = lambda_decouple * meta_eps_C + (1 - lambda_decouple) * eps
        dx_C = -state_C['x'] + bistable_layer(state_C['x'], alpha_dyn_C, theta_eff) + eps_dyn_C * state_C['ws'] + 0.01 * self_error_C + 0.03 * torch.randn(n_layers, device=device)
        state_C['x'] += dt * dx_C
        state_C['ws'] = (1 - k_ws) * state_C['ws'] + k_ws * torch.mean(state_C['x'])
        state_C['R_hist'].append(torch.abs(torch.mean(torch.exp(1j * state_C['x']))).item())
        # Update self_model_C
        x_np_C = state_C['x'].cpu().numpy()
        M_C = len(self_model_C)
        new_self_model_C = np.zeros_like(self_model_C)
        for i in range(M_C):
            start = (i * len(x_np_C)) // M_C
            end = ((i + 1) * len(x_np_C)) // M_C
            avg = np.mean(x_np_C[start:end])
            new_self_model_C[i] = self_model_C[i] * 0.75 + avg * 0.25
        self_model_C = new_self_model_C
        # Model D step
        state_D['why'] = torch.tanh(1.2 * state_D['why'])
        meta_alpha_D = meta_alpha_A
        meta_eps_D = meta_eps_A
        alpha_dyn_D = lambda_decouple * meta_alpha_D + (1 - lambda_decouple) * alpha
        eps_dyn_D = lambda_decouple * meta_eps_D + (1 - lambda_decouple) * eps
        dx_D = -state_D['x'] + bistable_layer(state_D['x'], alpha_dyn_D, theta_eff + 0.2 * state_D['why']) + 0.01 * self_error_D + 0.03 * torch.randn(n_layers, device=device)
        dy_D = -state_D['y'] + bistable_layer(state_D['y'], alpha_dyn_D, theta_eff - 0.2 * state_D['why']) + 0.01 * self_error_D + 0.03 * torch.randn(n_layers, device=device)
        state_D['x'] += dt * dx_D
        state_D['y'] += dt * dy_D
        combined_D = (state_D['x'] + state_D['y']) / 2.0
        state_D['ws'] = (1 - k_ws) * state_D['ws'] + k_ws * torch.mean(combined_D)
        state_D['x'] += eps_dyn_D * state_D['ws'] * dt
        state_D['y'] += eps_dyn_D * state_D['ws'] * dt
        state_D['R_hist'].append(torch.abs(torch.mean(torch.exp(1j * combined_D))).item())
        # Update self_model_D
        x_np_D = state_D['x'].cpu().numpy()
        y_np_D = state_D['y'].cpu().numpy()
        M_D = len(self_model_D) // 2
        new_self_model_D = np.zeros_like(self_model_D)
        for i in range(M_D):
            start = (i * len(x_np_D)) // M_D
            end = ((i + 1) * len(x_np_D)) // M_D
            avg_x = np.mean(x_np_D[start:end])
            avg_y = np.mean(y_np_D[start:end])
            new_self_model_D[i] = self_model_D[i] * 0.75 + avg_x * 0.25
            new_self_model_D[i+M_D] = self_model_D[i+M_D] * 0.75 + avg_y * 0.25
        self_model_D = new_self_model_D
        # Model E step
        state_E['why'] = torch.tanh(1.2 * state_E['why'])
        meta_alpha_E = meta_alpha_A
        meta_eps_E = meta_eps_A
        alpha_dyn_E = lambda_decouple * meta_alpha_E + (1 - lambda_decouple) * alpha
        eps_dyn_E = lambda_decouple * meta_eps_E + (1 - lambda_decouple) * eps
        dx_E = -state_E['x'] + bistable_layer(state_E['x'], alpha_dyn_E, theta_eff + 0.2 * state_E['why']) + 0.01 * self_error_E + 0.03 * torch.randn(n_layers, device=device)
        dy_E = -state_E['y'] + bistable_layer(state_E['y'], alpha_dyn_E, theta_eff - 0.2 * state_E['why']) + 0.01 * self_error_E + 0.03 * torch.randn(n_layers, device=device)
        state_E['x'] += dt * dx_E
        state_E['y'] += dt * dy_E
        combined_E = (state_E['x'] + state_E['y']) / 2.0
        state_E['ws'] = (1 - k_ws) * state_E['ws'] + k_ws * torch.mean(combined_E)
        state_E['x'] += eps_dyn_E * state_E['ws'] * dt
        state_E['y'] += eps_dyn_E * state_E['ws'] * dt
        state_E['R_hist'].append(torch.abs(torch.mean(torch.exp(1j * combined_E))).item())
        # Update self_model_E
        x_np_E = state_E['x'].cpu().numpy()
        y_np_E = state_E['y'].cpu().numpy()
        M_E = len(self_model_E) // 2
        new_self_model_E = np.zeros_like(self_model_E)
        for i in range(M_E):
            start = (i * len(x_np_E)) // M_E
            end = ((i + 1) * len(x_np_E)) // M_E
            avg_x = np.mean(x_np_E[start:end])
            avg_y = np.mean(y_np_E[start:end])
            new_self_model_E[i] = self_model_E[i] * 0.75 + avg_x * 0.25
            new_self_model_E[i+M_E] = self_model_E[i+M_E] * 0.75 + avg_y * 0.25
        self_model_E = new_self_model_E
        if step % 1000 == 0:
            print(f"Consensus step {step}/{steps}: lambda_decouple={lambda_decouple}")
    return {
        'A': state_A['R_hist'],
        'B': state_B['R_hist'],
        'C': state_C['R_hist'],
        'D': state_D['R_hist'],
        'E': state_E['R_hist']
    }


def run_model_B_meta(lambda_decouple=1.0):
    state = {}
    state['x'] = torch.randn(n_layers, device=device)
    state['y'] = torch.randn(n_layers, device=device)
    state['why'] = torch.tensor(0.0, device=device)
    state['ws'] = torch.tensor(0.0, device=device)
    state['R_hist'] = []
    for step in range(steps):
        state['why'] = torch.tanh(1.2 * state['why'])
        meta_alpha = alpha + 0.2 * np.sin(2 * np.pi * step / 5000)
        meta_eps = eps + 0.02 * np.cos(2 * np.pi * step / 7000)
        alpha_dyn = lambda_decouple * meta_alpha + (1 - lambda_decouple) * alpha
        eps_dyn = lambda_decouple * meta_eps + (1 - lambda_decouple) * eps
        dx = -state['x'] + bistable_layer(state['x'], alpha_dyn, theta_eff + 0.2 * state['why'])
        dy = -state['y'] + bistable_layer(state['y'], alpha_dyn, theta_eff - 0.2 * state['why'])
        dx += 0.03 * torch.randn(n_layers, device=device)
        dy += 0.03 * torch.randn(n_layers, device=device)
        state['x'] += dt * dx
        state['y'] += dt * dy
        combined = (state['x'] + state['y']) / 2.0
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(combined)
        state['x'] += eps_dyn * state['ws'] * dt
        state['y'] += eps_dyn * state['ws'] * dt
        R = torch.abs(torch.mean(torch.exp(1j * combined))).item()
        state['R_hist'].append(R)
    return state['R_hist']

def run_model_C_meta(lambda_decouple=1.0):
    state = {}
    state['x'] = torch.randn(n_layers, device=device)
    state['ws'] = torch.tensor(0.0, device=device)
    state['R_hist'] = []
    state['self_error_hist'] = []
    self_model = np.zeros(6)
    from lab import create_meta_tuner_and_optimizer, meta_autotune_update, train_meta_tuner_batch, ALPHA_MIN, ALPHA_MAX, EPS_MIN, EPS_MAX
    meta_tuner, tuner_optimizer = create_meta_tuner_and_optimizer()
    experience_buffer = deque(maxlen=1000)
    for step in range(steps):
        x_np = state['x'].detach().cpu().numpy()
        hist = np.histogram(x_np, bins=32, density=True)[0]
        hist = hist / (np.sum(hist) + 1e-12)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))
        r = torch.abs(torch.mean(torch.exp(1j * state['x']))).item()
        lyap = 0.0
        complexity = float(np.std(x_np))
        self_error = np.linalg.norm(self_model)
        self_awareness = 0.01 * self_error * (self_model[0] if self_model.size > 0 else 0.0)
        alpha_dyn, eps_dyn = meta_autotune_update(meta_tuner, entropy, r, lyap, complexity)
        alpha_dyn = lambda_decouple * alpha_dyn + (1 - lambda_decouple) * alpha
        eps_dyn = lambda_decouple * eps_dyn + (1 - lambda_decouple) * eps
        dx = -state['x'] + bistable_layer(state['x'], alpha_dyn, theta_eff) + eps_dyn * state['ws'] + self_awareness + 0.03 * torch.randn(n_layers, device=device)
        state['x'] += dt * dx
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(state['x'])
        R = torch.abs(torch.mean(torch.exp(1j * state['x']))).item()
        state['R_hist'].append(R)
        # Update self_model (simple moving average)
        M = len(self_model)
        new_self_model = np.zeros_like(self_model)
        for i in range(M):
            start = (i * len(x_np)) // M
            end = ((i + 1) * len(x_np)) // M
            avg = np.mean(x_np[start:end])
            new_self_model[i] = self_model[i] * 0.75 + avg * 0.25
        self_model = new_self_model
        state['self_error_hist'].append(self_error)
        alpha_norm = (alpha_dyn - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)
        eps_norm = (eps_dyn - EPS_MIN) / (EPS_MAX - EPS_MIN)
        reward = entropy + r
        experience_buffer.append(([entropy, r, lyap, complexity], [alpha_norm, eps_norm], reward))
        if len(experience_buffer) > 16 and step % 20 == 0:
            train_meta_tuner_batch(meta_tuner, tuner_optimizer, experience_buffer, batch_size=32, n_epochs=5, lr=1e-3)
    return state['R_hist']

def run_model_D_meta(lambda_decouple=1.0):
    state = {}
    state['x'] = torch.randn(n_layers, device=device)
    state['y'] = torch.randn(n_layers, device=device)
    state['why'] = torch.tensor(0.0, device=device)
    state['ws'] = torch.tensor(0.0, device=device)
    state['R_hist'] = []
    state['self_error_hist'] = []
    self_model = np.zeros(8)
    from lab import create_meta_tuner_and_optimizer, meta_autotune_update, train_meta_tuner_batch, ALPHA_MIN, ALPHA_MAX, EPS_MIN, EPS_MAX
    meta_tuner, tuner_optimizer = create_meta_tuner_and_optimizer()
    experience_buffer = deque(maxlen=1000)
    for step in range(steps):
        state['why'] = torch.tanh(1.2 * state['why'])
        combined = (state['x'] + state['y']) / 2.0
        x_np = combined.detach().cpu().numpy()
        hist = np.histogram(x_np, bins=32, density=True)[0]
        hist = hist / (np.sum(hist) + 1e-12)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))
        r = torch.abs(torch.mean(torch.exp(1j * combined))).item()
        lyap = 0.0
        complexity = float(np.std(x_np))
        self_error = np.linalg.norm(self_model)
        self_awareness = 0.01 * self_error * (self_model[0] if self_model.size > 0 else 0.0)
        alpha_dyn, eps_dyn = meta_autotune_update(meta_tuner, entropy, r, lyap, complexity)
        alpha_dyn = lambda_decouple * alpha_dyn + (1 - lambda_decouple) * alpha
        eps_dyn = lambda_decouple * eps_dyn + (1 - lambda_decouple) * eps
        dx = -state['x'] + bistable_layer(state['x'], alpha_dyn, theta_eff + 0.2 * state['why']) + self_awareness + 0.03 * torch.randn(n_layers, device=device)
        dy = -state['y'] + bistable_layer(state['y'], alpha_dyn, theta_eff - 0.2 * state['why']) + self_awareness + 0.03 * torch.randn(n_layers, device=device)
        state['x'] += dt * dx
        state['y'] += dt * dy
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(combined)
        state['x'] += eps_dyn * state['ws'] * dt
        state['y'] += eps_dyn * state['ws'] * dt
        R = torch.abs(torch.mean(torch.exp(1j * combined))).item()
        state['R_hist'].append(R)
        # Update self_model (simple moving average)
        x_np_x = state['x'].cpu().numpy()
        x_np_y = state['y'].cpu().numpy()
        M = len(self_model) // 2
        new_self_model = np.zeros_like(self_model)
        for i in range(M):
            start = (i * len(x_np_x)) // M
            end = ((i + 1) * len(x_np_x)) // M
            avg_x = np.mean(x_np_x[start:end])
            avg_y = np.mean(x_np_y[start:end])
            new_self_model[i] = self_model[i] * 0.75 + avg_x * 0.25
            new_self_model[i+M] = self_model[i+M] * 0.75 + avg_y * 0.25
        self_model = new_self_model
        state['self_error_hist'].append(self_error)
        alpha_norm = (alpha_dyn - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)
        eps_norm = (eps_dyn - EPS_MIN) / (EPS_MAX - EPS_MIN)
        reward = entropy + r
        experience_buffer.append(([entropy, r, lyap, complexity], [alpha_norm, eps_norm], reward))
        if len(experience_buffer) > 16 and step % 20 == 0:
            train_meta_tuner_batch(meta_tuner, tuner_optimizer, experience_buffer, batch_size=32, n_epochs=5, lr=1e-3)
    return state['R_hist']

def run_model_E_meta(lambda_decouple=1.0):
    state = {}
    state['x'] = torch.randn(n_layers, device=device)
    state['y'] = torch.randn(n_layers, device=device)
    state['why'] = torch.tensor(0.0, device=device)
    state['ws'] = torch.tensor(0.0, device=device)
    state['R_hist'] = []
    state['self_error_hist'] = []
    self_model = np.zeros(8)
    from lab import create_meta_tuner_and_optimizer, meta_autotune_update, train_meta_tuner_batch, ALPHA_MIN, ALPHA_MAX, EPS_MIN, EPS_MAX
    meta_tuner, tuner_optimizer = create_meta_tuner_and_optimizer()
    experience_buffer = deque(maxlen=1000)
    for step in range(steps):
        state['why'] = torch.tanh(1.2 * state['why'])
        combined = (state['x'] + state['y']) / 2.0
        x_np = combined.detach().cpu().numpy()
        hist = np.histogram(x_np, bins=32, density=True)[0]
        hist = hist / (np.sum(hist) + 1e-12)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))
        r = torch.abs(torch.mean(torch.exp(1j * combined))).item()
        lyap = 0.0
        complexity = float(np.std(x_np))
        self_error = np.linalg.norm(self_model)
        self_awareness = 0.01 * self_error * (self_model[0] if self_model.size > 0 else 0.0)
        alpha_dyn, eps_dyn = meta_autotune_update(meta_tuner, entropy, r, lyap, complexity)
        alpha_dyn = lambda_decouple * alpha_dyn + (1 - lambda_decouple) * alpha
        eps_dyn = lambda_decouple * eps_dyn + (1 - lambda_decouple) * eps
        dx = -state['x'] + bistable_layer(state['x'], alpha_dyn, theta_eff + 0.2 * state['why']) + self_awareness + 0.03 * torch.randn(n_layers, device=device)
        dy = -state['y'] + bistable_layer(state['y'], alpha_dyn, theta_eff - 0.2 * state['why']) + self_awareness + 0.03 * torch.randn(n_layers, device=device)
        state['x'] += dt * dx
        state['y'] += dt * dy
        state['ws'] = (1 - k_ws) * state['ws'] + k_ws * torch.mean(combined)
        state['x'] += eps_dyn * state['ws'] * dt
        state['y'] += eps_dyn * state['ws'] * dt
        R = torch.abs(torch.mean(torch.exp(1j * combined))).item()
        state['R_hist'].append(R)
        # Update self_model (simple moving average)
        x_np_x = state['x'].cpu().numpy()
        x_np_y = state['y'].cpu().numpy()
        M = len(self_model) // 2
        new_self_model = np.zeros_like(self_model)
        for i in range(M):
            start = (i * len(x_np_x)) // M
            end = ((i + 1) * len(x_np_x)) // M
            avg_x = np.mean(x_np_x[start:end])
            avg_y = np.mean(x_np_y[start:end])
            new_self_model[i] = self_model[i] * 0.75 + avg_x * 0.25
            new_self_model[i+M] = self_model[i+M] * 0.75 + avg_y * 0.25
        self_model = new_self_model
        state['self_error_hist'].append(self_error)
        alpha_norm = (alpha_dyn - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)
        eps_norm = (eps_dyn - EPS_MIN) / (EPS_MAX - EPS_MIN)
        reward = entropy + r
        experience_buffer.append(([entropy, r, lyap, complexity], [alpha_norm, eps_norm], reward))
        if len(experience_buffer) > 16 and step % 20 == 0:
            train_meta_tuner_batch(meta_tuner, tuner_optimizer, experience_buffer, batch_size=32, n_epochs=5, lr=1e-3)
    return state['R_hist']


print("\nGenerating report (metaoptimizer enabled, consensus)...")
results_meta = run_all_models_meta_with_consensus()
plt.figure(figsize=(12, 6))
for label, R_hist in results_meta.items():
    plt.plot(R_hist, label=f'Model {label} (meta, consensus)')
plt.xlabel('Step')
plt.ylabel('Global Coherence R')
plt.title('Ablation: Global Coherence R over 20K Steps (MetaOptimizer Consensus)')
plt.legend()
plt.tight_layout()
plt.savefig('ablation_report_meta.png')
print('Saved ablation_report_meta.png')
