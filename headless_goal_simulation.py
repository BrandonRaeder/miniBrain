import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from lab import create_meta_tuner_and_optimizer, MetaTunerNN, bistable_layer

# --- Simulation parameters ---
n_layers = 100
dt = 0.05
alpha = 1.95
eps = 0.08
theta_eff = 0.0
k_ws = 0.002
steps = 20000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model state ---
state = {}
state['x'] = torch.randn(n_layers, device=device)
state['ws'] = torch.tensor(0.0, device=device)
state['meta_tuner'], state['tuner_optimizer'] = create_meta_tuner_and_optimizer()
state['experience_buffer'] = deque(maxlen=2500)
state['goal_pattern'] = np.sin(np.linspace(0, 2 * np.pi, n_layers))
state['lstm_memory'] = nn.LSTM(input_size=n_layers, hidden_size=64, num_layers=1, batch_first=True).to(device)
state['lstm_hidden'] = None
state['lstm_last_out'] = None
state['R_hist'] = []
state['goal_dist_hist'] = []

# --- LSTM update utility ---
def update_lstm_memory(state, obs):
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    if state['lstm_hidden'] is None:
        output, hidden = state['lstm_memory'](obs_tensor)
    else:
        output, hidden = state['lstm_memory'](obs_tensor, state['lstm_hidden'])
    state['lstm_hidden'] = (hidden[0].detach(), hidden[1].detach())
    return output.detach().cpu().numpy().flatten()

# --- Main simulation loop ---
for step in range(steps):
    if step % 1000 == 0:
        print(f"Step {step}/{steps}")
    obs = state['x'].cpu().numpy()
    lstm_out = update_lstm_memory(state, obs)
    state['lstm_last_out'] = lstm_out
    # Dynamics (simple model C)
    noise = 0.03 * np.random.randn(n_layers)
    lstm_mod = 0.05 * np.mean(lstm_out)
    dx = -state['x'].cpu().numpy() + bistable_layer(state['x'].cpu().numpy(), alpha, theta_eff) + eps * state['ws'].cpu().numpy() + noise + lstm_mod
    state['x'] = torch.tensor(state['x'].cpu().numpy() + dt * dx, dtype=torch.float32, device=device)
    state['ws'] = torch.tensor((1 - k_ws) * state['ws'].cpu().numpy() + k_ws * np.mean(state['x'].cpu().numpy()), dtype=torch.float32, device=device)
    # Goal distance
    goal_dist = np.linalg.norm(state['x'].cpu().numpy() - state['goal_pattern']) / len(state['goal_pattern'])
    state['goal_dist_hist'].append(goal_dist)
    # Reward (example: negative goal distance)
    reward = -goal_dist
    state['R_hist'].append(reward)

# --- Plot and save ---
plt.figure(figsize=(10,4))
plt.plot(state['goal_dist_hist'], label='Goal Distance')
plt.xlabel('Step')
plt.ylabel('Distance to Goal')
plt.title('Goal Distance over 20K Steps')
plt.legend()
plt.tight_layout()
plt.savefig('goal_distance_20k.png')
print('Saved goal_distance_20k.png')
