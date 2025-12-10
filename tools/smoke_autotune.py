#!/usr/bin/env python3
import numpy as np
import threading
import time
from collections import deque

# simple fallback meta-autotune (same logic as lab fallback)
ALPHA_MIN, ALPHA_MAX = 0.5, 2.5
EPS_MIN, EPS_MAX = 0.01, 1.0

experience_buffer = deque(maxlen=2000)


def meta_autotune_update(entropy, r, lyap, complexity):
    e_norm = min(1.0, entropy / (1.0 + entropy))
    r_norm = (r + 1.0) / 2.0
    alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * (0.6 * e_norm + 0.4 * r_norm)
    eps = EPS_MIN + (EPS_MAX - EPS_MIN) * (0.7 * e_norm + 0.3 * r_norm)
    return float(alpha), float(eps)


def shannon_entropy(data, bins=20):
    if len(data)==0:
        return 0.0
    hist, _ = np.histogram(data, bins=bins, density=False)
    if np.sum(hist) == 0:
        return 0.0
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist))


def lyapunov_proxy(R_hist):
    diffs = np.abs(np.diff(R_hist))
    return np.mean(diffs) if len(diffs) > 0 else 0.0


def bistable_layer(x, alpha, theta_eff):
    return np.tanh(alpha * x - theta_eff)


def _autotune_worker(states, stop_event, interval=0.1, retrain_every=20):
    counter = 0
    while not stop_event.is_set():
        for state in states:
            R_hist = np.array(state.get('R_hist', []))
            entropy = shannon_entropy(R_hist) if len(R_hist) > 0 else 0.0
            r = float(R_hist[-1]) if len(R_hist) > 0 else 0.0
            lyap = lyapunov_proxy(R_hist)
            complexity = entropy + lyap
            alpha_s, eps_s = meta_autotune_update(entropy, r, lyap, complexity)
            state['alpha'] = alpha_s
            state['eps'] = eps_s
            norm_alpha = (alpha_s - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN) if (ALPHA_MAX - ALPHA_MIN) != 0 else 0.0
            norm_eps = (eps_s - EPS_MIN) / (EPS_MAX - EPS_MIN) if (EPS_MAX - EPS_MIN) != 0 else 0.0
            experience_buffer.append(((entropy, r, lyap, complexity), (norm_alpha, norm_eps)))
        counter += 1
        stop_event.wait(interval)


def start_autotune_for_states(states, interval=0.1):
    stop_event = threading.Event()
    t = threading.Thread(target=_autotune_worker, args=(states, stop_event, interval), daemon=True)
    t.start()
    return stop_event, t


def smoke_test(duration_s=1.0):
    n_layers = 100
    theta_eff = 0.0
    k_ws = 0.002
    dt = 0.05
    # initial params
    alpha0 = 1.95
    eps0 = 0.08

    state_c = {
        'x': np.random.randn(n_layers),
        'ws': 0.0,
        'R_hist': [],
        'x_history': np.zeros((n_layers, 2000)),
        'step': 0,
        'predicted_self': np.zeros(6),
        'self_model': np.zeros(6),
        'self_model_hist': [],
        'alpha': alpha0,
        'eps': eps0
    }

    stop_event, thread = start_autotune_for_states([state_c], interval=0.05)

    steps = int(duration_s / 0.01)
    for t in range(steps):
        a = state_c.get('alpha', alpha0)
        e = state_c.get('eps', eps0)
        dx = -state_c['x'] + bistable_layer(state_c['x'], a, theta_eff) + e * state_c['ws'] + 0.01 * np.random.randn(n_layers)
        state_c['x'] += dt * dx
        state_c['ws'] = (1 - k_ws) * state_c['ws'] + k_ws * np.mean(state_c['x'])
        R_c = np.mean(np.exp(1j * state_c['x'])).real
        state_c['R_hist'].append(R_c)
        mean_activity = np.mean(state_c['x'])
        std_activity = np.std(state_c['x'])
        trend = np.mean(np.diff(state_c['R_hist'][-10:])) if len(state_c['R_hist']) > 10 else 0.0
        ws_norm = np.tanh(state_c['ws'])
        entropy_c = -np.sum([r * np.log(r + 1e-10) for r in state_c['R_hist'][-10:] if r > 0]) if len(state_c['R_hist']) > 10 else 0.0
        coherence = np.abs(np.mean(np.exp(1j * state_c['x'])))
        self_model = np.array([mean_activity, std_activity, trend, ws_norm, entropy_c, coherence])
        state_c['self_model'] = self_model
        state_c['self_model_hist'].append(self_model.copy())
        state_c['step'] += 1
        time.sleep(0.01)

    stop_event.set()
    thread.join(timeout=1.0)
    print('Smoke run finished')
    print('Experience buffer size:', len(experience_buffer))
    for i, item in enumerate(list(experience_buffer)[-10:]):
        print(i, item)


if __name__ == '__main__':
    smoke_test(1.0)
