
# === MAGIC EDGE-OF-LIFE PARAMETERS (fixed for ablation) ===
n_layers  = 100
alpha     = 1.95
theta_eff = 0.0
eps       = 0.08
k_ws      = 0.002
dt        = 0.05
gamma     = 2.8
anti_mult = 1.2
noise_mult = 1.2

from lab import animate_workspace_heatmap_forever

# --- Add main block to launch GUI for ablation experiment ---
if __name__ == "__main__":
    print("Starting ablation experiment GUI...")
    animate_workspace_heatmap_forever(
        n_layers=n_layers,
        dt=dt,
        alpha=alpha,
        eps=eps,
        theta_eff=theta_eff,
        k_ws=k_ws,
        autostart_autotune=False
    )

# ablationLab.py
# Direct copy of lab.py with MetaOptimizer (meta-tuner/optimizer) frozen for ablation experiments.
# Meta-tuner/optimizer is FROZEN: all meta-parameters (alpha, eps, anti_mult, noise_mult) are fixed and do not adapt.

import time
script_start_time = time.time()
print(f"[{time.time() - script_start_time:.4f}s] Script execution started.")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy
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
	torch.autograd.set_detect_anomaly(True)
	import torch.nn as nn
	import torch.optim as optim
	HAVE_TORCH = True
	HAVE_IPEX = False
	if torch.cuda.is_available():
		if hasattr(torch.version, 'hip') and torch.version.hip is not None:
			device = torch.device('cuda')
			print("Using device: AMD GPU (ROCm)")
		else:
			device = torch.device('cuda')
			print("Using device: NVIDIA GPU (CUDA)")
	else:
		ipex = None
		try:
			try:
				import intel_extension_for_pytorch as ipex
			except ImportError:
				pass
			HAVE_IPEX = True
		except ImportError:
			pass
		if HAVE_IPEX and torch.xpu.is_available():
			device = torch.device('xpu')
			print("Using device: Intel GPU (XPU)")
		else:
			device = torch.device('cpu')
			print("Using device: CPU")
		try:
			import intel_extension_for_pytorch as ipex
		except ImportError:
			ipex = None
		HAVE_IPEX = ipex is not None
	print(f"Selected device: {device}")
	print(f"[{time.time() - script_start_time:.4f}s] PyTorch device setup finished.")
except Exception:
	torch = None
	nn = None
	optim = None
	HAVE_TORCH = False
	HAVE_IPEX = False
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

# ...[rest of lab.py code is copied here, but with the following changes for ablation]...

# --- ABLATION: Freeze meta-tuner/optimizer for all models ---
# 1. All meta-parameters (alpha, eps, anti_mult, noise_mult) are fixed constants for the entire run.
# 2. Remove or bypass all meta-tuner/optimizer training and parameter updates.
# 3. For models C, D, E: set alpha, eps, anti_mult, noise_mult to fixed values at initialization and do not update.
# 4. Experience buffer and meta-tuner objects are not used.

# === MAGIC EDGE-OF-LIFE PARAMETERS (fixed for ablation) ===
n_layers  = 100
alpha     = 1.95
theta_eff = 0.0
eps       = 0.08
k_ws      = 0.002
dt        = 0.05
gamma     = 2.8
anti_mult = 1.2
noise_mult = 1.2

# ...[rest of lab.py code, but everywhere meta-tuner/optimizer would update alpha/eps/anti_mult/noise_mult, use the above fixed values instead]...

# For example, in animate_workspace_heatmap_forever and all model state updates:
#   - Remove all meta-tuner/optimizer calls, experience buffer logic, and parameter adaptation.
#   - Set state['alpha'] = alpha, state['eps'] = eps, state['anti_mult'] = anti_mult, state['noise_mult'] = noise_mult at initialization and do not change.
#   - All reward/experience/meta-tuner training code is removed or commented out.

# The rest of the simulation, visualization, and model logic remains as in lab.py, but with meta-parameters frozen.
