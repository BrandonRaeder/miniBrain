# miniBrain

A minimal simulation laboratory for exploring bistable workspace dynamics, self-referential models, and autotuning in neural-like systems.

## Overview

miniBrain provides a compact, interactive environment to experiment with three related recurrent dynamics models:
- **Option A**: Bistable units with a global workspace coupling.
- **Option B**: Reflective hierarchical "why-loop" architecture.
- **Option C**: Self-referential workspace that maintains and predicts a compressed self-model.

The lab includes background autotuning (meta-tuner NN), perturbations to maintain entropy, and instrumentation for complexity metrics (Shannon entropy, Lyapunov proxy, Lempel-Ziv complexity, mutual information).

## Features

- **Interactive GUI**: Real-time heatmaps, phase coherence plots, rolling windows, and step counter.
- **Autotuning**: Background meta-learning to bias parameters toward high complexity (entropy + coherence).
- **Headless Mode**: CSV export and smoke tests for reproducibility.
- **Metrics**: Entropy, Lyapunov proxy, LZ complexity, pairwise mutual information.
- **Perturbations**: Irrational-time perturbations with drift to avoid resonances.
- **Documentation**: LaTeX findings document in `docs/findings.tex`.

## Installation

### Requirements
- Python 3.10+
- NumPy
- Matplotlib
- scikit-learn
- PyTorch (optional, for meta-tuner; falls back to heuristic if missing)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/BrandonRaeder/miniBrain.git
   cd miniBrain
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install numpy matplotlib scikit-learn torch
   ```

## Usage

### Interactive GUI
Run the full simulation with autotuning:
```bash
python3 lab.py
```
- Displays 3x2 layout: heatmaps + R-phase plots for each model.
- Autotune starts automatically; parameters update in real-time.
- Rolling window (default 500 steps) on phase plots.

### Headless Smoke Test
Quick validation without GUI:
```bash
python3 -c "import matplotlib; matplotlib.use('Agg'); import lab; lab.smoke_test_autotune(2.0)"
```
- Runs autotune for 2 seconds, prints buffer size and samples.

### Custom Runs
- Modify `n_layers`, `dt`, `rolling_window` in `lab.py`.
- For longer runs, adjust `ROLLOUT_STEPS` and reward weights.

## Project Structure

```
miniBrain/
├── lab.py                 # Main simulation and GUI
├── models/
│   └── self.py            # Self-model predictor and utilities
├── tools/
│   └── smoke_autotune.py  # Lightweight headless test
├── docs/
│   └── findings.tex       # LaTeX findings document
├── README.md              # This file
└── .venv/                 # Virtual environment (not committed)
```

## Contributing

- Fork the repo and submit pull requests.
- Report issues on GitHub.
- For large changes, discuss in issues first.

## License

MIT License. See LICENSE file for details.

## References

- Tononi, G. (2008). Consciousness as Integrated Information.
- Wolpert, D. M., et al. (1995). An internal model for sensorimotor integration.

For full details, see `docs/findings.tex`.