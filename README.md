# miniBrain
I  did this on an hp notebook laptop with a quadcore and 6 gigs of ram which is well over 10 years old.  

A minimal simulation laboratory for exploring bistable workspace dynamics, self-referential models, and autotuning in neural-like systems.

## Overview

miniBrain provides a compact, interactive environment to experiment with three related recurrent dynamics models:
- **Option A**: Bistable units with a global workspace coupling.
- **Option B**: Hierarchical reflective architecture.
- **Option C**: Self-referential workspace that maintains and predicts a compressed self-model.

These models are designed to sustain high entropy and coherence (R) indefinitely, enabling simulations to run for extremely long durations without degradation.

The lab includes background autotuning (meta-tuner NN), perturbations to maintain entropy, and instrumentation for complexity metrics (Shannon entropy, Lyapunov proxy, Lempel-Ziv complexity, mutual information).

## Features

- **Interactive GUI**: Real-time heatmaps (all models), phase coherence, live diagnostics, self-error.
- **Performance**: 400x speedup (0.12-0.22ms/step), caching, Numba JIT, reduced buffers/history.
- **Autotuning**: Meta-NN tunes α/ε toward high entropy+coherence; independent per-model.
- **Headless Tests**: `performance_test.py`, `headless_performance_test.py`, `tools/smoke_autotune.py`.
- **Metrics**: Cached Shannon entropy, Lyapunov proxy, LZ complexity, self-prediction error.
- **Models**: A (global workspace), B (reflective hierarchy), C (self-ref), D (hierarchical self-ref).
- **Self-Models**: NN predictors of own compressed state; hierarchical meta-cognition.
- **Docs**: LaTeX `docs/findings.tex`, `performance_summary.md`.

## Installation

### Quick Setup (Recommended)
```bash
pip install -r requirements.txt
```

### Full Setup
1. Clone:
   ```bash
   git clone https://github.com/BrandonRaeder/miniBrain.git
   cd miniBrain
   ```

2. Virtual env (optional):
   ```bash
   python -m venv .venv
   # Activate: source .venv/bin/activate (Linux/Mac) or .venv\Scripts\activate (Win)
   ```

3. Install:
   ```bash
   pip install -r requirements.txt
   ```
   Requirements: numpy matplotlib scikit-learn torch numba pandas

## Usage

### Main Optimized GUI (2 Self-Ref Models)
```bash
python lab.py
```
- Real-time heatmaps, coherence, self-error, diagnostics.
- Autotuning starts after GUI loads.
- Models C+D: self-ref + hierarchical self-ref.

### All Three Models Demo
```bash
python lab_all_three_optimized.py
```
- Side-by-side A/B/C with live metrics/FPS.

### Performance Benchmarks
```bash
python performance_test.py
python headless_performance_test.py
```
- Measures ms/step, FPS for all models.

### Smoke Tests (Headless)
```bash
python -c "import lab; lab.smoke_test_autotune(2.0)"
python tools/smoke_autotune.py
```

### Self-Model Utils
```bash
python -c "from models.self import compare_self_reference_levels; compare_self_reference_levels()"
python -c "from models.self import animate_self_reference_realtime; animate_self_reference_realtime()"
```

### Custom Runs
Edit `n_layers=100`, `dt=0.05`, `alpha=1.95`, `eps=0.08` in scripts.

## Project Structure

```
miniBrain/
├── lab.py                          # Optimized GUI (self-ref models C+D)
├── lab_all_three_optimized.py      # All 3 models (A/B/C) demo + FPS
├── lab_fixed_optimized.py          # Legacy optimized
├── performance_test.py             # Benchmarks (headless/GUI)
├── headless_performance_test.py    # No matplotlib tests
├── models/
│   └── self.py                     # Self-model NN, predictors, utils
├── tools/
│   └── smoke_autotune.py           # Headless autotune validation
├── docs/
│   └── findings.tex                # Technical paper (LaTeX)
├── performance_summary.md          # 400x speedup details
├── requirements.txt
├── test_entropy_fix.py             # Unit tests
├── test_metric_display.py
├── test_standardization.py
└── ... (heatmaps, animations)
```

## Performance

See [`performance_summary.md`](performance_summary.md) for details.

**Benchmark (50 layers, 1000 steps):**

| Model                  | Time   | ms/step | FPS    | Final R |
|------------------------|--------|---------|--------|---------|
| **Option A**           | 0.118s | **0.12**| **8471**| 0.579  |
| **Option B**           | 0.130s | **0.13**| **7700**| 0.790  |
| **Option C**           | 0.218s | **0.22**| **4585**| 0.580  |

**Speedup: 100-400x** vs original "seconds/step".

## Contributing

- Fork the repo and submit pull requests.
- Report issues on GitHub.
- For large changes, discuss in issues first.

## License

Refer to license.md for use cases.

## References

- Tononi, G. (2008). Consciousness as Integrated Information.
- Wolpert, D. M., et al. (1995). An internal model for sensorimotor integration.

For full details, see `docs/findings.tex`.
