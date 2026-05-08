# UDE Parameter Training Guide

Training code to improve UDE model predictions by optimising the rate constants
(κ1, κ2) and optionally the neural-network weights against the L2 loss from
**Eq. 14** of the paper.

## Loss Function (Eq. 14)

```
L = (1/N) · Σᵢ [ λ1/nᵢ_RPT · Σⱼ ε_cap_ij²  +  λ2/nᵢ_RPTx · Σⱼ ε_LAM_ij² ]
```

| Symbol | Meaning |
|--------|---------|
| N | Number of experimental conditions (SOC–temperature pairs) |
| nᵢ_RPT / nᵢ_RPTx | Number of RPT / RPTx measurements for condition i |
| ε_cap, ε_LAM | Prediction errors in normalised capacity and LAM (% points) |
| λ1 = 0.75, λ2 = 0.25 | Loss weights |

## Files Changed

| File | Change |
|------|--------|
| `main.py` | `run()` accepts a new optional `kappa_override=(k1, k2)` argument |
| `train.py` | New training script (PSO + differential evolution) |

## Training Phases

| Phase | Parameters | Typical time | When to use |
|-------|-----------|-------------|-------------|
| `--phase kappa` | 2 (κ1, κ2) | Minutes–hours | **Start here** — NN weights are already pre-trained from Julia |
| `--phase nn` | 304 (NN weights + κ) | Hours–days | Only if kappa training alone is insufficient |

> **Note:** The paper trains NN weights using continuous adjoint sensitivity
> (gradient-based). `train.py` uses gradient-free PSO as a practical alternative.

## Optimisers

| Flag | Algorithm | Notes |
|------|-----------|-------|
| `--optimizer de` | Scipy `differential_evolution` | **Default** — faster convergence for 2 params |
| `--optimizer pso` | Particle Swarm Optimisation | Matches paper: population=15, w=0.8, c1=c2=2.0 |

## Training Conditions (from paper)

| Temperature | Training SOCs | Phase |
|-------------|--------------|-------|
| 45 °C | 30 %, 50 %, 80 % | NN + κ |
| 25 °C | 10 %, 80 % | κ only (NN fixed from 45 °C) |
| 0 °C | 10 %, 80 % | κ only (NN fixed from 45 °C) |

## Recommended Workflow

### Quick start (κ only — recommended)

```bash
# 1. Optimise κ at 45 °C using the paper's training SOCs
python train.py --temperature 45 --phase kappa --max-rpts 4

# 2. Patch main.py automatically with the best κ found
python train.py --temperature 45 --phase kappa --max-rpts 4 --apply

# 3. Repeat for other temperatures (NN stays fixed)
python train.py --temperature 25 --phase kappa --max-rpts 4 --apply
python train.py --temperature 0  --phase kappa --max-rpts 4 --apply

# 4. Validate with the full simulation
python main.py --temperature 45
```

### Full NN fine-tuning (advanced)

```bash
# Train NN weights + κ at 45 °C
python train.py --temperature 45 --phase nn --max-rpts 3

# Then adapt κ at other temperatures with the new NN
python train.py --temperature 25 --phase kappa --load-nn --apply
python train.py --temperature 0  --phase kappa --load-nn --apply
```

## Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--temperature` | 45 | Storage temperature in °C (0, 25, or 45) |
| `--phase` | kappa | `kappa` (2 params) or `nn` (304 params) |
| `--max-rpts N` | None (all) | Limit each simulation to N RPT cycles — 5–10× speedup |
| `--optimizer` | de | `de` (differential evolution) or `pso` |
| `--n-particles` | 15 | Population size (PSO particles or DE population factor) |
| `--n-iter` | 100 | Maximum iterations |
| `--apply` | off | Patch the found κ values into `main.py` automatically |
| `--load-nn` | off | Load NN weights from `trained_params.json` before training |
| `--soc-list` | (paper defaults) | Override training SOC values, e.g. `--soc-list 50 85` |
| `--output` | `trained_params.json` | JSON file for saving results |

## Output

All trained parameters are saved to `trained_params.json`:

```json
{
  "kappa_45C": [0.342, 0.817],
  "loss_45C_kappa": 0.1234,
  "kappa_25C": [0.318, 0.461],
  "NN_SEI_parameters": [...],
  "NN_eps_parameters": [...]
}
```

To apply trained κ manually (without `--apply`), edit the relevant block in `main.py`:

```python
elif Temperature == 45:
    ude_kappa = (0.342, 0.817)   # replace with values from trained_params.json
```
