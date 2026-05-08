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

## Why `solve_ivp` Cannot Train the NN Directly

`solve_ivp` is a black-box numerical integrator — it has no mechanism to
propagate gradients back through the integration steps.  Three approaches
exist to work around this:

| Approach | Cost per gradient step | Scales to 304 params? | Notes |
|----------|----------------------|----------------------|-------|
| **Gradient-free (PSO / DE)** | n_particles × n_SOC forward passes | No — exponential in dimension | Used for κ (2 params) |
| **SPSA** | 2 forward passes always | Yes | Black-box; no code rewrite needed |
| **Differentiable ODE solver** (torchdiffeq / diffrax) | 1 forward + 1 adjoint ODE | Yes | Exact gradients; requires full PyTorch/JAX port of the 60-state RHS |

The paper uses the **adjoint method** via Julia's `SciMLSensitivity.jl`.
`train.py` implements **SPSA** as the practical Python alternative.

### SPSA algorithm (Spall 1998)

```
Δ_k  ~ Bernoulli(±1)         # perturb all params simultaneously
ĝ_k   = [L(θ + c_k·Δ) − L(θ − c_k·Δ)] / (2·c_k·Δ)
θ_{k+1} = θ_k − a_k · ĝ_k

a_k = a / (A + k + 1)^0.602   (decaying step size)
c_k = c / (k + 1)^0.101       (decaying perturbation)
```

Key property: **2 forward passes per step regardless of dimension** (vs 304
passes for finite differences).

## Training Phases

| Phase | Parameters | Algorithm | Typical time | When to use |
|-------|-----------|-----------|-------------|-------------|
| `--phase kappa` | 2 (κ1, κ2) | DE or PSO | Minutes–hours | **Start here** — NN weights already pre-trained from Julia |
| `--phase nn` | 304 (NN + κ) | SPSA | Hours–days | Only if kappa training alone is insufficient |

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

## Files

| File | Method | When to use |
|------|--------|-------------|
| `train.py` | DE / PSO / SPSA — black-box, no code rewrite | κ optimisation; SPSA NN fine-tuning |
| `train_torch.py` | Adam + torchdiffeq adjoint — exact gradients | Full NN + κ training (paper's approach) |

## Recommended Workflow

### Step 1 — κ only (fast, always run first)

```bash
# Optimise κ at 45 °C (paper's training SOCs: 30 %, 50 %, 80 %)
python train.py --temperature 45 --phase kappa --max-rpts 4 --apply

# Adapt κ at other temperatures (NN stays fixed)
python train.py --temperature 25 --phase kappa --max-rpts 4 --apply
python train.py --temperature 0  --phase kappa --max-rpts 4 --apply
```

### Step 2a — NN fine-tuning via SPSA (no extra dependencies)

```bash
# 2 forward passes per step regardless of parameter count
python train.py --temperature 45 --phase nn --max-rpts 4

# Adapt κ at other temperatures with the updated NN
python train.py --temperature 25 --phase kappa --load-nn --apply
python train.py --temperature 0  --phase kappa --load-nn --apply
```

### Step 2b — NN training via PyTorch adjoint (paper's method, exact gradients)

```bash
pip install torch torchdiffeq

# Train NN + κ at 45 °C with Adam (exact adjoint gradients)
python train_torch.py --temperature 45 --epochs 200 --lr 1e-3

# Use --max-days to limit data seen per epoch (speeds up early training)
python train_torch.py --temperature 45 --epochs 200 --max-days 400

# Adapt κ at other temperatures (load saved NN weights)
python train_torch.py --temperature 25 --soc-list 10 80 --load-nn --apply
python train_torch.py --temperature 0  --soc-list 10 80 --load-nn --apply
```

### Step 3 — Validate

```bash
python main.py --temperature 45
```

## Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--temperature` | 45 | Storage temperature in °C (0, 25, or 45) |
| `--phase` | kappa | `kappa` (2 params, DE/PSO) or `nn` (304 params, SPSA) |
| `--max-rpts N` | None (all) | Limit each simulation to N RPT cycles — 5–10× speedup |
| `--optimizer` | de | kappa phase only: `de` (differential evolution) or `pso` |
| `--n-particles` | 15 | DE/PSO population size |
| `--n-iter` | 100 / 500 | Max iterations (kappa default 100; nn default 500) |
| `--spsa-a` | 0.1 | SPSA step size — increase if loss barely moves, decrease if it diverges |
| `--spsa-c` | 0.05 | SPSA perturbation — should be ≈ std(noise in loss) |
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
