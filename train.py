"""train.py — Optimise UDE parameters using the L2 loss from Eq. 14.

Loss function (Eq. 14 of the paper):
    L = (1/N) * Σ_i [ λ1/n_RPT_i  * Σ_j ε_cap_ij²
                     + λ2/n_RPTx_i * Σ_j ε_LAM_ij² ]
    λ1 = 0.75,  λ2 = 0.25

Why solve_ivp cannot train the NN directly
------------------------------------------
scipy's solve_ivp is a black-box numerical integrator: it has no mechanism
to propagate gradients back through the integration steps.  Three practical
alternatives exist:

  1. Gradient-free (PSO / DE) — already in this file.  Works well for the
     2-parameter κ problem, but PSO over 304 NN weights is impractical:
     15 particles × 100 iterations × 3 SOCs = 4 500 forward passes.

  2. SPSA (Simultaneous Perturbation Stochastic Approximation) — estimates
     the gradient from only 2 forward passes regardless of dimension.
     Adds Bernoulli noise to ALL parameters at once, so the gradient cost
     is O(1) in the number of parameters.  This is the default for
     --phase nn.

  3. Differentiable ODE solver (torchdiffeq / diffrax) — rewrite the RHS
     in PyTorch/JAX; the solver then propagates exact gradients via the
     continuous adjoint method (what the paper uses with Julia's
     SciMLSensitivity.jl).  Requires a full PyTorch port of the ODE.

SPSA algorithm (Spall 1998):
  θ_{k+1} = θ_k − a_k * ĝ_k
  ĝ_k     = [L(θ_k + c_k·Δ) − L(θ_k − c_k·Δ)] / (2·c_k·Δ)   Δ ~ Bernoulli(±1)

  a_k = a / (A + k + 1)^α       (decaying step size)
  c_k = c / (k + 1)^γ           (decaying perturbation)

  Recommended constants (Spall 1998): α=0.602, γ=0.101.

Training conditions used in the paper:
  45 °C  →  SOC 30 %, 50 %, 80 %   (NN + κ training)
  25 °C  →  SOC 10 %, 80 %          (κ only; NN fixed from 45 °C)
   0 °C  →  SOC 10 %, 80 %          (κ only; NN fixed from 45 °C)

Usage
-----
  # Optimise κ at 45 °C (recommended first step, ~minutes–hours):
  python train.py --temperature 45 --phase kappa

  # Fine-tune NN weights via SPSA (2 forward passes per step):
  python train.py --temperature 45 --phase nn --max-rpts 4

  # Apply trained κ immediately (updates main.py hardcoded values):
  python train.py --temperature 45 --phase kappa --apply

Speed tip
---------
  --max-rpts N  limits each simulation to the first N RPT cycles.
  Using --max-rpts 4 reduces simulation time by ~5-10× at the cost of
  fitting only the early degradation regime.  After training with a small
  N, re-validate with the full simulation (no --max-rpts).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np
from scipy.optimize import differential_evolution

import model_parameters as Para
from main import run

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LAMBDA1, LAMBDA2 = 0.75, 0.25  # loss weights (Eq. 14)
PARAMS_FILE = "trained_params.json"

# Default training SOC lists (from paper)
TRAIN_SOC: dict[int, list[int]] = {
    45: [30, 50, 80],
    25: [10, 80],
    0:  [10, 80],
}

# Kappa search bounds
KAPPA_BOUNDS = [(0.01, 5.0), (0.01, 5.0)]


# ---------------------------------------------------------------------------
# NN parameter injection (patches module-level globals non-destructively)
# ---------------------------------------------------------------------------
def set_nn_params(sei_vec: np.ndarray, eps_vec: np.ndarray) -> None:
    """Inject new NN weight vectors into model_parameters at runtime."""
    Para._NN_SEI = Para._unpack_lux_chain(np.asarray(sei_vec))
    Para._NN_eps = Para._unpack_lux_chain(np.asarray(eps_vec))


def reset_nn_params() -> None:
    """Restore the original (pre-trained Julia) NN weights."""
    Para._NN_SEI = Para._unpack_lux_chain(Para.NN_SEI_parameters)
    Para._NN_eps = Para._unpack_lux_chain(Para.NN_eps_parameters)


# ---------------------------------------------------------------------------
# Loss function — Eq. 14
# ---------------------------------------------------------------------------
def eq14_loss(sim_results: list[dict]) -> float:
    """Compute the Eq. 14 loss averaged over N experimental conditions.

    Each element of sim_results is the dict returned by main.run().
    Returns a scalar in units of (%²) — lower is better.
    """
    total = 0.0
    for res in sim_results:
        # Capacity term
        cap_err = res["Exp_cap_norm"] - res["Q_norm"]   # percentage points
        cap_loss = float(np.dot(cap_err, cap_err)) / len(cap_err)

        # LAM term
        lam_err = res["lam_mean"] - res["LAM_sim"]      # percentage points
        lam_loss = float(np.dot(lam_err, lam_err)) / len(lam_err)

        total += LAMBDA1 * cap_loss + LAMBDA2 * lam_loss

    return total / len(sim_results)


# ---------------------------------------------------------------------------
# Forward-pass wrapper
# ---------------------------------------------------------------------------
_eval_count = [0]


def evaluate(
    kappa: tuple[float, float],
    temperature: int,
    soc_list: list[int],
    max_rpts: int | None,
    sei_vec: np.ndarray | None = None,
    eps_vec: np.ndarray | None = None,
) -> float:
    """Simulate all SOC conditions and return the Eq. 14 aggregate loss."""
    _eval_count[0] += 1
    tag = f"eval #{_eval_count[0]:4d}"

    if sei_vec is not None:
        set_nn_params(sei_vec, eps_vec)

    results = []
    for soc in soc_list:
        try:
            res = run(
                SOC=soc,
                Temperature=temperature,
                Model="UDE",
                max_rpts=max_rpts,
                verbose=False,
                kappa_override=kappa,
            )
            results.append(res)
        except Exception as exc:
            print(f"  {tag}  SOC={soc} failed: {exc}")
            return 1e6

    loss = eq14_loss(results)
    print(
        f"  {tag}  κ=({kappa[0]:.4f}, {kappa[1]:.4f})"
        f"  loss={loss:.4f}"
        f"  (RMSE cap={np.mean([r['rmse_cap'] for r in results]):.3f}%"
        f"  LAM={np.mean([r['rmse_lam'] for r in results]):.3f}%)"
    )
    return loss


# ---------------------------------------------------------------------------
# PSO — matches paper settings
# ---------------------------------------------------------------------------
def pso(
    loss_fn,
    bounds: list[tuple[float, float]],
    n_particles: int = 15,
    n_iter: int = 100,
    w: float = 0.8,
    c1: float = 2.0,
    c2: float = 2.0,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Particle Swarm Optimisation (PSO).

    Parameters match the paper: n_particles=15, w=0.8, c1=c2=2.0, 100 iter.
    """
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # Initialise
    X = rng.uniform(lo, hi, (n_particles, dim))
    V = np.zeros_like(X)
    pbest = X.copy()
    pbest_val = np.full(n_particles, np.inf)

    # Evaluate initial population
    print(f"\nEvaluating initial population ({n_particles} particles) ...")
    for p in range(n_particles):
        pbest_val[p] = loss_fn(X[p])
    pbest[:] = X

    gbest_idx = int(np.argmin(pbest_val))
    gbest = pbest[gbest_idx].copy()
    gbest_val = float(pbest_val[gbest_idx])
    print(f"Initial best: loss={gbest_val:.4f}  params={np.round(gbest, 4)}\n")

    # Main loop
    for it in range(n_iter):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = np.clip(X + V, lo, hi)

        for p in range(n_particles):
            val = loss_fn(X[p])
            if val < pbest_val[p]:
                pbest_val[p] = val
                pbest[p] = X[p].copy()

        best_idx = int(np.argmin(pbest_val))
        if pbest_val[best_idx] < gbest_val:
            gbest_val = float(pbest_val[best_idx])
            gbest = pbest[best_idx].copy()

        print(
            f"[PSO iter {it + 1:3d}/{n_iter}]"
            f"  best_loss={gbest_val:.4f}"
            f"  params={np.round(gbest, 4)}"
        )

    return gbest, gbest_val


# ---------------------------------------------------------------------------
# Phase 1 — kappa-only optimisation
# ---------------------------------------------------------------------------
def train_kappa(
    temperature: int,
    soc_list: list[int],
    max_rpts: int | None,
    optimizer: str = "de",
    n_particles: int = 15,
    n_iter: int = 100,
) -> tuple[float, float, float]:
    """Optimise κ1, κ2 (2 parameters) for a given temperature.

    Returns (kappa1, kappa2, best_loss).
    """
    print(
        f"\n{'=' * 60}\n"
        f"  Kappa optimisation\n"
        f"  Temperature = {temperature} °C\n"
        f"  SOC list    = {soc_list}\n"
        f"  max_rpts    = {max_rpts}\n"
        f"  Optimizer   = {optimizer}\n"
        f"{'=' * 60}\n"
    )
    _eval_count[0] = 0

    def loss_fn(x):
        return evaluate((float(x[0]), float(x[1])), temperature, soc_list, max_rpts)

    t0 = time.time()

    if optimizer == "pso":
        best, best_loss = pso(
            loss_fn, KAPPA_BOUNDS,
            n_particles=n_particles, n_iter=n_iter,
        )
    else:  # differential evolution
        # popsize * len(bounds) = total pop; use 8 → 16 particles (≈ paper's 15)
        _de_iter = [0]

        def _de_callback(xk, convergence):
            _de_iter[0] += 1
            print(
                f"[DE gen {_de_iter[0]:3d}]"
                f"  best params={np.round(xk, 4)}"
                f"  convergence={convergence:.4f}"
            )

        result = differential_evolution(
            loss_fn,
            KAPPA_BOUNDS,
            maxiter=n_iter,
            popsize=8,        # total = 8 × 2 = 16 particles
            seed=0,
            tol=1e-4,
            polish=True,
            callback=_de_callback,
            updating="deferred",
        )
        best, best_loss = result.x, float(result.fun)

    elapsed = time.time() - t0
    k1, k2 = float(best[0]), float(best[1])
    print(
        f"\nOptimisation complete in {elapsed:.0f} s"
        f"  ({_eval_count[0]} evaluations)\n"
        f"  Best κ1 = {k1:.4f},  κ2 = {k2:.4f},  loss = {best_loss:.4f}\n"
    )
    return k1, k2, best_loss


# ---------------------------------------------------------------------------
# SPSA — gradient estimator for high-dimensional black-box problems
# ---------------------------------------------------------------------------
def spsa(
    loss_fn,
    theta0: np.ndarray,
    n_iter: int = 500,
    a: float = 0.1,
    c: float = 0.05,
    A: float = 50.0,
    alpha: float = 0.602,
    gamma: float = 0.101,
    clip: tuple[float, float] | None = None,
    seed: int = 0,
    checkpoint_every: int = 50,
    checkpoint_fn=None,
) -> tuple[np.ndarray, float]:
    """Simultaneous Perturbation Stochastic Approximation (Spall 1998).

    Estimates the gradient using only 2 forward passes per iteration,
    regardless of the number of parameters.  Suitable for training 304 NN
    weights through a black-box solve_ivp call.

    The gradient estimate at iteration k:
        Δ_k  ~ Bernoulli(±1)   (random sign vector, same shape as θ)
        ĝ_k  = [L(θ + c_k·Δ) − L(θ − c_k·Δ)] / (2·c_k·Δ)
        θ_{k+1} = clip(θ_k − a_k · ĝ_k)

    Step-size schedules (Spall 1998 recommended exponents):
        a_k = a / (A + k + 1)^α      α = 0.602
        c_k = c / (k + 1)^γ          γ = 0.101

    Parameters
    ----------
    a, c    : Initial step-size and perturbation magnitude.
              Tune a so the first few steps change loss noticeably.
              Tune c ≈ std(noise in loss) to keep signal-to-noise > 1.
    A       : Stability constant — typically 10 % of n_iter.
    clip    : (lo, hi) hard bounds on all parameters; None = unbounded.
    checkpoint_fn : Called as checkpoint_fn(k, theta, loss) every
                    checkpoint_every iterations (e.g. to save progress).
    """
    rng = np.random.default_rng(seed)
    theta = theta0.copy()
    best_theta = theta.copy()
    best_loss = float("inf")

    for k in range(n_iter):
        ak = a / (A + k + 1) ** alpha
        ck = c / (k + 1) ** gamma

        # Bernoulli ±1 perturbation (all parameters perturbed simultaneously)
        delta = rng.choice([-1.0, 1.0], size=len(theta))

        theta_plus  = theta + ck * delta
        theta_minus = theta - ck * delta
        if clip is not None:
            theta_plus  = np.clip(theta_plus,  clip[0], clip[1])
            theta_minus = np.clip(theta_minus, clip[0], clip[1])

        L_plus  = loss_fn(theta_plus)
        L_minus = loss_fn(theta_minus)

        # Central-difference gradient estimate
        grad_est = (L_plus - L_minus) / (2.0 * ck * delta)

        theta = theta - ak * grad_est
        if clip is not None:
            theta = np.clip(theta, clip[0], clip[1])

        # Track best seen (SPSA is noisy; best ≠ last)
        avg_loss = 0.5 * (L_plus + L_minus)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_theta = theta.copy()

        print(
            f"[SPSA iter {k + 1:4d}/{n_iter}]"
            f"  a_k={ak:.2e}  c_k={ck:.2e}"
            f"  L+={L_plus:.4f}  L-={L_minus:.4f}  avg={avg_loss:.4f}"
            f"  best={best_loss:.4f}"
        )

        if checkpoint_fn is not None and (k + 1) % checkpoint_every == 0:
            checkpoint_fn(k + 1, best_theta, best_loss)

    return best_theta, best_loss


# ---------------------------------------------------------------------------
# Phase 2 — NN + kappa optimisation via SPSA
# ---------------------------------------------------------------------------
def train_nn_kappa(
    temperature: int,
    soc_list: list[int],
    max_rpts: int | None,
    n_iter: int = 500,
    spsa_a: float = 0.1,
    spsa_c: float = 0.05,
    output_path: str = PARAMS_FILE,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Fine-tune all 304 NN weights + κ1, κ2 via SPSA.

    SPSA needs only 2 forward passes per gradient step — the same cost
    regardless of whether there are 2 or 304 parameters.  This makes it
    tractable where PSO (which needs n_particles × n_iter passes) is not.

    The parameter vector is:
        x = [NN_SEI_params (151), NN_eps_params (151), κ1, κ2]   dim=304

    κ values are log-transformed internally so the search is unconstrained
    while keeping κ > 0.

    Returns (sei_params, eps_params, kappa1, kappa2, best_loss).
    """
    sei0 = Para.NN_SEI_parameters.copy()
    eps0 = Para.NN_eps_parameters.copy()
    n_sei = len(sei0)
    n_eps = len(eps0)

    # Start κ from the temperature-default hardcoded values (log-space)
    _kappa_defaults = {45: (1.0, 1.0), 25: (0.17, 0.46), 0: (0.19, 0.26)}
    k1_0, k2_0 = _kappa_defaults[temperature]
    log_k0 = np.log([k1_0, k2_0])

    # Full parameter vector: [sei_weights, eps_weights, log_κ1, log_κ2]
    theta0 = np.concatenate([sei0, eps0, log_k0])

    print(
        f"\n{'=' * 60}\n"
        f"  NN + kappa optimisation (SPSA)\n"
        f"  Temperature = {temperature} °C\n"
        f"  SOC list    = {soc_list}\n"
        f"  max_rpts    = {max_rpts}\n"
        f"  Parameters  = {len(theta0)}  (NN: {n_sei + n_eps}, κ: 2)\n"
        f"  n_iter      = {n_iter}  ({2 * n_iter} forward passes total)\n"
        f"{'=' * 60}\n"
    )
    _eval_count[0] = 0

    def loss_fn(x: np.ndarray) -> float:
        sei = x[:n_sei]
        eps = x[n_sei : n_sei + n_eps]
        k1  = float(np.exp(x[-2]))
        k2  = float(np.exp(x[-1]))
        return evaluate((k1, k2), temperature, soc_list, max_rpts,
                        sei_vec=sei, eps_vec=eps)

    def _checkpoint(step, theta, loss):
        sei = theta[:n_sei]
        eps = theta[n_sei : n_sei + n_eps]
        k1  = float(np.exp(theta[-2]))
        k2  = float(np.exp(theta[-1]))
        set_nn_params(sei, eps)
        save_params(output_path, {
            "NN_SEI_parameters": sei.tolist(),
            "NN_eps_parameters": eps.tolist(),
            f"kappa_{temperature}C": [k1, k2],
            f"loss_{temperature}C_nn": loss,
        })
        print(f"  [checkpoint @ step {step}]  κ=({k1:.4f}, {k2:.4f})  loss={loss:.4f}")

    t0 = time.time()
    best_theta, best_loss = spsa(
        loss_fn, theta0,
        n_iter=n_iter,
        a=spsa_a,
        c=spsa_c,
        A=0.1 * n_iter,
        clip=None,               # NN weights are unbounded; κ in log-space
        checkpoint_fn=_checkpoint,
    )
    elapsed = time.time() - t0

    sei_best = best_theta[:n_sei]
    eps_best = best_theta[n_sei : n_sei + n_eps]
    k1 = float(np.exp(best_theta[-2]))
    k2 = float(np.exp(best_theta[-1]))

    set_nn_params(sei_best, eps_best)

    print(
        f"\nOptimisation complete in {elapsed:.0f} s"
        f"  ({_eval_count[0]} evaluations)\n"
        f"  Best κ1 = {k1:.4f},  κ2 = {k2:.4f},  loss = {best_loss:.4f}\n"
    )
    return sei_best, eps_best, k1, k2, best_loss


# ---------------------------------------------------------------------------
# Save / load trained parameters
# ---------------------------------------------------------------------------
def save_params(path: str, update: dict) -> None:
    existing: dict = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing.update(update)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Parameters saved → {path}")


def load_params(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Apply trained kappa to main.py (in-place source edit)
# ---------------------------------------------------------------------------
def apply_kappa_to_main(temperature: int, k1: float, k2: float) -> None:
    """Patch the ude_kappa line for the given temperature in main.py."""
    main_path = os.path.join(os.path.dirname(__file__), "main.py")
    with open(main_path) as f:
        src = f.read()

    temp_map = {45: "45", 25: "25", 0: "0"}
    label = temp_map[temperature]

    # Match the ude_kappa line inside the Temperature == X block
    pattern = (
        rf"(if Temperature == {label}:.*?sei_phys = [^\n]+\n"
        rf"        ude_kappa = )\([^)]+\)"
    )
    replacement = rf"\g<1>({k1}, {k2})"
    new_src, n = re.subn(pattern, replacement, src, flags=re.DOTALL)
    if n == 0:
        print(f"[warn] Could not locate ude_kappa for T={temperature} in main.py")
        return
    with open(main_path, "w") as f:
        f.write(new_src)
    print(f"main.py updated: ude_kappa for {temperature} °C → ({k1}, {k2})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train UDE kappa/NN parameters (Eq. 14 loss)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--temperature", type=int, default=45, choices=[0, 25, 45],
                   help="Storage temperature in °C")
    p.add_argument("--phase", choices=["kappa", "nn"], default="kappa",
                   help="'kappa': optimise 2 rate constants; "
                        "'nn': optimise all 304 NN+kappa params (slow)")
    p.add_argument("--soc-list", type=int, nargs="+", default=None,
                   help="SOC values to include in training "
                        "(default: paper training SOCs)")
    p.add_argument("--max-rpts", type=int, default=None,
                   help="Truncate each simulation to this many RPT cycles "
                        "(faster but fits only early degradation)")
    p.add_argument("--optimizer", choices=["de", "pso"], default="de",
                   help="kappa phase: de=differential_evolution (default); "
                        "pso=particle swarm (matches paper)")
    p.add_argument("--n-particles", type=int, default=15,
                   help="PSO population / DE popsize factor (default: 15)")
    p.add_argument("--n-iter", type=int, default=100,
                   help="Iterations for kappa phase (default: 100); "
                        "for nn phase this is SPSA steps (default 500 if unset)")
    p.add_argument("--spsa-a", type=float, default=0.1,
                   help="SPSA initial step size a (nn phase, default: 0.1). "
                        "Increase if early loss barely moves; decrease if it diverges.")
    p.add_argument("--spsa-c", type=float, default=0.05,
                   help="SPSA perturbation magnitude c (nn phase, default: 0.05). "
                        "Should be ≈ std(noise in loss).")
    p.add_argument("--output", default=PARAMS_FILE,
                   help="JSON file for saving trained parameters")
    p.add_argument("--load-nn", action="store_true",
                   help="Load NN weights from --output before training "
                        "(useful when chaining NN → kappa phases)")
    p.add_argument("--apply", action="store_true",
                   help="After training, patch the kappa values in main.py")
    return p.parse_args()


def main() -> None:
    args = _parse()
    soc_list = args.soc_list or TRAIN_SOC[args.temperature]

    # Optionally load previously trained NN weights
    if args.load_nn:
        saved = load_params(args.output)
        if "NN_SEI_parameters" in saved:
            sei = np.array(saved["NN_SEI_parameters"])
            eps = np.array(saved["NN_eps_parameters"])
            set_nn_params(sei, eps)
            print(f"Loaded NN weights from {args.output}")
        else:
            print("[warn] --load-nn specified but no NN weights found in "
                  f"{args.output}; using pre-trained Julia weights.")

    # -----------------------------------------------------------------------
    if args.phase == "kappa":
        k1, k2, loss = train_kappa(
            temperature=args.temperature,
            soc_list=soc_list,
            max_rpts=args.max_rpts,
            optimizer=args.optimizer,
            n_particles=args.n_particles,
            n_iter=args.n_iter,
        )
        save_params(
            args.output,
            {
                f"kappa_{args.temperature}C": [k1, k2],
                f"loss_{args.temperature}C_kappa": loss,
            },
        )
        if args.apply:
            apply_kappa_to_main(args.temperature, k1, k2)
        else:
            print(
                f"\nTo apply: rerun with --apply, or manually set\n"
                f"  ude_kappa = ({k1:.4f}, {k2:.4f})  "
                f"in main.py (Temperature == {args.temperature} block)."
            )

    # -----------------------------------------------------------------------
    else:  # nn
        nn_iters = args.n_iter if args.n_iter != 100 else 500  # default 500 for SPSA
        sei, eps, k1, k2, loss = train_nn_kappa(
            temperature=args.temperature,
            soc_list=soc_list,
            max_rpts=args.max_rpts,
            n_iter=nn_iters,
            spsa_a=args.spsa_a,
            spsa_c=args.spsa_c,
            output_path=args.output,
        )
        save_params(
            args.output,
            {
                "NN_SEI_parameters": sei.tolist(),
                "NN_eps_parameters": eps.tolist(),
                f"kappa_{args.temperature}C": [k1, k2],
                f"loss_{args.temperature}C_nn": loss,
            },
        )
        print(
            "\nNN weights saved.  To load them in future runs, pass --load-nn.\n"
            "To use them directly, copy NN_SEI_parameters / NN_eps_parameters\n"
            "from trained_params.json into model_parameters.py."
        )
        if args.apply:
            apply_kappa_to_main(args.temperature, k1, k2)


if __name__ == "__main__":
    main()
