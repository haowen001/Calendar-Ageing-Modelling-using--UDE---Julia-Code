"""train.py — Optimise UDE parameters using the L2 loss from Eq. 14.

Loss function (Eq. 14 of the paper):
    L = (1/N) * Σ_i [ λ1/n_RPT_i  * Σ_j ε_cap_ij²
                     + λ2/n_RPTx_i * Σ_j ε_LAM_ij² ]
    λ1 = 0.75,  λ2 = 0.25

Two training phases (matching the paper):
  --phase kappa   Optimise κ1, κ2 only (2 params, minutes–hours)
  --phase nn      Fine-tune all 304 NN weights + κ1/κ2 (slow, hours–days)

Optimiser: Particle Swarm Optimisation (PSO) — matches paper settings
  (n_particles=15, w=0.8, c1=c2=2.0, 100 iterations).
  For kappa-only mode, scipy differential_evolution ('de') converges
  faster and is the default.

Training conditions used in the paper:
  45 °C  →  SOC 30 %, 50 %, 80 %   (NN + κ training)
  25 °C  →  SOC 10 %, 80 %          (κ only; NN fixed from 45 °C)
   0 °C  →  SOC 10 %, 80 %          (κ only; NN fixed from 45 °C)

Usage
-----
  # Optimise κ at 45 °C (recommended first step):
  python train.py --temperature 45 --phase kappa

  # Optimise κ at 25 °C (keep NN fixed):
  python train.py --temperature 25 --phase kappa

  # Fine-tune NN weights at 45 °C (very slow — use --max-rpts to limit):
  python train.py --temperature 45 --phase nn --max-rpts 4 --optimizer pso

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
# Phase 2 — NN + kappa optimisation (slow)
# ---------------------------------------------------------------------------
def train_nn_kappa(
    temperature: int,
    soc_list: list[int],
    max_rpts: int | None,
    n_particles: int = 15,
    n_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Fine-tune all 302 NN weights + κ1, κ2 (304 parameters total) via PSO.

    Starts from the pre-trained Julia NN weights as the initial guess region.
    Returns (sei_params, eps_params, kappa1, kappa2, best_loss).

    NOTE: PSO over 304 dimensions is very slow.  Use --max-rpts 3-4 and
    expect multiple hours.  The paper trains the NN using continuous adjoint
    sensitivity (gradient-based); this script uses gradient-free PSO as a
    practical alternative.
    """
    sei0 = Para.NN_SEI_parameters.copy()
    eps0 = Para.NN_eps_parameters.copy()
    n_sei = len(sei0)
    n_eps = len(eps0)

    # Search bounds: ± 2× magnitude of pre-trained weights, min ± 0.5
    def _weight_bounds(v):
        half = np.maximum(np.abs(v) * 2.0, 0.5)
        return list(zip((v - half).tolist(), (v + half).tolist()))

    bounds = _weight_bounds(sei0) + _weight_bounds(eps0) + KAPPA_BOUNDS

    print(
        f"\n{'=' * 60}\n"
        f"  NN + kappa optimisation\n"
        f"  Temperature = {temperature} °C\n"
        f"  SOC list    = {soc_list}\n"
        f"  max_rpts    = {max_rpts}\n"
        f"  Parameters  = {len(bounds)}\n"
        f"{'=' * 60}\n"
    )
    _eval_count[0] = 0

    def loss_fn(x):
        sei = x[:n_sei]
        eps = x[n_sei : n_sei + n_eps]
        k1, k2 = float(x[-2]), float(x[-1])
        return evaluate(
            (k1, k2), temperature, soc_list, max_rpts,
            sei_vec=sei, eps_vec=eps,
        )

    t0 = time.time()
    best, best_loss = pso(loss_fn, bounds, n_particles=n_particles, n_iter=n_iter)
    elapsed = time.time() - t0

    sei_best = best[:n_sei]
    eps_best = best[n_sei : n_sei + n_eps]
    k1, k2 = float(best[-2]), float(best[-1])

    # Restore best NN weights into module for immediate use
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
                   help="de=differential_evolution (default, faster for 2 params); "
                        "pso=particle swarm (matches paper)")
    p.add_argument("--n-particles", type=int, default=15,
                   help="PSO population / DE popsize factor (default: 15)")
    p.add_argument("--n-iter", type=int, default=100,
                   help="Maximum iterations (default: 100)")
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
        sei, eps, k1, k2, loss = train_nn_kappa(
            temperature=args.temperature,
            soc_list=soc_list,
            max_rpts=args.max_rpts,
            n_particles=args.n_particles,
            n_iter=args.n_iter,
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
