"""train_torch.py — PyTorch + torchdiffeq NN training for the UDE model.

Why this file exists
--------------------
scipy.integrate.solve_ivp is a black-box integrator: it has no mechanism to
propagate gradients back through its integration steps.  Backpropagation
through an ODE requires a differentiable solver.

This file replaces solve_ivp with torchdiffeq.odeint_adjoint, which solves a
*second* ODE (the adjoint / co-state equation) backwards in time to compute
dL/dθ without storing the full trajectory.  This is the continuous adjoint
method used by the paper's Julia SciMLSensitivity.jl backend.

Gradient flow:
    θ (NN weights + κ)
     └─► DegradationODE.forward(t, s)
           └─► torchdiffeq adjoint → exact dL/dθ
                 └─► Adam update

Design approximations vs. the full Julia UDE
---------------------------------------------
The full system is a 60-state DAE coupling fast electrochemistry (seconds)
with slow degradation (days).  This file implements only the 3-state
degradation sub-system:

    d/dt [L_SEI, eps_neg, eps_act_neg] = f(NN, κ, state, ψ)

Two approximations make this feasible:

  1. ψ inputs are constant during storage (I = 0 during storage steps)
        I = 0  →  η_SEI_ohmic = I·R_SEI·L_SEI = 0
                   η_SEI_rxn  = (2RT/F)·arcsinh(0) = 0
        ∴  η̄_n = OCP_n(c_n_surf),   η̄_p = OCP_p(c_p_surf)
     Surface concentrations are approximately fixed at the storage SOC,
     so ψ values are computed once from SOC and treated as constants.
     This exploits the large electrochemistry–degradation timescale ratio
     (seconds vs. days) that the paper also uses to speed up the adjoint.

  2. Capacity ∝ eps_act_neg  (LAM-dominated calendar ageing)
        Q_rel = eps_act_neg / eps_act_neg_BOL × 100 %
     Avoids event-triggered discharge simulation (not differentiable).
     Valid because active-material loss is the primary calendar ageing
     mechanism; SEI resistance effects on capacity are secondary.

Requirements
------------
    pip install torch torchdiffeq

Usage
-----
    # Train NN + κ at 45 °C (paper's primary training condition):
    python train_torch.py --temperature 45

    # Smaller learning rate, more epochs, limit to 400 days of data:
    python train_torch.py --temperature 45 --lr 5e-4 --epochs 500 --max-days 400

    # Other temperatures (NN pre-loaded from JSON, only κ adapts):
    python train_torch.py --temperature 25 --soc-list 10 80 --load-nn

    # Write best κ back into main.py after training:
    python train_torch.py --temperature 45 --apply

Outputs
-------
    trained_params.json  — NN weight vectors (Lux flat format) + κ values
    (compatible with train.py --load-nn)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint_adjoint as odeint
except ImportError:
    raise SystemExit(
        "torchdiffeq not installed.\n"
        "Install with:  pip install torchdiffeq\n"
        "Docs:          https://github.com/rtqichen/torchdiffeq"
    )

import model_parameters as Para
import experiment as Exp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LAMBDA1, LAMBDA2 = 0.75, 0.25   # Eq. 14 loss weights
PARAMS_FILE = "trained_params.json"
TRAIN_SOC: dict[int, list[int]] = {45: [30, 50, 80], 25: [10, 80], 0: [10, 80]}

# Default κ starting points (used when training from scratch)
_KAPPA_DEFAULTS: dict[int, tuple[float, float]] = {
    45: (1.0, 1.0),
    25: (0.17, 0.46),
    0:  (0.19, 0.26),
}


# ===========================================================================
# 1.  LuxNet — PyTorch reimplementation of the Julia Lux MLP
# ===========================================================================
class LuxNet(nn.Module):
    """Dense(2→10)→tanh→Dense(10→10)→tanh→Dense(10→1)→square.

    Initialized from the 151-element Lux flat parameter vector stored in
    model_parameters.py.  Lux serialises weights in column-major order
    (out × in, Fortran layout), so we reshape with order='F'.
    """

    def __init__(self, params_np: np.ndarray):
        super().__init__()
        p = params_np
        self.fc1 = nn.Linear(2,  10, bias=True, dtype=torch.float64)
        self.fc2 = nn.Linear(10, 10, bias=True, dtype=torch.float64)
        self.fc3 = nn.Linear(10,  1, bias=True, dtype=torch.float64)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.from_numpy(p[0:20].reshape(10, 2, order="F")))
            self.fc1.bias.copy_(torch.from_numpy(p[20:30]))
            self.fc2.weight.copy_(torch.from_numpy(p[30:130].reshape(10, 10, order="F")))
            self.fc2.bias.copy_(torch.from_numpy(p[130:140]))
            self.fc3.weight.copy_(torch.from_numpy(p[140:150].reshape(1, 10, order="F")))
            self.fc3.bias.copy_(torch.from_numpy(p[150:151]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        return self.fc3(h) ** 2   # square output activation (ensures positivity)

    def to_lux_vec(self) -> np.ndarray:
        """Serialise back to 151-element Lux flat vector for saving / reuse."""
        sd = {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}
        return np.concatenate([
            sd["fc1.weight"].reshape(-1, order="F"),   # (10×2) col-major → 20
            sd["fc1.bias"],                             # 10
            sd["fc2.weight"].reshape(-1, order="F"),   # (10×10) col-major → 100
            sd["fc2.bias"],                             # 10
            sd["fc3.weight"].reshape(-1, order="F"),   # (1×10) col-major → 10
            sd["fc3.bias"],                             # 1
        ])


# ===========================================================================
# 2.  OCP helpers (differentiable torch versions)
# ===========================================================================
def _neg_ocp(c: torch.Tensor) -> torch.Tensor:
    sto = c / Para.c_neg_max
    return (
        1.9793 * torch.exp(-39.3631 * sto) + 0.2482
        - 0.0909  * torch.tanh(29.8538 * (sto - 0.1234))
        - 0.04478 * torch.tanh(14.9159 * (sto - 0.2769))
        - 0.0205  * torch.tanh(30.4444 * (sto - 0.6103))
    )


def _pos_ocp(c: torch.Tensor) -> torch.Tensor:
    sto = c / Para.c_pos_max
    return (
        -0.8090 * sto + 4.4875
        - 0.0428  * torch.tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * torch.tanh(15.7890 * (sto - 0.3117))
        + 17.5842 * torch.tanh(15.9308 * (sto - 0.3120))
    )


def psi_from_soc(soc_frac: float) -> tuple[torch.Tensor, ...]:
    """Compute the four NN inputs ψ1–ψ4 from storage SOC.

    Valid during storage (I = 0):
        η̄_n = OCP_n(c_n_surf),   η̄_p = OCP_p(c_p_surf)
    Surface concentrations are fixed at the stoichiometry for the storage SOC.
    """
    c_n = torch.tensor(
        Para.c_neg_max * (Para.theta_n_SOC0 + soc_frac * (Para.theta_n_SOC100 - Para.theta_n_SOC0)),
        dtype=torch.float64,
    )
    c_p = torch.tensor(
        Para.c_pos_max * (Para.theta_p_SOC0 + soc_frac * (Para.theta_p_SOC100 - Para.theta_p_SOC0)),
        dtype=torch.float64,
    )
    eta_n = _neg_ocp(c_n)
    eta_p = _pos_ocp(c_p)

    x    = 0.16 - eta_n
    psi1 = torch.clamp(x, min=0.0)                                           # ReLU
    psi2 = torch.exp(27.0 * (eta_p - 4.12))
    psi3 = torch.where(x >= 0, torch.exp(10.0 * x), torch.exp(0.15 * x))   # leaky-ReLU
    psi4 = eta_p
    return psi1, psi2, psi3, psi4


# ===========================================================================
# 3.  3-state degradation ODE — compatible with torchdiffeq
# ===========================================================================
class DegradationODE(nn.Module):
    """d/dt [L_SEI, eps_neg, eps_act_neg] as a torch.nn.Module.

    Registering nn_sei, nn_eps, and log_kappa as module attributes causes
    torchdiffeq.odeint_adjoint to automatically compute adjoint sensitivities
    for all of them (they appear in ode.parameters()).

    State vector:
        s[0] = L_SEI        (m)
        s[1] = eps_neg      (–)   electrolyte porosity, negative electrode
        s[2] = eps_act_neg  (–)   active-material volume fraction, negative electrode

    Parameters (learnable):
        nn_sei    — controls SEI growth rate
        nn_eps    — controls active-material loss rate
        log_kappa — [log κ1, log κ2], temperature-dependent rate constants
    """

    def __init__(
        self,
        nn_sei: LuxNet,
        nn_eps: LuxNet,
        log_kappa: nn.Parameter,
        psi1: torch.Tensor,
        psi2: torch.Tensor,
        psi3: torch.Tensor,
        psi4: torch.Tensor,
    ):
        super().__init__()
        self.nn_sei    = nn_sei       # registered as submodule → in parameters()
        self.nn_eps    = nn_eps
        self.log_kappa = log_kappa   # nn.Parameter → in parameters()
        # psi values are constant inputs (not learnable)
        self.register_buffer("psi1", psi1)
        self.register_buffer("psi2", psi2)
        self.register_buffer("psi3", psi3)
        self.register_buffer("psi4", psi4)
        self._gamma = 24.0 * 3600.0  # seconds per day

    def forward(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        L_SEI   = s[0]
        eps_neg = s[1]
        eps_act = s[2]

        kappa1 = torch.exp(self.log_kappa[0])   # log-space keeps κ > 0
        kappa2 = torch.exp(self.log_kappa[1])
        a_neg  = Para.a_neg_BOL * eps_act / Para.eps_act_neg_BOL

        psi_sei = torch.stack([self.psi1, self.psi2])
        psi_eps = torch.stack([self.psi3, self.psi4])

        sei_out = self.nn_sei(psi_sei).squeeze()   # scalar ≥ 0 (square activation)
        eps_out = self.nn_eps(psi_eps).squeeze()   # scalar ≥ 0

        # SEI growth (Eq. 9 in paper, UDE form)
        dL_SEI = (1e-9 / self._gamma) * sei_out * kappa1 / (1e9 * L_SEI)

        # Active-material loss (Eq. 10 in paper, UDE form)
        d_eps_act = (
            -(1.0 / 100.0) / self._gamma * eps_out * kappa2
            / (1.0 + 100.0 * (Para.eps_act_neg_BOL - eps_act))
        )

        # Porosity tracks SEI growth (pore-clogging)
        d_eps_neg = -(1.0 / a_neg) * dL_SEI

        return torch.stack([dL_SEI, d_eps_neg, d_eps_act])


# ===========================================================================
# 4.  Forward pass — one (SOC, temperature) condition
# ===========================================================================
def simulate_soc(
    soc: int,
    temperature: int,
    nn_sei: LuxNet,
    nn_eps: LuxNet,
    log_kappa: nn.Parameter,
    max_days: float | None = None,
) -> dict:
    """Run the 3-state ODE and return differentiable capacity / LAM predictions.

    The ODE is integrated from t = 0 to the last measurement day, sampled
    at every RPT (capacity) and RPTx (LAM) time point.

    torchdiffeq.odeint_adjoint is used so that loss.backward() flows
    through the ODE solution via the adjoint method.
    """
    dates,    exp_cap,  exp_cap_std = Exp.get_capacity_data(temperature, soc)
    lam_dates, lam_mean, lam_std   = Exp.get_LAM_data(temperature, soc)

    if max_days is not None:
        keep_c = dates <= max_days
        dates, exp_cap, exp_cap_std = dates[keep_c], exp_cap[keep_c], exp_cap_std[keep_c]
        keep_l = lam_dates <= max_days
        lam_dates = lam_dates[keep_l]
        lam_mean  = lam_mean[keep_l]
        lam_std   = lam_std[keep_l] if lam_std.ndim == 1 else lam_std[:, keep_l]

    if len(dates) < 2:
        raise ValueError(f"SOC={soc} T={temperature}: fewer than 2 RPT points after max_days filter")

    psi = psi_from_soc(soc / 100.0)
    ode = DegradationODE(nn_sei, nn_eps, log_kappa, *psi)

    s0 = torch.tensor(
        [Para.L_SEI_ini, Para.eps_neg_BOL, Para.eps_act_neg_BOL],
        dtype=torch.float64,
    )

    # Merge RPT and LAM times into one sorted array (torchdiffeq needs increasing t)
    all_t_np = np.unique(np.concatenate([[0.0], dates * 86400.0, lam_dates * 86400.0]))
    all_t    = torch.tensor(all_t_np, dtype=torch.float64)

    # Adjoint integration — gradient flows back through the ODE solution
    # adjoint_params is inferred from ode.parameters() (nn_sei, nn_eps, log_kappa)
    sol = odeint(
        ode, s0, all_t,
        method="dopri5",
        rtol=1e-5,
        atol=1e-7,
        adjoint_rtol=1e-5,
        adjoint_atol=1e-7,
    )
    # sol: (n_times, 3)  — differentiable w.r.t. ode.parameters()

    # Index back to RPT and LAM measurement times
    rpt_idx = [int(np.searchsorted(all_t_np, d * 86400.0)) for d in dates]
    lam_idx = [int(np.searchsorted(all_t_np, d * 86400.0)) for d in lam_dates]

    eps_rpt = sol[rpt_idx, 2]   # eps_act_neg at capacity measurement times
    eps_lam = sol[lam_idx, 2]   # eps_act_neg at LAM measurement times

    # Capacity ≈ eps_act_neg / eps_act_neg_BOL × 100 (LAM-dominated approximation)
    Q_raw  = eps_rpt / Para.eps_act_neg_BOL * 100.0
    Q_norm = Q_raw / Q_raw[0] * 100.0        # normalised so first RPT = 100 %

    LAM_sim = 100.0 * (1.0 - eps_lam / Para.eps_act_neg_BOL)

    Exp_cap_norm = torch.tensor(exp_cap / exp_cap[0] * 100.0, dtype=torch.float64)
    lam_mean_t   = torch.tensor(lam_mean, dtype=torch.float64)
    lam_std_np   = lam_std if lam_std.ndim == 1 else lam_std[0]

    return dict(
        Q_norm=Q_norm,
        Exp_cap_norm=Exp_cap_norm,
        Exp_cap_std_norm=torch.tensor(exp_cap_std / exp_cap[0] * 100.0, dtype=torch.float64),
        LAM_sim=LAM_sim,
        lam_mean=lam_mean_t,
        lam_std=torch.tensor(lam_std_np, dtype=torch.float64),
        dates=dates,
        lam_dates=lam_dates,
        rmse_cap=float(torch.sqrt(torch.mean((Exp_cap_norm - Q_norm) ** 2))),
        rmse_lam=float(torch.sqrt(torch.mean((lam_mean_t - LAM_sim) ** 2))),
    )


# ===========================================================================
# 5.  Differentiable Eq. 14 loss
# ===========================================================================
def eq14_loss(results: list[dict]) -> torch.Tensor:
    """Eq. 14 — L2 loss on capacity and LAM, averaged over N conditions.

    Both tensors are differentiable w.r.t. NN weights and κ via the adjoint.
    """
    total = torch.zeros(1, dtype=torch.float64)
    for r in results:
        cap_e = r["Exp_cap_norm"] - r["Q_norm"]
        lam_e = r["lam_mean"]    - r["LAM_sim"]
        total = (
            total
            + LAMBDA1 * (cap_e @ cap_e) / cap_e.numel()
            + LAMBDA2 * (lam_e @ lam_e) / lam_e.numel()
        )
    return total / len(results)


# ===========================================================================
# 6.  Training loop
# ===========================================================================
def train(
    temperature: int,
    soc_list:    list[int],
    n_epochs:    int   = 200,
    lr:          float = 1e-3,
    max_days:    float | None = None,
    output:      str   = PARAMS_FILE,
    apply_kappa: bool  = False,
    load_nn:     bool  = False,
) -> tuple[LuxNet, LuxNet, nn.Parameter]:
    """Train NN weights + κ with Adam and exact adjoint gradients.

    Returns (nn_sei, nn_eps, log_kappa) — the trained modules.
    """
    # ── Initialise NNs ──────────────────────────────────────────────────────
    sei_params_np = Para.NN_SEI_parameters.copy()
    eps_params_np = Para.NN_eps_parameters.copy()
    if load_nn:
        saved = _load_json(output)
        if "NN_SEI_parameters" in saved:
            sei_params_np = np.array(saved["NN_SEI_parameters"])
            eps_params_np = np.array(saved["NN_eps_parameters"])
            print(f"Loaded NN weights from {output}")
        else:
            print("[warn] --load-nn: no saved NN weights found; using pre-trained Julia weights.")

    nn_sei = LuxNet(sei_params_np)
    nn_eps = LuxNet(eps_params_np)

    k1_0, k2_0 = _KAPPA_DEFAULTS[temperature]
    if load_nn:
        saved = _load_json(output)
        kv = saved.get(f"kappa_{temperature}C")
        if kv:
            k1_0, k2_0 = kv
    log_kappa = nn.Parameter(
        torch.tensor([np.log(k1_0), np.log(k2_0)], dtype=torch.float64)
    )

    all_params = list(nn_sei.parameters()) + list(nn_eps.parameters()) + [log_kappa]
    optimizer  = torch.optim.Adam(all_params, lr=lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-6, verbose=True
    )

    n_params = sum(p.numel() for p in all_params)
    print(
        f"\n{'=' * 60}\n"
        f"  PyTorch adjoint training  (torchdiffeq)\n"
        f"  Temperature = {temperature} °C\n"
        f"  SOC list    = {soc_list}\n"
        f"  Epochs      = {n_epochs},  lr = {lr}\n"
        f"  Parameters  = {n_params}  (NN: {n_params - 2}, κ: 2)\n"
        f"  max_days    = {max_days}\n"
        f"{'=' * 60}\n"
    )

    best_loss  = float("inf")
    best_state: dict | None = None

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        optimizer.zero_grad()

        results = []
        for soc in soc_list:
            try:
                res = simulate_soc(soc, temperature, nn_sei, nn_eps, log_kappa, max_days)
                results.append(res)
            except Exception as exc:
                print(f"  [!] SOC={soc}: {exc}")

        if not results:
            print("All SOC conditions failed — stopping.")
            break

        loss = eq14_loss(results)
        loss.backward()
        nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        loss_val = float(loss)
        k1 = float(torch.exp(log_kappa[0]))
        k2 = float(torch.exp(log_kappa[1]))
        rmse_c = float(np.mean([r["rmse_cap"] for r in results]))
        rmse_l = float(np.mean([r["rmse_lam"] for r in results]))
        elapsed = time.time() - t0

        print(
            f"[{epoch:4d}/{n_epochs}]"
            f"  loss={loss_val:.4f}"
            f"  κ=({k1:.4f}, {k2:.4f})"
            f"  RMSE cap={rmse_c:.3f}%  LAM={rmse_l:.3f}%"
            f"  {elapsed:.1f}s"
        )

        if loss_val < best_loss:
            best_loss  = loss_val
            best_state = dict(
                sei_lux   = nn_sei.to_lux_vec(),
                eps_lux   = nn_eps.to_lux_vec(),
                log_kappa = log_kappa.detach().cpu().numpy().copy(),
            )

        if epoch % 20 == 0 and best_state is not None:
            _save_checkpoint(best_state, temperature, best_loss, output)

    if best_state is not None:
        _save_checkpoint(best_state, temperature, best_loss, output)

    if apply_kappa and best_state is not None:
        k1_b = float(np.exp(best_state["log_kappa"][0]))
        k2_b = float(np.exp(best_state["log_kappa"][1]))
        _apply_kappa_to_main(temperature, k1_b, k2_b)

    return nn_sei, nn_eps, log_kappa


# ===========================================================================
# 7.  Persistence helpers
# ===========================================================================
def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_checkpoint(state: dict, temperature: int, loss: float, path: str) -> None:
    k1 = float(np.exp(state["log_kappa"][0]))
    k2 = float(np.exp(state["log_kappa"][1]))
    existing = _load_json(path)
    existing.update({
        "NN_SEI_parameters":          state["sei_lux"].tolist(),
        "NN_eps_parameters":          state["eps_lux"].tolist(),
        f"kappa_{temperature}C":      [k1, k2],
        f"loss_{temperature}C_torch": float(loss),
    })
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  → saved  κ=({k1:.4f}, {k2:.4f})  loss={loss:.4f}  → {path}")


def _apply_kappa_to_main(temperature: int, k1: float, k2: float) -> None:
    """Patch the ude_kappa line for the given temperature in main.py."""
    main_path = os.path.join(os.path.dirname(__file__), "main.py")
    with open(main_path, encoding="utf-8") as f:
        src = f.read()
    pattern = (
        rf"(if Temperature == {temperature}:.*?sei_phys = [^\n]+\n"
        rf"        ude_kappa = )\([^)]+\)"
    )
    new_src, n = re.subn(pattern, rf"\g<1>({k1}, {k2})", src, flags=re.DOTALL)
    if n == 0:
        print(f"[warn] Could not locate ude_kappa for T={temperature} in main.py")
        return
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(new_src)
    print(f"main.py updated: ude_kappa for {temperature} °C → ({k1:.4f}, {k2:.4f})")


# ===========================================================================
# 8.  CLI
# ===========================================================================
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PyTorch + torchdiffeq adjoint training (Eq. 14 loss)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--temperature", type=int, default=45, choices=[0, 25, 45],
                   help="Storage temperature in °C (default: 45)")
    p.add_argument("--soc-list", type=int, nargs="+", default=None,
                   help="Training SOC values (default: paper values)")
    p.add_argument("--epochs", type=int, default=200,
                   help="Training epochs — each epoch = one Adam step per SOC (default: 200)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam learning rate (default: 1e-3)")
    p.add_argument("--max-days", type=float, default=None,
                   help="Truncate experimental data to this many days (speeds up each epoch)")
    p.add_argument("--load-nn", action="store_true",
                   help="Initialise NN weights from trained_params.json instead of Julia defaults")
    p.add_argument("--output", default=PARAMS_FILE,
                   help="JSON file for saving trained parameters (default: trained_params.json)")
    p.add_argument("--apply", action="store_true",
                   help="After training, patch κ values into main.py")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    soc_list = args.soc_list or TRAIN_SOC[args.temperature]
    train(
        temperature=args.temperature,
        soc_list=soc_list,
        n_epochs=args.epochs,
        lr=args.lr,
        max_days=args.max_days,
        output=args.output,
        apply_kappa=args.apply,
        load_nn=args.load_nn,
    )
