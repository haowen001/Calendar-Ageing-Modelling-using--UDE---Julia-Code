"""Differentiable PyTorch trainer for the UDE neural networks.

This script trains the UDE degradation networks with PyTorch autograd and a
differentiable ODE solve.  It intentionally lives beside the SciPy reproduction
instead of replacing it:

* ``main.py`` remains the full SPMe/RPT reproduction.
* this trainer learns UDE degradation dynamics against capacity and LAM data,
  then exports the same ``.npz`` format that ``main.py --ude-params`` can load.

The differentiable model integrates the calendar-ageing degradation subsystem
(``L_SEI`` and ``eps_act_neg``) during storage.  Capacity is represented by a
small differentiable readout from the learned degradation states, because the
full RPT capacity protocol in ``main.py`` uses event-driven SciPy solves and is
not differentiable.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

np = None
Exp = None
main = None
Para = None


def _require_torch():
    # Windows conda environments can load MKL/OpenMP through NumPy/SciPy before
    # Torch, which trips PyTorch's duplicate OpenMP runtime guard.
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            'PyTorch is required for train_ude_torch.py. Install dependencies '
            'with: pip install -r requirements.txt'
        ) from exc
    return torch


def default_kappa(temperature):
    if temperature == 45:
        return np.array([1.0, 1.0], dtype=np.float64)
    if temperature == 25:
        return np.array([0.17, 0.46], dtype=np.float64)
    if temperature == 0:
        return np.array([0.19, 0.26], dtype=np.float64)
    raise ValueError(f'Unsupported temperature {temperature}')


def _to_tensor(torch, value, device):
    return torch.as_tensor(value, dtype=torch.float64, device=device)


class LuxVectorNet:
    """Torch equivalent of Lux Chain(Dense(2,10), Dense(10,10), Dense(10,1,sqr))."""

    def __init__(self, torch, parameters, device, trainable=True):
        self.torch = torch
        p = _to_tensor(torch, parameters, device).clone()
        p.requires_grad_(trainable)
        self.p = p

    def parameters(self):
        return [self.p] if self.p.requires_grad else []

    def vector(self):
        return self.p.detach().cpu().numpy().astype(np.float64)

    def __call__(self, x):
        torch = self.torch
        W1 = self.p[0:20].reshape(2, 10).T
        b1 = self.p[20:30]
        W2 = self.p[30:130].reshape(10, 10).T
        b2 = self.p[130:140]
        W3 = self.p[140:150].reshape(10, 1).T
        b3 = self.p[150:151]
        h = torch.tanh(W1 @ x + b1)
        h = torch.tanh(W2 @ h + b2)
        y = (W3 @ h + b3) ** 2
        return y.squeeze()


class DegradationUDE:
    def __init__(self, torch, sei_net, eps_net, eta_n, eta_p, log_kappa):
        self.torch = torch
        self.sei_net = sei_net
        self.eps_net = eps_net
        self.eta_n = eta_n
        self.eta_p = eta_p
        self.log_kappa = log_kappa

    def __call__(self, t, y):
        torch = self.torch
        L_sei = torch.clamp(y[0], min=1e-12)
        eps_act = torch.clamp(y[1], min=1e-6)
        kappa = torch.exp(self.log_kappa)

        psi1 = torch.relu(0.16 - self.eta_n)
        psi2 = torch.exp(27.0 * (self.eta_p - 4.12))
        x = 0.16 - self.eta_n
        psi3 = torch.exp(10.0 * torch.where(x >= 0.0, x, 0.015 * x))
        psi4 = self.eta_p

        dL_dt_seconds = (
            1e-9 * self.sei_net(torch.stack([psi1, psi2])) *
            kappa[0] / (86400.0 * 1e9 * L_sei)
        )
        d_eps_dt_seconds = (
            -(1.0 / 100.0) * self.eps_net(torch.stack([psi3, psi4])) *
            kappa[1] /
            (86400.0 * (1.0 + 100.0 * (Para.eps_act_neg_BOL - eps_act)))
        )

        # The independent variable in this trainer is days, not seconds.
        return torch.stack([86400.0 * dL_dt_seconds,
                            86400.0 * d_eps_dt_seconds])


def torch_negative_ocp(torch, surf_conc):
    sto = surf_conc / Para.c_neg_max
    return (
        1.9793 * torch.exp(-39.3631 * sto)
        + 0.2482
        - 0.0909 * torch.tanh(29.8538 * (sto - 0.1234))
        - 0.04478 * torch.tanh(14.9159 * (sto - 0.2769))
        - 0.0205 * torch.tanh(30.4444 * (sto - 0.6103))
    )


def torch_positive_ocp(torch, surf_conc):
    sto = surf_conc / Para.c_pos_max
    return (
        -0.8090 * sto
        + 4.4875
        - 0.0428 * torch.tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * torch.tanh(15.7890 * (sto - 0.3117))
        + 17.5842 * torch.tanh(15.9308 * (sto - 0.3120))
    )


def storage_ocp_features(torch, soc, device):
    soc_frac = soc / 100.0
    c_n = Para.c_neg_max * (
        Para.theta_n_SOC0 +
        (Para.theta_n_SOC100 - Para.theta_n_SOC0) * soc_frac
    )
    c_p = Para.c_pos_max * (
        Para.theta_p_SOC0 +
        (Para.theta_p_SOC100 - Para.theta_p_SOC0) * soc_frac
    )
    eta_n = torch_negative_ocp(torch, _to_tensor(torch, c_n, device))
    eta_p = torch_positive_ocp(torch, _to_tensor(torch, c_p, device))
    return eta_n, eta_p


def fixed_rk4(torch, func, y0, t_eval, step_days):
    """Differentiable fixed-step RK4 fallback used when torchdiffeq is absent."""
    ys = [y0]
    y = y0
    t = t_eval[0]
    for target in t_eval[1:]:
        while bool((target - t) > 1e-12):
            dt = torch.minimum(target - t, torch.as_tensor(
                step_days, dtype=t_eval.dtype, device=t_eval.device))
            k1 = func(t, y)
            k2 = func(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = func(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = func(t + dt, y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + dt
        ys.append(y)
    return torch.stack(ys)


def integrate_degradation(torch, func, y0, t_eval, args):
    if args.solver == 'torchdiffeq':
        try:
            from torchdiffeq import odeint
        except ImportError as exc:
            raise SystemExit(
                'torchdiffeq is required for --solver torchdiffeq. '
                'Install dependencies with: pip install -r requirements.txt'
            ) from exc
        return odeint(
            func, y0, t_eval, method=args.ode_method,
            rtol=args.rtol, atol=args.atol,
        )
    return fixed_rk4(torch, func, y0, t_eval, args.rk4_step_days)


def train(args):
    global np, Exp, main, Para
    torch = _require_torch()
    import numpy as np_module
    import experiment as Exp_module
    import main as main_module
    import model_parameters as Para_module
    np = np_module
    Exp = Exp_module
    main = main_module
    Para = Para_module
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    dates, exp_capacity, _ = Exp.get_capacity_data(args.temperature, args.soc)
    lam_dates, lam_mean, _ = Exp.get_LAM_data(args.temperature, args.soc)
    if args.max_rpts is not None and args.max_rpts < len(dates):
        dates = dates[:args.max_rpts]
        exp_capacity = exp_capacity[:args.max_rpts]
        keep = lam_dates <= dates[-1]
        lam_dates = lam_dates[keep]
        lam_mean = lam_mean[keep]

    cap_target = _to_tensor(
        torch, exp_capacity / exp_capacity[0] * 100.0, device)
    lam_target = _to_tensor(torch, lam_mean, device)

    all_times = np.unique(np.concatenate([[0.0], dates, lam_dates])).astype(float)
    t_eval = _to_tensor(torch, all_times, device)
    cap_indices = [int(np.where(np.isclose(all_times, d))[0][0]) for d in dates]
    lam_indices = [int(np.where(np.isclose(all_times, d))[0][0]) for d in lam_dates]

    sei_p = Para.NN_SEI_parameters.copy()
    eps_p = Para.NN_eps_parameters.copy()
    kappa = default_kappa(args.temperature)
    if args.init_params:
        init = np.load(args.init_params)
        if 'NN_SEI_parameters' in init:
            sei_p = init['NN_SEI_parameters'].astype(np.float64)
        if 'NN_eps_parameters' in init:
            eps_p = init['NN_eps_parameters'].astype(np.float64)
        if 'ude_kappa' in init:
            kappa = init['ude_kappa'].astype(np.float64)
        print(f'Loaded initial UDE parameters from {args.init_params}')

    sei_net = LuxVectorNet(torch, sei_p, device, trainable=not args.freeze_nn)
    eps_net = LuxVectorNet(torch, eps_p, device, trainable=not args.freeze_nn)
    log_kappa = _to_tensor(torch, np.log(kappa), device).clone()
    log_kappa.requires_grad_(not args.freeze_kappa)

    # Positive readout coefficients.  They map differentiable degradation
    # states to relative capacity for Eq. 14 training.
    raw_cap_lam = _to_tensor(torch, args.cap_lam_init, device).clone()
    raw_cap_sei = _to_tensor(torch, args.cap_sei_init, device).clone()
    raw_cap_lam.requires_grad_(True)
    raw_cap_sei.requires_grad_(True)

    params = []
    params += sei_net.parameters()
    params += eps_net.parameters()
    if log_kappa.requires_grad:
        params.append(log_kappa)
    params += [raw_cap_lam, raw_cap_sei]

    optimizer = torch.optim.Adam(params, lr=args.lr)
    eta_n, eta_p = storage_ocp_features(torch, args.soc, device)
    y0 = _to_tensor(torch, [Para.L_SEI_ini, Para.eps_act_neg_BOL], device)

    base_sei = _to_tensor(torch, sei_p, device)
    base_eps = _to_tensor(torch, eps_p, device)
    best = {'loss': float('inf'), 'state': None}
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        func = DegradationUDE(torch, sei_net, eps_net, eta_n, eta_p, log_kappa)
        sol = integrate_degradation(torch, func, y0, t_eval, args)
        L = sol[:, 0]
        eps_act = sol[:, 1]
        lam_sim = 100.0 - 100.0 * eps_act / Para.eps_act_neg_BOL

        cap_lam = torch.nn.functional.softplus(raw_cap_lam)
        cap_sei = torch.nn.functional.softplus(raw_cap_sei)
        cap_sim = 100.0 - cap_lam * lam_sim - cap_sei * (
            (L - Para.L_SEI_ini) / Para.L_SEI_ini)

        cap_res = cap_sim[cap_indices] - cap_target
        lam_res = lam_sim[lam_indices] - lam_target
        loss_data = (
            args.capacity_weight * torch.sum(cap_res ** 2) +
            args.lam_weight * torch.sum(lam_res ** 2)
        )
        reg = args.regularization * (
            torch.mean((sei_net.p - base_sei) ** 2) +
            torch.mean((eps_net.p - base_eps) ** 2)
        )
        loss = loss_data + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()

        with torch.no_grad():
            if not args.freeze_kappa:
                log_kappa.clamp_(np.log(args.kappa_min), np.log(args.kappa_max))
            sei_net.p.clamp_(base_sei - args.nn_bound, base_sei + args.nn_bound)
            eps_net.p.clamp_(base_eps - args.nn_bound, base_eps + args.nn_bound)

            rmse_cap = torch.sqrt(torch.mean(cap_res ** 2)).item()
            rmse_lam = torch.sqrt(torch.mean(lam_res ** 2)).item()
            loss_value = loss_data.item()
            if loss_value < best['loss']:
                best['loss'] = loss_value
                best['state'] = (
                    sei_net.vector(),
                    eps_net.vector(),
                    torch.exp(log_kappa).detach().cpu().numpy().astype(np.float64),
                    cap_lam.item(),
                    cap_sei.item(),
                    rmse_cap,
                    rmse_lam,
                )

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f'epoch {epoch:04d}: loss={loss_value:.6g}, '
                f'RMSE cap={rmse_cap:.4g} %, LAM={rmse_lam:.4g} %, '
                f'kappa=({torch.exp(log_kappa)[0].item():.5g}, '
                f'{torch.exp(log_kappa)[1].item():.5g})',
                flush=True,
            )

    sei_best, eps_best, kappa_best, cap_lam, cap_sei, rmse_cap, rmse_lam = best['state']
    metadata = {
        'soc': args.soc,
        'temperature': args.temperature,
        'mode': 'torch-degradation-ude',
        'solver': args.solver,
        'epochs': args.epochs,
        'capacity_weight': args.capacity_weight,
        'lam_weight': args.lam_weight,
        'regularization': args.regularization,
        'ude_kappa': kappa_best.tolist(),
        'capacity_readout_lam': cap_lam,
        'capacity_readout_sei': cap_sei,
        'surrogate_rmse_cap': rmse_cap,
        'surrogate_rmse_lam': rmse_lam,
        'elapsed_s': time.time() - start,
    }

    output = Path(args.output)
    np.savez(
        output,
        ude_kappa=kappa_best,
        NN_SEI_parameters=sei_best,
        NN_eps_parameters=eps_best,
        metadata=json.dumps(metadata, indent=2),
    )
    print(f'Saved Torch-trained UDE parameters to {output}')
    print(json.dumps(metadata, indent=2))

    if args.validate_main:
        Para.set_NN_parameters(sei_best, eps_best)
        res = main.run(
            SOC=args.soc,
            Temperature=args.temperature,
            Model='UDE',
            max_rpts=args.max_rpts,
            verbose=True,
            ude_kappa_override=kappa_best,
        )
        if args.plot:
            main.plot_results(res, output=args.plot)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--soc', type=int, default=85)
    p.add_argument('--temperature', type=int, default=45, choices=[0, 25, 45])
    p.add_argument('--init-params', default=None)
    p.add_argument('--output', default=None)
    p.add_argument('--plot', default='results_trained_torch.png')
    p.add_argument('--validate-main', action='store_true')
    p.add_argument('--max-rpts', type=int, default=None)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=666)
    p.add_argument('--solver', choices=['rk4', 'torchdiffeq'], default='rk4')
    p.add_argument('--ode-method', default='dopri5')
    p.add_argument('--rk4-step-days', type=float, default=1.0)
    p.add_argument('--rtol', type=float, default=1e-6)
    p.add_argument('--atol', type=float, default=1e-8)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--grad-clip', type=float, default=10.0)
    p.add_argument('--regularization', type=float, default=1e-4)
    p.add_argument('--capacity-weight', type=float, default=1.0)
    p.add_argument('--lam-weight', type=float, default=1.0)
    p.add_argument('--nn-bound', type=float, default=0.5)
    p.add_argument('--freeze-kappa', action='store_true')
    p.add_argument('--freeze-nn', action='store_true')
    p.add_argument('--kappa-min', type=float, default=1e-3)
    p.add_argument('--kappa-max', type=float, default=20.0)
    p.add_argument('--cap-lam-init', type=float, default=0.0)
    p.add_argument('--cap-sei-init', type=float, default=-8.0)
    p.add_argument('--print-every', type=int, default=25)
    args = p.parse_args()
    if args.output is None:
        args.output = f'trained_ude_torch_T{args.temperature}_SOC{args.soc}.npz'
    return args


if __name__ == '__main__':
    train(parse_args())
