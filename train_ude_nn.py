"""Fine-tune the UDE neural-network parameters.

The Python/SciPy reproduction cannot backpropagate through ``solve_ivp`` like
the differentiable Julia/SciML UDE training loop.  This script therefore uses
SPSA, a derivative-free optimizer that can perturb many NN weights with only
two forward simulations per iteration.

Examples
--------
    python train_ude_nn.py --soc 85 --temperature 45 --mode last-layer --freeze-kappa --init-params trained_ude_T45_SOC85.npz --max-nfev 80
    python train_ude_nn.py --soc 85 --temperature 45 --mode nn --freeze-kappa --init-params trained_ude_T45_SOC85.npz --max-nfev 120
    python main.py --ude-params trained_ude_nn_T45_SOC85.npz --output results_trained_nn.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize

import main
import model_parameters as Para


def default_kappa(temperature):
    if temperature == 45:
        return np.array([1.0, 1.0], dtype=np.float64)
    if temperature == 25:
        return np.array([0.17, 0.46], dtype=np.float64)
    if temperature == 0:
        return np.array([0.19, 0.26], dtype=np.float64)
    raise ValueError(f'Unsupported temperature {temperature}')


def _lam_std_1d(lam_std):
    return lam_std[0] if getattr(lam_std, 'ndim', 1) > 1 else lam_std


def _loss_residuals(res, cap_weight, lam_weight, min_std, loss):
    if loss == 'paper-l2':
        cap_sigma = 1.0
        lam_sigma = 1.0
    elif loss == 'std-weighted':
        cap_sigma = np.maximum(res['Exp_cap_std_norm'], min_std)
        lam_sigma = np.maximum(_lam_std_1d(res['lam_std']), min_std)
    else:
        raise ValueError(f'Unsupported loss {loss}')
    cap = np.sqrt(cap_weight) * (res['Q_norm'] - res['Exp_cap_norm']) / cap_sigma
    lam = np.sqrt(lam_weight) * (res['LAM_sim'] - res['lam_mean']) / lam_sigma
    return np.concatenate([cap, lam])


def _loss_value(res, cap_weight, lam_weight, min_std, loss):
    residual = _loss_residuals(res, cap_weight, lam_weight, min_std, loss)
    return float(np.dot(residual, residual))


def make_packers(mode, base_kappa, base_sei, base_eps, nn_bound,
                 freeze_kappa=False):
    tail = slice(140, 151)

    if mode == 'last-layer':
        x0 = np.concatenate([np.log(base_kappa), np.zeros(22)])
        lower = np.concatenate([
            np.log([1e-3, 1e-3]), -nn_bound * np.ones(22)])
        upper = np.concatenate([
            np.log([20.0, 20.0]), nn_bound * np.ones(22)])

        def unpack(x):
            sei = base_sei.copy()
            eps = base_eps.copy()
            sei[tail] += x[2:13]
            eps[tail] += x[13:24]
            return np.exp(x[:2]), sei, eps, x[2:]

        if freeze_kappa:
            lower[:2] = x0[:2]
            upper[:2] = x0[:2]
        return x0, lower, upper, unpack

    if mode == 'nn':
        x0 = np.concatenate([np.log(base_kappa), np.zeros(302)])
        lower = np.concatenate([
            np.log([1e-3, 1e-3]), -nn_bound * np.ones(302)])
        upper = np.concatenate([
            np.log([20.0, 20.0]), nn_bound * np.ones(302)])

        def unpack(x):
            sei = base_sei + x[2:153]
            eps = base_eps + x[153:304]
            return np.exp(x[:2]), sei, eps, x[2:]

        if freeze_kappa:
            lower[:2] = x0[:2]
            upper[:2] = x0[:2]
        return x0, lower, upper, unpack

    raise ValueError(f'Unsupported mode {mode}')


def run_spsa(objective, x0, lower, upper, args):
    """Derivative-free SPSA optimizer for high-dimensional black-box UDE fits."""
    rng = np.random.default_rng(args.seed)
    x = x0.copy()
    best_x = x.copy()
    best_value = objective(x)
    print(f'spsa initial objective={best_value:.6g}', flush=True)

    for k in range(1, args.max_nfev // 2 + 1):
        ak = args.spsa_a / ((k + args.spsa_A) ** args.spsa_alpha)
        ck = args.spsa_c / (k ** args.spsa_gamma)
        delta = rng.choice([-1.0, 1.0], size=x.size)

        x_plus = np.clip(x + ck * delta, lower, upper)
        x_minus = np.clip(x - ck * delta, lower, upper)
        y_plus = objective(x_plus)
        y_minus = objective(x_minus)
        ghat = (y_plus - y_minus) / (2.0 * ck) * delta

        x = np.clip(x - ak * ghat, lower, upper)
        current = objective(x) if args.spsa_eval_current else min(y_plus, y_minus)
        if current < best_value:
            best_value = current
            best_x = x.copy()

        print(
            f'spsa iter {k:03d}: best={best_value:.6g}, '
            f'last_pair=({y_plus:.6g}, {y_minus:.6g}), '
            f'ak={ak:.3g}, ck={ck:.3g}',
            flush=True,
        )

    return best_x


def train(args):
    base_sei = Para.NN_SEI_parameters.copy()
    base_eps = Para.NN_eps_parameters.copy()
    base_kappa = default_kappa(args.temperature)
    if args.init_params:
        init = np.load(args.init_params)
        if 'NN_SEI_parameters' in init:
            base_sei = init['NN_SEI_parameters'].astype(np.float64)
        if 'NN_eps_parameters' in init:
            base_eps = init['NN_eps_parameters'].astype(np.float64)
        if 'ude_kappa' in init:
            base_kappa = init['ude_kappa'].astype(np.float64)
        print(f'Loaded initial UDE parameters from {args.init_params}',
              flush=True)
    x0, lower, upper, unpack = make_packers(
        args.mode, base_kappa, base_sei, base_eps, args.nn_bound,
        args.freeze_kappa)

    baseline = main.run(
        SOC=args.soc,
        Temperature=args.temperature,
        Model='UDE',
        max_rpts=args.max_rpts,
        verbose=False,
        ude_kappa_override=base_kappa,
    )
    baseline_residual = _loss_residuals(
        baseline, args.capacity_weight, args.lam_weight, args.min_std,
        args.loss)
    best = {
        'cost': 0.5 * float(np.dot(baseline_residual, baseline_residual)),
        'res': baseline,
        'x': x0.copy(),
    }
    print(
        f'baseline: RMSE cap={baseline["rmse_cap"]:.4g} %, '
        f'LAM={baseline["rmse_lam"]:.4g} %',
        flush=True,
    )
    start = time.time()
    eval_count = 0

    def evaluate(x):
        nonlocal eval_count
        eval_count += 1
        kappa, sei, eps, tuned_offsets = unpack(x)
        Para.set_NN_parameters(sei, eps)
        try:
            res = main.run(
                SOC=args.soc,
                Temperature=args.temperature,
                Model='UDE',
                max_rpts=args.max_rpts,
                verbose=False,
                ude_kappa_override=kappa,
            )
            residual = _loss_residuals(
                res, args.capacity_weight, args.lam_weight, args.min_std,
                args.loss)
            if args.regularization > 0.0 and tuned_offsets.size:
                residual = np.concatenate([
                    residual,
                    np.sqrt(args.regularization) * tuned_offsets,
                ])
            cost = 0.5 * float(np.dot(residual, residual))
            if cost < best['cost']:
                best.update(cost=cost, res=res, x=x.copy())
            print(
                f'eval {eval_count:03d}: cost={cost:.4g}, '
                f'kappa=({kappa[0]:.5g}, {kappa[1]:.5g}), '
                f'RMSE cap={res["rmse_cap"]:.4g} %, '
                f'LAM={res["rmse_lam"]:.4g} %',
                flush=True,
            )
            return residual
        except Exception as exc:
            print(f'eval {eval_count:03d}: failed with {exc}', flush=True)
            return np.full_like(baseline_residual, args.penalty)

    if args.optimizer == 'powell':
        def scalar_objective(x):
            residual = evaluate(x)
            return float(np.dot(residual, residual))

        fit = minimize(
            scalar_objective,
            x0,
            method='Powell',
            bounds=list(zip(lower, upper)),
            options={
                'maxfev': args.max_nfev,
                'xtol': args.xtol,
                'ftol': args.ftol,
                'disp': True,
            },
        )
        x_final = best['x'] if args.keep_best else fit.x
    elif args.optimizer == 'spsa':
        def scalar_objective(x):
            residual = evaluate(x)
            return float(np.dot(residual, residual))

        x_spsa = run_spsa(scalar_objective, x0, lower, upper, args)
        x_final = best['x'] if args.keep_best else x_spsa
    else:
        fit = least_squares(
            evaluate,
            x0,
            bounds=(lower, upper),
            max_nfev=args.max_nfev,
            x_scale='jac',
            verbose=2,
        )
        x_final = best['x'] if args.keep_best else fit.x

    kappa, sei, eps, _ = unpack(x_final)
    Para.set_NN_parameters(sei, eps)
    final = main.run(
        SOC=args.soc,
        Temperature=args.temperature,
        Model='UDE',
        max_rpts=args.max_rpts,
        verbose=False,
        ude_kappa_override=kappa,
    )

    output = Path(args.output)
    metadata = {
        'soc': args.soc,
        'temperature': args.temperature,
        'mode': args.mode,
        'max_rpts': args.max_rpts,
        'max_nfev': args.max_nfev,
        'loss': args.loss,
        'optimizer': args.optimizer,
        'capacity_weight': args.capacity_weight,
        'lam_weight': args.lam_weight,
        'regularization': args.regularization,
        'objective_l2': _loss_value(
            final, args.capacity_weight, args.lam_weight, args.min_std,
            args.loss),
        'ude_kappa': kappa.tolist(),
        'rmse_cap': final['rmse_cap'],
        'rmse_lam': final['rmse_lam'],
        'elapsed_s': time.time() - start,
    }
    np.savez(
        output,
        ude_kappa=kappa,
        NN_SEI_parameters=sei,
        NN_eps_parameters=eps,
        metadata=json.dumps(metadata, indent=2),
    )
    print(f'Saved fitted UDE parameters to {output}')
    print(json.dumps(metadata, indent=2))

    if args.plot:
        main.plot_results(final, output=args.plot)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--soc', type=int, default=85)
    p.add_argument('--temperature', type=int, default=45, choices=[0, 25, 45])
    p.add_argument('--mode', choices=['last-layer', 'nn'],
                   default='last-layer',
                   help='last-layer fits NN outputs; nn fits all NN weights')
    p.add_argument('--loss', choices=['paper-l2', 'std-weighted'],
                   default='paper-l2',
                   help='paper-l2 matches Eq. 14: unweighted L2 capacity + LAM')
    p.add_argument('--optimizer', choices=['spsa', 'least-squares', 'powell'],
                   default='spsa')
    p.add_argument('--max-rpts', type=int, default=None,
                   help='Use fewer RPTs for quick experiments')
    p.add_argument('--init-params', default=None,
                   help='Optional NPZ to continue training from')
    p.add_argument('--max-nfev', type=int, default=30,
                   help='Maximum optimizer model evaluations')
    p.add_argument('--capacity-weight', type=float, default=1.0)
    p.add_argument('--lam-weight', type=float, default=1.0)
    p.add_argument('--min-std', type=float, default=0.25,
                   help='Minimum percent-data sigma used for residual scaling')
    p.add_argument('--regularization', type=float, default=1e-2,
                   help='L2 penalty for NN final-layer offsets')
    p.add_argument('--nn-bound', type=float, default=0.5,
                   help='Absolute bound for NN parameter offsets from the loaded values')
    p.add_argument('--freeze-kappa', action='store_true',
                   help='Keep kappa fixed while tuning NN parameters')
    p.add_argument('--xtol', type=float, default=1e-3)
    p.add_argument('--ftol', type=float, default=1e-3)
    p.add_argument('--keep-best', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='Save the best evaluated parameters, not just final iterate')
    p.add_argument('--seed', type=int, default=666)
    p.add_argument('--spsa-a', type=float, default=0.002,
                   help='SPSA learning-rate numerator')
    p.add_argument('--spsa-c', type=float, default=0.05,
                   help='SPSA perturbation numerator')
    p.add_argument('--spsa-A', type=float, default=10.0)
    p.add_argument('--spsa-alpha', type=float, default=0.602)
    p.add_argument('--spsa-gamma', type=float, default=0.101)
    p.add_argument('--spsa-eval-current', action='store_true',
                   help='Spend an extra solve per SPSA iteration to evaluate the updated point')
    p.add_argument('--penalty', type=float, default=1e3)
    p.add_argument('--output', default=None)
    p.add_argument('--plot', default=None,
                   help='Optional output PNG for the fitted model')
    args = p.parse_args()
    if args.output is None:
        args.output = f'trained_ude_nn_T{args.temperature}_SOC{args.soc}.npz'
    return args


if __name__ == '__main__':
    train(parse_args())
