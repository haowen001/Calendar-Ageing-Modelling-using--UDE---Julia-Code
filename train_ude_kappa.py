"""Fit only the two UDE degradation scale factors.

This is the stable calibration step for the SciPy reproduction.  It keeps the
trained neural-network weights from ``model_parameters.py`` fixed and optimizes
only the temperature-dependent ``kappa1`` and ``kappa2`` factors against the
relative-capacity and LAM data.

Examples
--------
    python train_ude_kappa.py --soc 85 --temperature 45 --max-nfev 30 --plot results_trained.png
    python main.py --ude-params trained_ude_T45_SOC85.npz --output results_trained.png
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


def _objective_l2(res, args):
    residual = _loss_residuals(
        res, args.capacity_weight, args.lam_weight, args.min_std, args.loss)
    return float(np.dot(residual, residual))


def train(args):
    base_kappa = default_kappa(args.temperature)
    if args.init_params:
        init = np.load(args.init_params)
        if 'ude_kappa' in init:
            base_kappa = init['ude_kappa'].astype(np.float64)
        print(f'Loaded initial kappa from {args.init_params}', flush=True)

    x0 = np.log(base_kappa)
    lower = np.log([args.kappa_min, args.kappa_min])
    upper = np.log([args.kappa_max, args.kappa_max])

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
        kappa = np.exp(x)
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
            cost = 0.5 * float(np.dot(residual, residual))
            if cost < best['cost']:
                best.update(cost=cost, x=x.copy())
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

    kappa = np.exp(x_final)
    final = main.run(
        SOC=args.soc,
        Temperature=args.temperature,
        Model='UDE',
        max_rpts=args.max_rpts,
        verbose=False,
        ude_kappa_override=kappa,
    )

    metadata = {
        'soc': args.soc,
        'temperature': args.temperature,
        'mode': 'kappa',
        'max_rpts': args.max_rpts,
        'max_nfev': args.max_nfev,
        'loss': args.loss,
        'optimizer': args.optimizer,
        'capacity_weight': args.capacity_weight,
        'lam_weight': args.lam_weight,
        'objective_l2': _objective_l2(final, args),
        'ude_kappa': kappa.tolist(),
        'rmse_cap': final['rmse_cap'],
        'rmse_lam': final['rmse_lam'],
        'elapsed_s': time.time() - start,
    }

    output = Path(args.output)
    np.savez(
        output,
        ude_kappa=kappa,
        NN_SEI_parameters=Para.NN_SEI_parameters,
        NN_eps_parameters=Para.NN_eps_parameters,
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
    p.add_argument('--loss', choices=['paper-l2', 'std-weighted'],
                   default='paper-l2',
                   help='paper-l2 matches Eq. 14: unweighted L2 capacity + LAM')
    p.add_argument('--optimizer', choices=['powell', 'least-squares'],
                   default='powell')
    p.add_argument('--max-rpts', type=int, default=None)
    p.add_argument('--init-params', default=None,
                   help='Optional NPZ to continue kappa fitting from')
    p.add_argument('--max-nfev', type=int, default=30)
    p.add_argument('--capacity-weight', type=float, default=1.0)
    p.add_argument('--lam-weight', type=float, default=1.0)
    p.add_argument('--min-std', type=float, default=0.25)
    p.add_argument('--kappa-min', type=float, default=1e-3)
    p.add_argument('--kappa-max', type=float, default=20.0)
    p.add_argument('--xtol', type=float, default=1e-3)
    p.add_argument('--ftol', type=float, default=1e-3)
    p.add_argument('--keep-best', action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument('--penalty', type=float, default=1e3)
    p.add_argument('--output', default=None)
    p.add_argument('--plot', default=None)
    args = p.parse_args()
    if args.output is None:
        args.output = f'trained_ude_T{args.temperature}_SOC{args.soc}.npz'
    return args


if __name__ == '__main__':
    train(parse_args())
