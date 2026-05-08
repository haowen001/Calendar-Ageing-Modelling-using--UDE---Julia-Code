"""Python port of Main.jl.

Runs the SPMe + SEI calendar-ageing simulation (Physics or UDE variant) and
plots model vs. experimental capacity and anode LAM.

Differences from the Julia driver:

* Julia uses a singular mass-matrix DAE (last state algebraic). Python's
  ``solve_ivp`` does not support DAEs natively, so the algebraic equation is
  enforced via a stiff penalty ODE handled by the BDF method.
* Step transitions are driven by terminal events (Voltage / |Current|) or by
  integrating the full step duration (Time).

Usage
-----
    python main.py                # SOC=85, T=45 C, UDE
    python main.py --soc 50 --temperature 25 --model Physics
    python main.py --max-rpts 3   # quick smoke test
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless friendly
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import model_parameters as Para
import experiment as Exp


# Constants pulled to module scope for legibility inside hot RHS.
F = Para.F
R = Para.R
T_ref = Para.T_ref


# ---------------------------------------------------------------------------
# Voltage / RHS helpers
# ---------------------------------------------------------------------------
def _electrolyte_conc_potential(c_e):
    """Scott Moura SPMeT diffusion-potential split (negative & positive).

    A small floor is applied to ``c_e`` so that brief integrator overshoots
    that produce slightly negative values do not poison the log.
    """
    coeff = 2 * R * T_ref / F * (1 - Para.t_plus)
    safe = np.maximum(c_e, 1.0)
    phi_n = coeff * np.log(safe[:15] / 1000.0)
    phi_p = coeff * np.log(safe[21:36] / 1000.0)
    return phi_n, phi_p


def voltage(u, I):
    c_n = u[0:10]
    c_p = u[10:20]
    c_e = u[20:56]
    L_SEI = u[56]
    eps_neg = u[57]
    eps_act_neg = u[58]

    a_neg = Para.a_neg_BOL * (eps_act_neg / Para.eps_act_neg_BOL)
    Brug_neg = eps_neg ** 1.5

    OCP_pos = Para.Positive_OCP(1.5 * c_p[-1] - 0.5 * c_p[-2])
    OCP_neg = Para.Negative_OCP(1.5 * c_n[-1] - 0.5 * c_n[-2])
    OCP = OCP_pos - OCP_neg

    phi_n, phi_p = _electrolyte_conc_potential(c_e)
    Phi_e_conc = phi_p.mean() - phi_n.mean()

    safe_e = np.maximum(c_e, 1.0)
    sigma_e_neg = np.array([Para.electrolyte_conductivity(c) for c in safe_e[:15]])
    sigma_e_pos = np.array([Para.electrolyte_conductivity(c) for c in safe_e[21:36]])

    Phi_e_ohmic_n = np.mean((I * Para.L_neg / Para.A_electrode / Brug_neg / 6.0) / sigma_e_neg)
    Phi_e_ohmic_p = -np.mean((I * Para.L_pos / Para.A_electrode / Para.Brug_pos / 6.0) / sigma_e_pos)
    Phi_e_ohmic = Phi_e_ohmic_p - Phi_e_ohmic_n

    Phi_s = -(I / Para.A_electrode) * Para.L_neg / Para.sigma_neg \
            - (I / Para.A_electrode) * Para.L_pos / Para.sigma_pos
    Phi_SEI = -I * L_SEI * Para.R_SEI

    jn = I / Para.A_electrode / Para.L_neg
    jp = -I / Para.A_electrode / Para.L_pos

    c_e_avg_n = np.mean(np.sqrt(safe_e[:15])) ** 2
    j0_n = Para.Negative_exchange_current_density(
        c_e_avg_n, 1.5 * c_n[-1] - 0.5 * c_n[-2], Para.c_neg_max, T_ref)
    eta_n = 2 * R * T_ref / F * np.arcsinh(jn / j0_n / a_neg)

    c_e_avg_p = np.mean(np.sqrt(safe_e[21:36])) ** 2
    j0_p = Para.Positive_exchange_current_density(
        c_e_avg_p, 1.5 * c_p[-1] - 0.5 * c_p[-2], Para.c_pos_max, T_ref)
    eta_p = 2 * R * T_ref / F * np.arcsinh(jp / j0_p / Para.a_pos_BOL)

    return OCP + Phi_e_conc + Phi_e_ohmic + Phi_s + Phi_SEI + (eta_p - eta_n)


def make_rhs(model_kind, sei_phys, ude_kappa, T_exp,
             current_func, t_step_start, voltage_hold, V_target,
             tau_alg=2.0):
    """Return a closure ``rhs(t, u) -> du/dt`` for the current step."""
    alpha_SEI, k_SEI, D_SEI, U_SEI, beta = sei_phys
    kappa1, kappa2 = ude_kappa
    gamma = 24 * 3600.0

    Neg_M = Para.NegParticle_StateMatrix
    Pos_M = Para.PosParticle_StateMatrix
    Grad_M = Para.Electrolyte_conc_interfaceGrad
    Mean_M = Para.Mean_of_node_at_interface
    Div_M = Para.Electrolyte_divergence_matrix
    elec_src = Para.electrolyte_source

    eps_node_template_sep = Para.eps_sep_BOL * np.ones(Para.N_FVM_sep)
    eps_node_template_pos = Para.eps_pos_BOL * np.ones(Para.N_FVM_pos_elec)

    def rhs(t, u):
        c_n = u[0:10]
        c_p = u[10:20]
        c_e = u[20:56]
        L_SEI = u[56]
        eps_neg = u[57]
        eps_act_neg = u[58]
        I_corr = u[59]

        I = current_func(t - t_step_start) - I_corr
        a_neg = Para.a_neg_BOL * (eps_act_neg / Para.eps_act_neg_BOL)

        # SEI calculations
        c_n_surf = 1.5 * c_n[-1] - 0.5 * c_n[-2]
        c_p_surf = 1.5 * c_p[-1] - 0.5 * c_p[-2]
        OCP_n = Para.Negative_OCP(c_n_surf)
        OCP_p = Para.Positive_OCP(c_p_surf)

        eta_SEI_ohmic = I * Para.R_SEI * L_SEI
        c_e_avg_n = np.mean(np.sqrt(np.maximum(c_e[:15], 1.0))) ** 2
        j0_n = Para.Negative_exchange_current_density(
            c_e_avg_n, c_n_surf, Para.c_neg_max, T_ref)
        j_n = I / Para.A_electrode / Para.L_neg
        eta_SEI_reaction = 2 * R * T_ref / F * np.arcsinh(j_n / j0_n / a_neg)

        eta_n_bar = OCP_n + eta_SEI_reaction + eta_SEI_ohmic
        eta_p_bar = OCP_p

        if model_kind == 'Physics':
            sei_exp_term = np.exp(-alpha_SEI * (F / R / T_exp) *
                                  (eta_n_bar - U_SEI))
            j_SEI = (-a_neg * F * Para.c0_SEI * k_SEI * sei_exp_term /
                     (1 + k_SEI * L_SEI * sei_exp_term / D_SEI))
            dL_SEI = -Para.Molar_weight_SEI / Para.rho_SEI / F / 2 / Para.a_neg_BOL * j_SEI
            d_eps_neg = -1.0 / a_neg * dL_SEI
            d_eps_act_neg = beta * j_SEI
        else:  # UDE
            psi1 = max(0.16 - eta_n_bar, 0.0)                           # ReLU
            psi2 = np.exp(27.0 * (eta_p_bar - 4.12))
            x = 0.16 - eta_n_bar
            psi3 = np.exp(10.0 * (x if x >= 0 else 0.015 * x))           # leaky-relu
            psi4 = eta_p_bar

            dL_SEI = (1e-9 * (1.0 / gamma) *
                      Para.NN_SEI_forward(np.array([psi1, psi2])) *
                      kappa1 / (1e9 * L_SEI))
            d_eps_act_neg = (-(1.0 / 100.0) * (1.0 / gamma) *
                             Para.NN_eps_forward(np.array([psi3, psi4])) *
                             kappa2 / (1.0 + 100.0 * (Para.eps_act_neg_BOL - eps_act_neg)))
            j_SEI = -a_neg * 2 * F * Para.rho_SEI / Para.Molar_weight_SEI * dL_SEI
            d_eps_neg = -1.0 / a_neg * dL_SEI

        # Particle diffusion
        dc_n = Neg_M @ c_n
        dc_p = Pos_M @ c_p

        j_p = -I / Para.A_electrode / Para.L_pos
        j_n_curr = I / Para.A_electrode / Para.L_neg
        N_p_surf = j_p / F / Para.a_pos_BOL
        N_n_surf = (j_n_curr - j_SEI) / F / a_neg

        dc_p[-1] = dc_p[-1] - Para.AbyV_outer_cntrl_vol_pos * N_p_surf
        dc_n[-1] = dc_n[-1] - Para.AbyV_outer_cntrl_vol_neg * N_n_surf

        # Electrolyte diffusion (porosity in negative tracks degradation)
        eps_node = np.concatenate([
            eps_neg * np.ones(Para.N_FVM_neg_elec),
            eps_node_template_sep,
            eps_node_template_pos,
        ])
        Brug_node = eps_node ** 1.5
        grad_ce = Grad_M @ c_e
        brug_iface = Mean_M @ Brug_node
        D_e_node = (8.794e-11 * (c_e / 1000.0) ** 2
                    - 3.972e-10 * (c_e / 1000.0) + 4.862e-10)
        D_e_iface = Mean_M @ D_e_node
        flux = D_e_iface * grad_ce * brug_iface
        dc_e = (Div_M @ flux + elec_src * I) / eps_node

        # Algebraic / penalty equation for current correction.
        # Outside voltage hold, ``u[59]`` is reset to 0 between steps and we
        # leave its derivative at zero, which avoids gratuitous stiffness
        # during long storage steps.
        if voltage_hold:
            V = voltage(u, I)
            dI_corr = (V_target - V) / tau_alg
        else:
            dI_corr = 0.0

        du = np.empty_like(u)
        du[0:10] = dc_n
        du[10:20] = dc_p
        du[20:56] = dc_e
        du[56] = dL_SEI
        du[57] = d_eps_neg
        du[58] = d_eps_act_neg
        du[59] = dI_corr
        return du

    return rhs


# ---------------------------------------------------------------------------
# Step termination event factory
# ---------------------------------------------------------------------------
def make_event(term_kind, threshold, current_func, t_step_start):
    if term_kind == 'Time':
        return None  # rely on tspan end

    if term_kind == 'Voltage':
        def event(t, u):
            I = current_func(t - t_step_start) - u[59]
            return voltage(u, I) - threshold
        event.terminal = True
        event.direction = 0
        return event

    if term_kind == 'Abs_Current':
        def event(t, u):
            return abs(current_func(t - t_step_start) - u[59]) - threshold
        event.terminal = True
        event.direction = 0
        return event

    raise ValueError(f'Unknown termination kind: {term_kind}')


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run(SOC=85, Temperature=45, Model='UDE', max_rpts=None, verbose=True,
        kappa_override=None):
    """Run the SPMe + SEI calendar-ageing simulation.

    Parameters
    ----------
    kappa_override : tuple(float, float) or None
        If provided, overrides the default (kappa1, kappa2) for the UDE model,
        enabling training loops to inject candidate parameters without modifying
        the module-level defaults.
    """
    if Model not in ('Physics', 'UDE'):
        raise ValueError("Model must be 'Physics' or 'UDE'")

    dates, exp_capacity, exp_capacity_std = Exp.get_capacity_data(Temperature, SOC)
    lam_dates, lam_mean, lam_std = Exp.get_LAM_data(Temperature, SOC)

    if max_rpts is not None and max_rpts < len(dates):
        dates = dates[:max_rpts]
        exp_capacity = exp_capacity[:max_rpts]
        exp_capacity_std = exp_capacity_std[:max_rpts]
        lam_keep = lam_dates <= dates[-1]
        lam_dates = lam_dates[lam_keep]
        lam_mean = lam_mean[lam_keep]
        lam_std = lam_std[lam_keep] if lam_std.ndim == 1 else lam_std[:, lam_keep]

    experiment = Exp.calendar_ageing_exp_from_dates(
        dates, SOC / 100.0, exp_capacity)

    # Temperature-dependent parameters
    if Temperature == 45:
        sei_phys = (0.5335, 7.32e-16, 1.16e-21, 0.4, 4.526e-10)  # alpha, k, D, U, beta
        ude_kappa = (1.0, 1.0)
    elif Temperature == 25:
        sei_phys = (0.5, 1.098e-16, 1.856e-22, 0.4, 12.22e-10)
        ude_kappa = (0.17, 0.46)
    elif Temperature == 0:
        sei_phys = (0.385, 2.928e-16, 8.12e-23, 0.4, 13.85e-10)
        ude_kappa = (0.19, 0.26)
    else:
        raise ValueError(f'Unsupported temperature {Temperature}')

    if kappa_override is not None:
        ude_kappa = tuple(kappa_override)

    T_exp = Temperature + 273.15

    # Initial state
    u0 = np.concatenate([
        Para.c_n_ini * np.ones(Para.N_FVM_particle),
        Para.c_p_ini * np.ones(Para.N_FVM_particle),
        Para.c_e_ref * np.ones(Para.N_FVM_electrode),
        [Para.L_SEI_ini, Para.eps_neg_BOL, Para.eps_act_neg_BOL, 0.0],
    ])

    Q_RPT = np.zeros(len(dates))
    cycle_num = 0
    t_curr = 0.0
    t_step_start = 0.0
    u = u0.copy()

    # Save full trajectory (time + selected scalar states) for post-processing.
    history_t = [t_curr]
    history_L_SEI = [u[56]]
    history_eps_act = [u[58]]

    t_start_wall = time.time()
    for step_idx, step in enumerate(experiment):
        is_v_hold, I_func, term_kind, threshold, duration, label, dtmax = step

        rhs = make_rhs(Model, sei_phys, ude_kappa, T_exp,
                       I_func, t_curr, is_v_hold, threshold)
        event = make_event(term_kind, threshold, I_func, t_curr)

        # Julia's time-terminated steps stop at ``threshold``; the 5th tuple
        # element is just an outer safety bound. Voltage / current events use
        # the full ``duration`` buffer.
        tspan_end = t_curr + threshold if term_kind == 'Time' else t_curr + duration

        sol = solve_ivp(
            rhs,
            (t_curr, tspan_end),
            u,
            method='BDF',
            events=event,
            max_step=min(dtmax, tspan_end - t_curr),
            rtol=1e-6,
            atol=1e-8,
            dense_output=False,
        )
        if not sol.success:
            raise RuntimeError(f'Step {step_idx} failed: {sol.message}')

        if event is not None and len(sol.t_events[0]) > 0:
            t_end = float(sol.t_events[0][0])
            u_end = sol.y_events[0][0].copy()
        else:
            t_end = float(sol.t[-1])
            u_end = sol.y[:, -1].copy()

        if label == 'RPT-Capacity':
            cycle_num += 1
            I_last = I_func(t_end - t_curr) - u_end[59]
            if cycle_num <= len(Q_RPT):
                Q_RPT[cycle_num - 1] = (t_end - t_curr) * I_last / 3600.0
            if verbose:
                print(f'  RPT {cycle_num}/{len(dates)}: '
                      f'Q = {Q_RPT[cycle_num - 1]:.4f} Ah, '
                      f't = {t_end / 86400.0:.1f} d')

        # Reset algebraic current correction at step boundaries
        u_end[59] = 0.0

        # Save lightweight history
        history_t.append(t_end)
        history_L_SEI.append(u_end[56])
        history_eps_act.append(u_end[58])

        u = u_end
        t_step_start = t_curr
        t_curr = t_end

    wall = time.time() - t_start_wall
    if verbose:
        print(f'Simulation complete in {wall:.1f} s '
              f'({len(experiment)} steps, {cycle_num} RPTs)')

    # Trajectory of eps_act_neg vs. time (for LAM at requested days)
    history_t = np.array(history_t)
    history_eps_act = np.array(history_eps_act)
    eps_act_at = np.interp(lam_dates * 86400.0, history_t, history_eps_act)
    LAM_sim = 100.0 - 100.0 * eps_act_at / Para.eps_act_neg_BOL

    Q_used = Q_RPT[:cycle_num]
    dates_used = dates[:cycle_num]
    if cycle_num == 0:
        raise RuntimeError('No RPTs completed.')
    Q_norm = Q_used / Q_used[0] * 100.0
    Exp_cap_norm = exp_capacity[:cycle_num] / exp_capacity[0] * 100.0
    Exp_cap_std_norm = exp_capacity_std[:cycle_num] / exp_capacity[0] * 100.0

    rmse_cap = float(np.sqrt(np.mean((Exp_cap_norm - Q_norm) ** 2)))
    rmse_lam = float(np.sqrt(np.mean((lam_mean - LAM_sim) ** 2)))
    if verbose:
        print(f'RMSE capacity = {rmse_cap:.3f} %, RMSE LAM = {rmse_lam:.3f} %')

    return dict(
        dates=dates_used, Q_norm=Q_norm,
        Exp_cap_norm=Exp_cap_norm, Exp_cap_std_norm=Exp_cap_std_norm,
        lam_dates=lam_dates, LAM_sim=LAM_sim,
        lam_mean=lam_mean, lam_std=lam_std,
        rmse_cap=rmse_cap, rmse_lam=rmse_lam,
        Model=Model,
    )


def plot_results(res, output='results.png'):
    color = 'green' if res['Model'] == 'UDE' else 'red'
    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    ax = axes[0]
    ax.scatter(res['dates'], res['Q_norm'], marker='D', color=color, s=50,
               label=res['Model'])
    ax.errorbar(res['dates'], res['Exp_cap_norm'], yerr=res['Exp_cap_std_norm'],
                fmt='o', color='black', label='Experiment')
    ax.set_xlabel('Days of storage')
    ax.set_ylabel('Relative Capacity (%)')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(res['lam_dates'], res['LAM_sim'], marker='D', color=color, s=50,
               label=res['Model'])
    yerr = res['lam_std']
    if yerr.ndim > 1:
        yerr = yerr[0]
    ax.errorbar(res['lam_dates'], res['lam_mean'], yerr=yerr,
                fmt='o', color='black', label='Experiment')
    ax.set_xlabel('Days of storage')
    ax.set_ylabel('LAM (%)')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=120)
    print(f'Saved figure to {output}')


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--soc', type=int, default=85)
    p.add_argument('--temperature', type=int, default=45,
                   choices=[0, 25, 45])
    p.add_argument('--model', choices=['Physics', 'UDE'], default='UDE')
    p.add_argument('--max-rpts', type=int, default=None,
                   help='Truncate the experiment to this many RPTs')
    p.add_argument('--output', default='results.png')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    res = run(SOC=args.soc, Temperature=args.temperature,
              Model=args.model, max_rpts=args.max_rpts)
    plot_results(res, output=args.output)
