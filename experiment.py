"""Python port of Experiment.jl.

Loads RPT capacity / LAM data from the .mat files and constructs the list of
experiment "steps" that the DAE driver walks through.

Each step is the same 7-tuple as in the Julia version:
    (voltage_hold, current_func, termination_kind, threshold,
     duration, label, dtmax)
where:
    voltage_hold     : bool, True during a CV hold
    current_func(dt) : applied current as a function of step-local time
    termination_kind : "Time" | "Voltage" | "Abs_Current"
    threshold        : value paired with the termination kind
    duration         : safety upper bound for the step
    label            : optional tag (e.g. "RPT-Capacity") or None
    dtmax            : suggested maximum integrator step size
"""

import os
import h5py
import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))


def _read_soc_group(filename, temperature, soc):
    path = os.path.join(_HERE, filename)
    with h5py.File(path, 'r') as f:
        temp_group = f.get(f'Temperature_{temperature}')
        if temp_group is None:
            raise ValueError(f'No Temperature_{temperature} group')
        soc_group = temp_group.get(f'SOC_{soc}')
        if soc_group is None:
            raise ValueError(f'No SOC_{soc} for Temperature_{temperature}')
        out = {k: np.array(soc_group[k]).squeeze() for k in soc_group.keys()}
    return out


def get_capacity_data(temperature: int, soc: int):
    d = _read_soc_group('RPT_analysis_data.mat', temperature, soc)
    days = d['Days']
    cap_mean = d['Capacity_Mean']
    cap_std = d['Capacity_Std']
    if temperature == 45 and soc == 90:
        return days[:-4], cap_mean[:-4], cap_std[:-4]
    return days, cap_mean, cap_std


def get_LAM_data(temperature: int, soc: int):
    d = _read_soc_group('RPTx_analysis_data.mat', temperature, soc)
    lam_days = d['LAM_days']
    lam_mean = d['LAM_mean']
    lam_std = d['LAM_std']
    if temperature == 45 and soc == 90:
        return lam_days[:-1], lam_mean, lam_std
    return lam_days, lam_mean, lam_std


def _const(value):
    """Return a function that ignores its arg and returns ``value``."""
    return lambda dt, _v=value: _v


def _storage_step(storage_SOC, days_between_RPT, capacity):
    """Construct the steps that take the cell to ``storage_SOC`` and rest."""
    time_to_storage_SOC = (capacity / 5.0 * 3.0) * 3600.0 * (1.0 - storage_SOC)
    storage_dur = days_between_RPT * 24 * 3600.0
    if storage_SOC == 1.0:
        steps = [
            (False, _const(0.0), 'Time', 3600.0, 5000.0, None, 600.0),
            (False, _const(-5/3), 'Voltage', 4.2, 3600 * 10, None, 300.0),
            (True,  _const(-5/3), 'Abs_Current', 0.25, 50000.0, 4.2, 180.0),
            (False, _const(0.0), 'Time', 1800.0, 50000.0, None, 400.0),
            (False, _const(0.0), 'Time', storage_dur, storage_dur * 2.0,
             'Storage_step', storage_dur / 5.0),
            (False, _const(5/3), 'Voltage', 2.5, 3600.0 * 5.0, None, 300.0),
        ]
    else:
        steps = [
            (False, _const(0.0), 'Time', 3600.0, 5000.0, None, 600.0),
            (False, _const(-5/3), 'Voltage', 4.2, 3600 * 10, None, 300.0),
            (True,  _const(-5/3), 'Abs_Current', 0.25, 50000.0, 4.2, 180.0),
            (False, _const(0.0), 'Time', 1800.0, 50000.0, None, 400.0),
            (False, _const(5/3), 'Time', time_to_storage_SOC,
             time_to_storage_SOC * 2.0, None, time_to_storage_SOC / 5.0),
            (False, _const(0.0), 'Time', storage_dur, storage_dur * 2.0,
             'Storage_step', storage_dur / 5.0),
            (False, _const(5/3), 'Voltage', 2.5, 3600.0 * 5.0, None, 300.0),
        ]
    if storage_SOC == 0:
        steps[4] = (False, _const(5/3), 'Voltage', 2.5, 3600.0 * 5.0,
                    None, 300.0)
        steps = steps[:6]
    return steps


def calendar_ageing_exp_from_dates(dates, storage_SOC, exp_capacity):
    """Replicates ``Calendar_ageing_exp_from_dates`` from Experiment.jl."""
    discharge_to_zero = (False, _const(5/3), 'Voltage', 2.5,
                        3600 * 1.1, None, 600.0)
    rpt_cycle = [
        (False, _const(0.0), 'Time', 3600.0, 5000.0, None, 600.0),
        (False, _const(-5/3), 'Voltage', 4.2, 3600 * 10, None, 300.0),
        (True,  _const(-5/3), 'Abs_Current', 0.25, 50000, 4.2, 180.0),
        (False, _const(0.0), 'Time', 1800, 50000, None, 400.0),
        (False, _const(5/3), 'Voltage', 2.5, 3600 * 5.0,
         'RPT-Capacity', 300.0),
    ]
    ageing = [discharge_to_zero] + list(rpt_cycle)
    for i in range(len(dates) - 1):
        capacity = float(exp_capacity[i])
        days_between = float(dates[i + 1] - dates[i])
        ageing += _storage_step(storage_SOC, days_between, capacity)
        ageing += list(rpt_cycle)
    return ageing
