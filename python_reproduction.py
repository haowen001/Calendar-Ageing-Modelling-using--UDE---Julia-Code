"""Python reproduction of the Julia calendar-ageing workflow.

This script mirrors the data-loading, experiment construction, and baseline
physics/UDE parameter selection done in `Main.jl`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Step:
    voltage_hold: bool
    current_fn: Callable[[float], float]
    stop_type: str
    stop_value: float
    timeout: float
    label: Optional[str]
    sample_time: float


def _read_mat(path: Path) -> dict:
    return loadmat(path, struct_as_record=False, squeeze_me=True)


def _extract_group(data: dict, temp: int, soc: int):
    temp_key = f"Temperature_{temp}"
    soc_key = f"SOC_{soc}"
    if temp_key not in data:
        raise KeyError(f"Missing {temp_key} in MAT data")
    temperature_group = data[temp_key]
    if not hasattr(temperature_group, soc_key):
        raise KeyError(f"Missing {soc_key} under {temp_key}")
    return getattr(temperature_group, soc_key)


def get_capacity_data(temp: int, soc: int):
    data = _read_mat(ROOT / "RPT_analysis_data.mat")
    group = _extract_group(data, temp, soc)
    days = np.asarray(group.Days, dtype=float)
    mean = np.asarray(group.Capacity_Mean, dtype=float)
    std = np.asarray(group.Capacity_Std, dtype=float)
    if temp == 45 and soc == 90:
        return days[:-4], mean[:-4], std[:-4]
    return days, mean, std


def get_lam_data(temp: int, soc: int):
    data = _read_mat(ROOT / "RPTx_analysis_data.mat")
    group = _extract_group(data, temp, soc)
    days = np.asarray(group.LAM_days, dtype=float)
    mean = np.asarray(group.LAM_mean, dtype=float)
    std = np.asarray(group.LAM_std, dtype=float)
    if temp == 45 and soc == 90:
        return days[:-1], mean, std
    return days, mean, std


def storage_step(storage_soc: float, storage_days: float, capacity: float) -> list[Step]:
    time_to_storage_soc = (capacity / 5 * 3) * 3600.0 * (1.0 - storage_soc)
    storage_duration = storage_days * 24 * 3600.0

    if storage_soc == 1.0:
        return [
            Step(False, lambda _: 0.0, "Time", 3600.0, 5000.0, None, 600.0),
            Step(False, lambda _: -5 / 3, "Voltage", 4.2, 36000.0, None, 300.0),
            Step(True, lambda _: -5 / 3, "Abs_Current", 0.25, 50000.0, None, 180.0),
            Step(False, lambda _: 0.0, "Time", 1800.0, 50000.0, None, 400.0),
            Step(False, lambda _: 0.0, "Time", storage_duration, storage_duration * 2.0, "Storage_step", storage_duration / 5.0),
            Step(False, lambda _: 5 / 3, "Voltage", 2.5, 18000.0, None, 300.0),
        ]

    steps = [
        Step(False, lambda _: 0.0, "Time", 3600.0, 5000.0, None, 600.0),
        Step(False, lambda _: -5 / 3, "Voltage", 4.2, 36000.0, None, 300.0),
        Step(True, lambda _: -5 / 3, "Abs_Current", 0.25, 50000.0, None, 180.0),
        Step(False, lambda _: 0.0, "Time", 1800.0, 50000.0, None, 400.0),
        Step(False, lambda _: 5 / 3, "Time", time_to_storage_soc, time_to_storage_soc * 2.0, None, max(time_to_storage_soc / 5.0, 1.0)),
        Step(False, lambda _: 0.0, "Time", storage_duration, storage_duration * 2.0, "Storage_step", storage_duration / 5.0),
        Step(False, lambda _: 5 / 3, "Voltage", 2.5, 18000.0, None, 300.0),
    ]
    if storage_soc == 0.0:
        steps[4] = Step(False, lambda _: 5 / 3, "Voltage", 2.5, 18000.0, None, 300.0)
        steps = steps[:6]
    return steps


def calendar_ageing_exp_from_dates(dates: np.ndarray, storage_soc: float, capacities: np.ndarray) -> list[Step]:
    discharge = Step(False, lambda _: 5 / 3, "Voltage", 2.5, 3960.0, None, 600.0)
    rpt = [
        Step(False, lambda _: 0.0, "Time", 3600.0, 5000.0, None, 600.0),
        Step(False, lambda _: -5 / 3, "Voltage", 4.2, 36000.0, None, 300.0),
        Step(True, lambda _: -5 / 3, "Abs_Current", 0.25, 50000.0, None, 180.0),
        Step(False, lambda _: 0.0, "Time", 1800.0, 50000.0, None, 400.0),
        Step(False, lambda _: 5 / 3, "Voltage", 2.5, 18000.0, "RPT-Capacity", 300.0),
    ]
    exp = [discharge, *rpt]
    for i in range(len(dates) - 1):
        exp.extend(storage_step(storage_soc, float(dates[i + 1] - dates[i]), float(capacities[i])))
        exp.extend(rpt)
    return exp


def temperature_params(temp: int, model: str):
    if temp == 45:
        physics = dict(k_SEI=7.32e-16, alpha_SEI=0.5335, D_SEI=1.16e-21, U_SEI=0.4, beta=4.526e-10)
        ude = dict(k1=1.0, k2=1.0)
    elif temp == 25:
        physics = dict(k_SEI=1.098e-16, alpha_SEI=0.5, D_SEI=1.856e-22, U_SEI=0.4, beta=12.22e-10)
        ude = dict(k1=0.17, k2=0.46)
    elif temp == 0:
        physics = dict(k_SEI=2.928e-16, alpha_SEI=0.385, D_SEI=8.12e-23, U_SEI=0.4, beta=13.85e-10)
        ude = dict(k1=0.19, k2=0.26)
    else:
        raise ValueError("Supported temperatures are 0, 25, 45")
    return physics if model.lower() == "physics" else ude


def main(soc: int = 85, temperature: int = 45, model: str = "UDE"):
    dates, cap_mean, cap_std = get_capacity_data(temperature, soc)
    lam_days, lam_mean, lam_std = get_lam_data(temperature, soc)
    experiment = calendar_ageing_exp_from_dates(dates, soc / 100.0, cap_mean)
    params = temperature_params(temperature, model)

    print(f"Built {len(experiment)} experiment steps for SOC={soc}% at {temperature}°C")
    print(f"Using {model} params: {params}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].errorbar(dates, cap_mean, yerr=cap_std, fmt="o", capsize=3)
    ax[0].set_title("Capacity data")
    ax[0].set_xlabel("Days")
    ax[0].set_ylabel("Capacity")

    ax[1].errorbar(lam_days, lam_mean, yerr=lam_std, fmt="o", capsize=3, color="tab:orange")
    ax[1].set_title("Anode LAM data")
    ax[1].set_xlabel("Days")
    ax[1].set_ylabel("LAM")

    plt.tight_layout()
    out = ROOT / "python_reproduction_preview.png"
    plt.savefig(out, dpi=150)
    print(f"Saved preview plot to {out}")


if __name__ == "__main__":
    main()
