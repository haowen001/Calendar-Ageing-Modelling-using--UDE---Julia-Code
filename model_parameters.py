"""Python port of Model_parameters.jl.

All electrochemical, geometric, FVM and SEI parameters used by the SPMe
calendar-ageing model, plus the trained Lux neural-network parameter vectors.
"""

import numpy as np
from scipy.sparse import diags, csr_matrix

# ----------------------------------------------------------------------------
# Physical / global constants
# ----------------------------------------------------------------------------
T_ref = 298.15           # Reference Temperature [K]
F = 96485.3329           # Faraday constant [C/mol]
R = 8.314                # Ideal gas constant [J/(mol K)]

# ----------------------------------------------------------------------------
# Electrode parameters
# ----------------------------------------------------------------------------
L_pos = 75.6e-6
L_neg = 85.2e-6
L_sep = 12.0e-6
R_pos = 5.22e-6
R_neg = 5.86e-6
A_electrode = 0.1

eps_pos_BOL = 0.335
eps_neg_BOL = 0.25
eps_sep_BOL = 0.47
Brug_pos = eps_pos_BOL ** 1.5
Brug_sep = eps_sep_BOL ** 1.5

a_neg_BOL = 3.84e5
a_pos_BOL = 3.82e5
eps_act_neg_BOL = 0.75
eps_act_pos_BOL = 0.665

N_FVM_pos_elec = 15
N_FVM_neg_elec = 15
N_FVM_sep = 6
N_FVM_electrode = N_FVM_pos_elec + N_FVM_neg_elec + N_FVM_sep
N_FVM_particle = 10

AbyV_outer_cntrl_vol_neg = 1.889097389e6
AbyV_outer_cntrl_vol_pos = 2.120710862e6

c_neg_max = 33133.0
c_pos_max = 63104.0
c_e_ref = 1000.0

t_plus = 0.2594
sigma_neg = 0.18
sigma_pos = 215.0

SOC_ini = 1.0
theta_n_SOC0, theta_n_SOC100 = 0.02637, 0.910612
theta_p_SOC0, theta_p_SOC100 = 0.8539736, 0.263848
c_n_ini = c_neg_max * (theta_n_SOC0 + (theta_n_SOC100 - theta_n_SOC0) * SOC_ini)
c_p_ini = c_pos_max * (theta_p_SOC0 + (theta_p_SOC100 - theta_p_SOC0) * SOC_ini)
L_SEI_ini = 1e-9

R_SEI = 6.1e4
c0_SEI = 4541.0
Molar_weight_SEI = 0.162
rho_SEI = 1690.0


# ----------------------------------------------------------------------------
# Particle diffusion state matrices (Tridiagonal, FVM, 10 elements)
# ----------------------------------------------------------------------------
_PosParticle_LDiag = np.array([
    0.006291326148638871, 0.009271428008520441, 0.010712258036871591,
    0.011551287354877928, 0.012098704131997828, 0.012483576294936972,
    0.012768786207000784, 0.012988544306867347, 0.013163032938296088,
])
_PosParticle_Diag = np.array([
    -0.044039283040472096, -0.03145663074319435, -0.030132141027691434,
    -0.02975627232464331, -0.02960017384687469, -0.029520838082074702,
    -0.029475110696378964, -0.029446384518185483, -0.029427170695246332,
    -0.013163032938296088,
])
_PosParticle_UDiag = np.array([
    0.044039283040472096, 0.025165304594555484, 0.020860713019170994,
    0.019044014287771718, 0.018048886491996763, 0.01742213395007687,
    0.01699153440144199, 0.016677598311184698, 0.016438626388378987,
])
PosParticle_StateMatrix = diags(
    [_PosParticle_LDiag, _PosParticle_Diag, _PosParticle_UDiag],
    offsets=[-1, 0, 1], format='csr',
)
NegParticle_StateMatrix = 3.9278145348227707 * PosParticle_StateMatrix


# ----------------------------------------------------------------------------
# Electrolyte FVM matrices
# ----------------------------------------------------------------------------
dx_neg = L_neg / N_FVM_neg_elec
dx_pos = L_pos / N_FVM_pos_elec
dx_sep = L_sep / N_FVM_sep
dx_neg_sep = (dx_neg + dx_sep) / 2
dx_pos_sep = (dx_pos + dx_sep) / 2

_e_vec = 1.0 / np.concatenate([
    dx_neg * np.ones(N_FVM_neg_elec - 1),
    [dx_neg_sep],
    dx_sep * np.ones(N_FVM_sep - 1),
    [dx_pos_sep],
    dx_pos * np.ones(N_FVM_pos_elec - 1),
])
# Rectangular sparse matrix (N_FVM_electrode-1) x N_FVM_electrode
_n = N_FVM_electrode
_rows = np.repeat(np.arange(_n - 1), 2)
_cols = np.empty(2 * (_n - 1), dtype=int)
_cols[0::2] = np.arange(_n - 1)        # column j (negative diagonal contribution)
_cols[1::2] = np.arange(1, _n)         # column j+1 (positive)
_data = np.empty(2 * (_n - 1))
_data[0::2] = -_e_vec
_data[1::2] = _e_vec
Electrolyte_conc_interfaceGrad = csr_matrix(
    (_data, (_rows, _cols)), shape=(_n - 1, _n)
)

_left_weight = 0.5 * np.ones(N_FVM_electrode - 1)
_right_weight = 0.5 * np.ones(N_FVM_electrode - 1)
_left_weight[N_FVM_neg_elec - 1] = dx_neg / (dx_neg + dx_sep)   # Julia idx N_FVM_neg_elec
_right_weight[N_FVM_neg_elec - 1] = dx_sep / (dx_neg + dx_sep)
_left_weight[N_FVM_neg_elec + N_FVM_sep - 1] = dx_sep / (dx_sep + dx_pos)
_right_weight[N_FVM_neg_elec + N_FVM_sep - 1] = dx_pos / (dx_sep + dx_pos)
_data2 = np.empty(2 * (_n - 1))
_data2[0::2] = _left_weight
_data2[1::2] = _right_weight
Mean_of_node_at_interface = csr_matrix(
    (_data2, (_rows, _cols)), shape=(_n - 1, _n)
)

# Divergence matrix: shape (N_FVM_electrode) x (N_FVM_electrode-1)
_left_weight1 = 1.0 / np.concatenate([
    dx_neg * np.ones(N_FVM_neg_elec - 1),
    dx_sep * np.ones(N_FVM_sep),
    dx_pos * np.ones(N_FVM_pos_elec),
])
_right_weight1 = 1.0 / np.concatenate([
    dx_neg * np.ones(N_FVM_neg_elec),
    dx_sep * np.ones(N_FVM_sep),
    dx_pos * np.ones(N_FVM_pos_elec - 1),
])
# spdiagm(N, N-1, 0=>right_weight1, -1=>-left_weight1) (Julia indexing)
# row i, col i  : right_weight1[i] for i in 0..N-2
# row i, col i-1: -left_weight1[i-1] for i in 1..N-1
N = N_FVM_electrode
rows_d = []
cols_d = []
data_d = []
for i in range(N - 1):
    rows_d.append(i)
    cols_d.append(i)
    data_d.append(_right_weight1[i])
for i in range(1, N):
    rows_d.append(i)
    cols_d.append(i - 1)
    data_d.append(-_left_weight1[i - 1])
Electrolyte_divergence_matrix = csr_matrix(
    (np.asarray(data_d), (np.asarray(rows_d), np.asarray(cols_d))),
    shape=(N, N - 1),
)

Source_neg = (1 - t_plus) / (F * L_neg * A_electrode)
Source_sep = 0.0
Source_pos = -(1 - t_plus) / (F * A_electrode * L_pos)
electrolyte_source = np.concatenate([
    Source_neg * np.ones(N_FVM_neg_elec),
    Source_sep * np.ones(N_FVM_sep),
    Source_pos * np.ones(N_FVM_pos_elec),
])


# ----------------------------------------------------------------------------
# Open-circuit potentials and exchange current densities
# ----------------------------------------------------------------------------
def Positive_OCP(surf_conc):
    sto = surf_conc / c_pos_max
    return (
        -0.8090 * sto
        + 4.4875
        - 0.0428 * np.tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * np.tanh(15.7890 * (sto - 0.3117))
        + 17.5842 * np.tanh(15.9308 * (sto - 0.3120))
    )


def Negative_OCP(surf_conc):
    sto = surf_conc / c_neg_max
    return (
        1.9793 * np.exp(-39.3631 * sto)
        + 0.2482
        - 0.0909 * np.tanh(29.8538 * (sto - 0.1234))
        - 0.04478 * np.tanh(14.9159 * (sto - 0.2769))
        - 0.0205 * np.tanh(30.4444 * (sto - 0.6103))
    )


def Negative_exchange_current_density(c_e, c_s_surf, c_s_max, T):
    m_ref = 2.1e-5 * 0.3
    E_r = 35000.0
    arrhenius = np.exp(E_r / 8.314 * (1.0 / 298.15 - 1.0 / T))
    if c_s_surf < 0 or (c_s_max - c_s_surf) < 0:
        return -(m_ref * arrhenius * c_e ** 0.5 *
                 abs(c_s_surf) ** 0.5 * abs(c_s_max - c_s_surf) ** 0.5)
    return (m_ref * arrhenius * c_e ** 0.5 *
            c_s_surf ** 0.5 * (c_s_max - c_s_surf) ** 0.5)


def Positive_exchange_current_density(c_e, c_s_surf, c_s_max, T):
    m_ref = 3.42e-6 * 0.3
    E_r = 17800.0
    arrhenius = np.exp(E_r / 8.314 * (1.0 / 298.15 - 1.0 / T))
    if c_s_surf < 0 or (c_s_max - c_s_surf) < 0:
        return -(m_ref * arrhenius * c_e ** 0.5 *
                 abs(c_s_surf) ** 0.5 * abs(c_s_max - c_s_surf) ** 0.5)
    return (m_ref * arrhenius * c_e ** 0.5 *
            c_s_surf ** 0.5 * (c_s_max - c_s_surf) ** 0.5)


def electrolyte_conductivity(ce):
    return 1.297e-10 * ce ** 3 - 7.937e-5 * ce ** 1.5 + 3.329 * ce


# ----------------------------------------------------------------------------
# Trained neural-network parameter vectors (151 floats each).
# Layout per Lux Chain([Dense(2,10), Dense(10,10), Dense(10,1)]):
#   [0:20]   layer1 weights (column-major 10x2)
#   [20:30]  layer1 bias
#   [30:130] layer2 weights (column-major 10x10)
#   [130:140] layer2 bias
#   [140:150] layer3 weights (column-major 1x10)
#   [150:151] layer3 bias
# ----------------------------------------------------------------------------
NN_SEI_parameters = np.array([
    -0.60477513, -1.4223812, -1.9463432, 1.2615505, -1.0674226, -0.49557072,
    1.0772254, 0.16943657, 1.2391795, -0.7318138, -0.33428714, -0.30335915,
    0.21019438, -0.98710126, 0.20369251, -0.07983856, -0.5079126, 0.06553077,
    0.14329456, 0.1499584, -0.19964486, -0.27076286, -0.7546108, 0.57861954,
    -0.62474096, 0.29337394, -0.26975965, -0.04097848, 0.19088614, -0.40307194,
    -0.08983634, 0.44503197, 0.73680454, -0.02104578, -0.34765416, -0.3361742,
    -0.28648275, -0.23142889, -0.24647379, 0.27002525, -0.4739367, 0.29854122,
    0.51182306, 0.27766714, -0.2928416, 0.5229706, -0.53903323, -0.28664652,
    0.3326829, 0.024485804, -0.52752787, 0.57316685, -1.2521281, 0.9732371,
    -0.6036009, 0.06242689, 0.17228335, -0.2199403, 0.12000578, 0.9813996,
    0.038461227, -0.050157327, 1.2965958, -0.067490146, -0.21412455,
    -0.83767426, 0.6673814, 0.3364494, -0.024463827, 0.43399918, -0.3390732,
    0.7110837, -1.0971395, 1.0076795, -0.80497605, 0.32683915, -0.05166485,
    0.19983414, -0.29613578, 0.6551344, -0.6411093, 0.74382144, -0.19749607,
    -0.16417795, -0.88852596, -0.30207637, -0.6440141, -0.29514316, 0.33611032,
    0.29858583, -0.12703069, -0.02621218, 0.5058579, 0.70943207, 0.23768118,
    -0.65298235, 0.105020516, -0.28916904, -0.035281904, 0.2844425, 0.67168105,
    -0.45174876, -0.1915499, -0.70602304, -0.1214413, -0.08233488, 0.31870523,
    0.8495964, -0.73499477, -0.91755766, 0.75990367, -0.5644614, -0.2655123,
    -0.21955533, 0.6083121, 0.13302465, 0.85968244, 0.14392613, -0.2541667,
    0.054571975, -0.8377351, 0.8365496, -0.80237365, 0.80164635, -0.3780746,
    0.2017563, -0.08727326, -0.83301795, 0.51981133, 0.40772763, -0.19167162,
    -0.057614625, 0.33620682, -0.20916627, 0.20712937, -0.20479766,
    0.0009777294, 0.19825661, -0.19775464, -0.20917015, 0.2819379, -0.4639101,
    0.4234424, -1.0033692, 1.1036776, -0.40092075, 0.8623977, 0.67802763,
    -0.15286118, -0.61830676, 0.20663543,
], dtype=np.float64)

NN_eps_parameters = np.array([
    0.42191094, 0.20118989, 0.35413122, -0.5432369, -0.4986403, -0.18667749,
    0.21885203, -0.7306449, -0.10493427, -0.19297396, 0.7157549, -0.402475,
    -0.16469051, 0.2889012, 0.14279369, 0.14241295, -0.4837099, 0.10582558,
    -0.375999, 0.06123723, 0.088364415, 0.015491825, -0.008030636, 0.018120088,
    0.008857643, 0.007827843, 0.021841424, 0.019537583, -0.031081112,
    -0.004776785, 0.2026659, -0.00079054444, -0.06101859, 0.08544222,
    0.37282178, 0.3419867, 0.27998963, -0.50469273, 0.1164694, -0.23116423,
    -0.2504506, 0.33864278, -0.38898435, -0.41606963, 0.38243607, -0.19415867,
    -0.14037709, -0.06251307, -0.19111623, -0.047402292, -0.35934597,
    0.47374278, 0.5118375, 0.2369531, -0.32861808, -0.44555452, 0.3787947,
    0.19961663, 0.1227025, 0.40261543, -0.3260156, -0.4004237, -0.2672989,
    0.12357884, -0.3896626, -0.36983708, -0.379496, -0.07166313, -0.07896875,
    -0.19958979, -0.13647936, -0.117574014, 0.32616362, 0.2873505, -0.49535987,
    0.47174898, -0.2586965, -0.13681568, -0.35830438, 0.14955145, 0.53756326,
    -0.06313884, -0.002839236, -0.21031426, -0.081732005, 0.04186685,
    -0.04059988, -0.43278596, 0.1557861, 0.34416404, 0.17957051, -0.30967146,
    0.38346875, -0.060320266, 0.25736454, 0.3502868, -0.26316535, -0.014941,
    0.3835583, 0.22380139, -0.57189095, -0.12206105, 0.50239116, 0.155497,
    -0.005381463, 0.073775314, -0.03556286, 0.09926961, -0.115919024,
    -0.32975292, -0.1506773, 0.1824003, 0.16473849, 0.017518608, -0.1702015,
    0.220345, 0.39771873, 0.15599634, -0.11521477, 0.41497606, -0.7119716,
    0.036377236, -0.029655773, 0.10119009, -0.3541841, 0.33646864, 0.10629044,
    0.113817684, 0.49155775, 0.33287266, -0.018964266, -0.015657067,
    0.016849961, -0.0075436737, 0.0062382533, 0.02493839, -0.0027739727,
    -0.015016735, 0.0059041437, -0.011554108, 0.4097754, 0.5904527,
    -0.28207374, 0.67585266, -0.59707284, -0.009153091, 0.29546967, 0.33302197,
    -0.23176794, 0.05792673, -0.010098953,
], dtype=np.float64)


def _unpack_lux_chain(p):
    """Decode a 151-element Lux Chain(Dense(2,10),Dense(10,10),Dense(10,1)).

    Lux/Julia stores Dense weights as (out, in) in column-major layout, so
    the flattened vector iterates over input dim slowest. ``order='F'``
    reproduces that exactly.
    """
    W1 = p[0:20].reshape((10, 2), order='F')
    b1 = p[20:30]
    W2 = p[30:130].reshape((10, 10), order='F')
    b2 = p[130:140]
    W3 = p[140:150].reshape((1, 10), order='F')
    b3 = p[150:151]
    return W1, b1, W2, b2, W3, b3


_NN_SEI = _unpack_lux_chain(NN_SEI_parameters)
_NN_eps = _unpack_lux_chain(NN_eps_parameters)


def NN_SEI_forward(x):
    W1, b1, W2, b2, W3, b3 = _NN_SEI
    h = np.tanh(W1 @ x + b1)
    h = np.tanh(W2 @ h + b2)
    y = (W3 @ h + b3) ** 2  # square activation
    return y[0]


def NN_eps_forward(x):
    W1, b1, W2, b2, W3, b3 = _NN_eps
    h = np.tanh(W1 @ x + b1)
    h = np.tanh(W2 @ h + b2)
    y = (W3 @ h + b3) ** 2
    return y[0]
