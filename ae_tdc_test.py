from argparse import ArgumentParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from utils import hTDCAutoEncoder, TDCLoss, create_temporal_dataset, moving_average

seed = 42

# PyTorch CPU seed
torch.manual_seed(seed)

# PyTorch GPU seed (if using CUDA)
torch.cuda.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

timestamp = "20260529_092357" 

print(timestamp)

model_path = Path("models", "TDCAE", timestamp, "trained_model.pth") #"
# Create a directory with the timestamp
folder_path = Path(f"./models/TDCAE/{timestamp}")



# Load and preprocess training data
train_file = Path("data", "BATADAL_dataset03.csv") # Replace with your training file path
test_file = Path("data", "BATADAL_testdataset.csv")    # Replace with your test file path

df_Train = pd.read_csv(train_file)
df_Test = pd.read_csv(test_file)
time = df_Train["DATETIME"]
measurements_columns = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1', 'S_PU1', 'F_PU2', 'S_PU2', 'F_PU3', 'S_PU3', 'F_PU4', 'S_PU4', 'F_PU5', 'S_PU5', 'F_PU6', 'S_PU6', 'F_PU7', 'S_PU7', 'F_PU8', 'S_PU8', 'F_PU9', 'S_PU9', 'F_PU10', 'S_PU10', 'F_PU11', 'S_PU11', 'F_V2', 'S_V2', 'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415', 'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']


df_train = df_Train[measurements_columns]
df_test = df_Test[measurements_columns]
y_test = df_Test["ATT_FLAG"]

# Split into training (90%) and validation (10%) 
split_index = int(len(df_train) * 0.9)


# smoothing with moving average
X_train_smooth = df_train.to_numpy() #moving_average(df_train.to_numpy(), 12, axis=0)
print(X_train_smooth.shape)
X_train = X_train_smooth[:split_index]
X_val = X_train_smooth[split_index:]

# Standardize the data
scaler = StandardScaler()
scaler.fit(X_train)

# Create temporal datasets for training and validation
X_train_temporal = create_temporal_dataset(X_train, scaler)
print(X_train_temporal.shape)

X_val_temporal = create_temporal_dataset(X_val, scaler)
X_test_temporal = create_temporal_dataset(df_test, scaler)  # Test set processed separately

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(X_train_temporal, torch.zeros(X_train_temporal.shape[0]))
dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

val_dataset = TensorDataset(X_val_temporal, torch.zeros(X_val_temporal.shape[0]))
dataloader_val = DataLoader(val_dataset, batch_size=1, shuffle=False)

test_dataset = TensorDataset(X_test_temporal, torch.zeros(X_test_temporal.shape[0]))
dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

neuron_count = len(measurements_columns)
latent_det = 10

latent_stat = 4

AE = hTDCAutoEncoder(neuron_count, latent_det=latent_det, latent_stat=latent_stat)

AE.load_state_dict(torch.load(model_path))

######################### LATENT EXTRACTION (ordered, all splits) ###############
# NOTE: dataloader uses shuffle=True — not usable for temporal analysis.
# Run a single ordered forward pass on each split instead.

AE.eval()
with torch.no_grad():
    _, det_latent_train, _ = AE(X_train_temporal[:, 0, :])
    _, det_latent_val,   _ = AE(X_val_temporal[:, 0, :])
    _, det_latent_test,  _ = AE(X_test_temporal[:, 0, :])

# shape: (T, latent_det=10)  — columns 0:D are z, columns D:2D are z_dot  (D=5)
det_latent_train = det_latent_train.numpy()
det_latent_val   = det_latent_val.numpy()
det_latent_test  = det_latent_test.numpy()

D = latent_det // 2   # 5 state dims, 5 derivative dims

##########################################################################################



# ============================================================
# SAFETY RECTANGLE
# ============================================================
def build_safety_rectangle(latent_space_array, mode="minmax", low_q=0.1, high_q=99.9):
    """
    Per-dimension thresholds on the latent space.

    mode : "minmax"     → use min and max per dim (tightest envelope from data)
           "percentile" → use low_q / high_q percentiles per dim

    latent_space_array : shape (T, 2D)
        First half of columns = states z
        Second half           = derivatives z_dot
    """
    if mode == "minmax":
        low  = latent_space_array.min(axis=0)
        high = latent_space_array.max(axis=0)
        label_low, label_high = "min", "max"
    elif mode == "percentile":
        low  = np.percentile(latent_space_array, low_q,  axis=0)
        high = np.percentile(latent_space_array, high_q, axis=0)
        label_low  = f"{low_q:.1f}th pct"
        label_high = f"{high_q:.1f}th pct"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "low": low, "high": high,
        "label_low": label_low, "label_high": label_high,
        "mode": mode,
    }


def in_rectangle(point, rect):
    """Check whether a single latent vector lies inside the safety rectangle."""
    return np.all((point >= rect["low"]) & (point <= rect["high"]))


def rectangle_flags(latent_space_array, rect):
    """Vectorized: returns boolean array, True where point is OUTSIDE the rectangle."""
    below = latent_space_array < rect["low"]
    above = latent_space_array > rect["high"]
    return np.any(below | above, axis=1)


# ============================================================
# PLOTTER
# ============================================================
def plot_latent_with_rectangle(
    latent_space_array,
    rect,
    slice_range=(0, 1000),
    n_latent=None,
    figsize_per_row=(12, 2.2),
):
    if n_latent is None:
        n_latent = latent_space_array.shape[1]
    assert n_latent % 2 == 0, "Latent dim must be even (z and z_dot halves)."
    D = n_latent // 2

    start, end = slice_range
    end = min(end, latent_space_array.shape[0])
    t = np.arange(start, end)
    data = latent_space_array[start:end]

    fig, axes = plt.subplots(
        D, 2,
        figsize=(figsize_per_row[0], figsize_per_row[1] * D),
        sharex=True,
        squeeze=False,
    )

    for i in range(D):
        # --- z_i ---
        ax_z = axes[i, 0]
        ax_z.plot(t, data[:, i], color="tab:blue", lw=0.9)
        ax_z.axhline(rect["low"][i],  color="tab:red", ls="--", lw=0.8,
                     label=rect["label_low"])
        ax_z.axhline(rect["high"][i], color="tab:red", ls="--", lw=0.8,
                     label=rect["label_high"])
        ax_z.set_ylabel(f"z_{i}")
        ax_z.grid(alpha=0.3)
        if i == 0:
            ax_z.set_title("States (z)")
            ax_z.legend(loc="upper right", fontsize=8)

        # --- z_dot_i ---
        j = i + D
        ax_zd = axes[i, 1]
        ax_zd.plot(t, data[:, j], color="tab:green", lw=0.9)
        ax_zd.axhline(rect["low"][j],  color="tab:red", ls="--", lw=0.8)
        ax_zd.axhline(rect["high"][j], color="tab:red", ls="--", lw=0.8)
        ax_zd.set_ylabel(f"z_dot_{i}")
        ax_zd.grid(alpha=0.3)
        if i == 0:
            ax_zd.set_title("Derivatives (z_dot)")

    axes[-1, 0].set_xlabel("time index")
    axes[-1, 1].set_xlabel("time index")
    fig.suptitle(
        f"Latent space with safety rectangle ({rect['mode']}), "
        f"samples {start}:{end}",
        y=1.0,
    )
    fig.tight_layout()
    return fig, axes


# ============================================================
# USAGE — drop in right after your training loop
# ============================================================
# latent_space_array is shape (T, 2D), with z in [:, :D] and z_dot in [:, D:]

rect = build_safety_rectangle(det_latent_train, mode="percentile")

print("Safety rectangle (min/max):")
for d in range(det_latent_train.shape[1]):
    print(f"  dim {d:2d}:  [{rect['low'][d]: .4f},  {rect['high'][d]: .4f}]")

out = rectangle_flags(det_latent_train, rect)
print(f"\nTraining points outside rectangle: {out.sum()} / {len(out)} "
      f"({100*out.mean():.2f}%)")

fig, axes = plot_latent_with_rectangle(det_latent_val, rect, slice_range=(0, 1000))
plt.show()


# ============================================================
# POLYNOMIAL ENVELOPE FIT — configurable rolling visualisation
# ============================================================
INIT_TS     = 12    # polynomial fitting window size
POLY_DEGREE = 1     # 1 = linear; bump to 2 if residuals look biased
PLOT_START  = 0     # first timestep to display
PLOT_END    = 1000   # last  timestep to display


def fit_poly_window(t_vals, z_vals, degree=1):
    """Fit a polynomial to z, return (poly, derivative poly)."""
    coeffs = np.polyfit(t_vals, z_vals, deg=degree)
    poly   = np.poly1d(coeffs)
    return poly, poly.deriv()


def plot_envelope_fit(
    latent_array,
    rect,
    init_ts=20,
    degree=1,
    mu=None, Sigma=None, tau=None,
    plot_start=0,
    plot_end=None,       # defaults to plot_start + init_ts (single window)
    figsize_per_row=(12, 2.2),
):
    """
    Plots latent trajectories from plot_start to plot_end.

    The displayed range is divided into non-overlapping windows of init_ts steps.
    Each window gets its own polynomial fit (benign points only).
    Vertical grey lines mark the window boundaries.

    Columns
    -------
    Left  : z_i  (blue) + piecewise poly fit (orange dashed) +
            rect bounds (red dotted) + τ-band (orange shaded)
    Right : ż_i (green) + d/dt poly fit (orange dashed) +
            rect bounds (red dotted) + derived τ-band (orange shaded)
            band_ż = τ · √(Σ_z[i,i] / 2)  — propagated from z via central diff

    Parameters
    ----------
    plot_start : int   first absolute timestep to display
    plot_end   : int   last  absolute timestep to display
                       (default: plot_start + init_ts → single-window mode)
    """
    n_dims = latent_array.shape[1]
    assert n_dims % 2 == 0, "latent_det must be even (z + z_dot halves)"
    D = n_dims // 2

    if plot_end is None:
        plot_end = plot_start + init_ts
    plot_end = min(plot_end, len(latent_array))

    t_abs  = np.arange(plot_start, plot_end)   # absolute time axis for plotting
    n_plot = len(t_abs)

    # ---- pre-compute sliding polynomial fits (step=1, lookback=init_ts) ----
    # For each timestep t, window = latent_array[t-init_ts+1 : t+1] (ends at t inclusive).
    # Polynomial is evaluated at the last relative position (= current point t).
    # This is exactly what the online detector does, just visualised over the whole range.
    z_fit_arr    = np.full((n_plot, D), np.nan)
    zdot_fit_arr = np.full((n_plot, D), np.nan)

    for idx in range(n_plot):
        t         = plot_start + idx
        win_s     = max(0, t - init_ts + 1)
        win_e     = t + 1
        actual_ts = win_e - win_s

        if actual_ts < degree + 2:
            continue

        window      = latent_array[win_s:win_e]
        t_rel       = np.arange(actual_ts, dtype=float)
        benign_mask = ~rectangle_flags(window, rect)

        if benign_mask.sum() < degree + 2:
            continue

        benign_pts = window[benign_mask]
        benign_t   = t_rel[benign_mask]
        t_eval     = float(actual_ts - 1)   # relative position of point t in window

        for i in range(D):
            poly, dpoly          = fit_poly_window(benign_t, benign_pts[:, i], degree)
            z_fit_arr[idx, i]    = poly(t_eval)
            zdot_fit_arr[idx, i] = dpoly(t_eval)

    # ---- plotting ----
    fig, axes = plt.subplots(
        D, 2,
        figsize=(figsize_per_row[0], figsize_per_row[1] * D),
        sharex=True,
        squeeze=False,
    )

    for i in range(D):
        z_i    = latent_array[plot_start:plot_end, i]
        zdot_i = latent_array[plot_start:plot_end, i + D]

        # ---- z_i ----
        ax = axes[i, 0]
        if Sigma is not None and tau is not None:
            band = tau * np.sqrt(Sigma[i, i])
            ax.fill_between(t_abs,
                            z_fit_arr[:, i] - band,
                            z_fit_arr[:, i] + band,
                            alpha=0.25, color="tab:orange",
                            label=f"τ-envelope (τ={tau:.2f})")
        ax.plot(t_abs, z_i,              color="tab:blue",   lw=1.0, label=f"z_{i}")
        ax.plot(t_abs, z_fit_arr[:, i],  color="tab:orange", lw=1.8, ls="--",
                label=f"poly k={degree}")
        ax.axhline(rect["low"][i],  color="tab:red", ls=":", lw=0.9,
                   label=rect["label_low"])
        ax.axhline(rect["high"][i], color="tab:red", ls=":", lw=0.9,
                   label=rect["label_high"])
        ax.set_ylabel(f"z_{i}")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_title(f"States  z   (poly degree {degree})")
            ax.legend(fontsize=8, loc="upper right")

        # ---- ż_i ----
        ax = axes[i, 1]
        if Sigma is not None and tau is not None:
            band_dot = tau * np.sqrt(Sigma[i, i] / 2)
            ax.fill_between(t_abs,
                            zdot_fit_arr[:, i] - band_dot,
                            zdot_fit_arr[:, i] + band_dot,
                            alpha=0.25, color="tab:orange",
                            label=f"τ-envelope derived (τ={tau:.2f})")
        ax.plot(t_abs, zdot_i,              color="tab:green",  lw=1.0, label=f"ż_{i}")
        ax.plot(t_abs, zdot_fit_arr[:, i],  color="tab:orange", lw=1.8, ls="--",
                label="d/dt poly")
        ax.axhline(rect["low"][i + D],  color="tab:red", ls=":", lw=0.9)
        ax.axhline(rect["high"][i + D], color="tab:red", ls=":", lw=0.9)
        ax.set_ylabel(f"ż_{i}")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_title("Derivatives  ż   (d/dt poly)")
            ax.legend(fontsize=8, loc="upper right")

    axes[-1, 0].set_xlabel("time step")
    axes[-1, 1].set_xlabel("time step")
    tau_str = f", τ={tau:.3f}" if tau is not None else ""
    fig.suptitle(
        f"Rolling envelope fit  [{plot_start}:{plot_end}], "
        f"window={init_ts}, poly degree={degree}{tau_str}",
        y=1.01,
    )
    fig.tight_layout()
    return fig, axes


# ============================================================
# CALIBRATE ENVELOPE  (pseudocode Step 2 — z-only Mahalanobis)
# ============================================================
def calibrate_envelope(latent_array, rect, ts, k=1, coverage=0.99, eps=1e-4):
    """
    Roll a window of size ts across latent_array.
    For each window:
      - keep only benign points (inside safety rectangle)
      - fit a degree-k polynomial to each z dimension
      - collect z residuals  r_z  ∈ R^D   (states only)
      - also collect r_zdot for the TDC consistency check (not used in detection)

    Mahalanobis distance is computed in z-space only (D×D covariance).
    The ż tolerance is derived analytically:  band_ż = τ · √(Σ_z[i,i] / 2)
    based on the central-difference propagation  r_ż ≈ (ε(t+1) − ε(t−1)) / 2.

    Returns
    -------
    mu        : (D,)   mean of z residuals  (≈ 0 if fit is unbiased)
    Sigma     : (D,D)  regularised z covariance
    Sigma_inv : (D,D)  inverse z covariance
    tau       : float  Mahalanobis threshold at `coverage` quantile
    """
    n_dims = latent_array.shape[1]
    D      = n_dims // 2
    t_rel  = np.arange(ts, dtype=float)
    Rz     = []    # z residuals   — used for Mahalanobis
    Rzdot  = []    # ż residuals   — used only for TDC consistency check

    for start in range(len(latent_array) - ts + 1):
        window      = latent_array[start : start + ts]
        benign_mask = ~rectangle_flags(window, rect)

        if benign_mask.sum() < k + 2:
            continue

        benign_pts = window[benign_mask]
        benign_t   = t_rel[benign_mask]
        n_benign   = benign_mask.sum()

        r_z    = np.empty((n_benign, D))
        r_zdot = np.empty((n_benign, D))

        for i in range(D):
            coeffs       = np.polyfit(benign_t, benign_pts[:, i], deg=k)
            poly         = np.poly1d(coeffs)
            dpoly        = poly.deriv()
            r_z[:, i]    = benign_pts[:, i]     - poly(benign_t)
            r_zdot[:, i] = benign_pts[:, i + D] - dpoly(benign_t)

        Rz.append(r_z)
        Rzdot.append(r_zdot)

    Rz    = np.vstack(Rz)                          # (N, D)
    Rzdot = np.vstack(Rzdot)                       # (N, D)

    mu        = Rz.mean(axis=0)                    # (D,)
    Sigma     = np.cov(Rz.T) + eps * np.eye(D)    # (D, D)
    Sigma_inv = np.linalg.inv(Sigma)

    diff  = Rz - mu
    m_cal = np.sqrt(np.einsum('ij,jk,ik->i', diff, Sigma_inv, diff))
    tau   = float(np.quantile(m_cal, coverage))

    # --- TDC dynamic-consistency check (coverage parity) ---
    #
    # Two levels of check:
    #
    # 1) Per-dimension marginal — directly comparable to the visual plot.
    #    Derived band: τ · √(Σ_z[i,i] / 2).  Verdict lives here.
    #
    # 2) Joint empirical — uses Σ_ż estimated directly from Rzdot (no theoretical
    #    assumption about off-diagonals).  Low joint coverage when Σ_ż_empirical ≠ Σ_z/2
    #    off-diagonally is NOT a TDC failure; it just means the temporal cross-correlation
    #    structure of z breaks the i.i.d. assumption in the diagonal formula.
    #    τ_ż is reported for comparison to τ_z (scale parity).

    # 1) Per-dim marginal
    dim_bands    = tau * np.sqrt(np.diag(Sigma) / 2)
    dim_coverage = (np.abs(Rzdot) <= dim_bands).mean(axis=0)   # (D,)
    mean_marginal = float(dim_coverage.mean())
    gap_marginal  = abs(mean_marginal - coverage)
    verdict = ("✓ consistent" if gap_marginal < 0.03
               else ("⚠ moderate gap" if gap_marginal < 0.10
                     else "✗ TDC underfit — derivatives noisier than states imply"))

    # 2) Joint empirical Σ_ż (no Σ_z/2 assumption)
    mu_zdot       = Rzdot.mean(axis=0)
    Sig_zdot      = np.cov(Rzdot.T) + eps * np.eye(D)
    Sig_zdot_inv  = np.linalg.inv(Sig_zdot)
    diff_zdot     = Rzdot - mu_zdot
    m_zdot_emp    = np.sqrt(np.einsum('ij,jk,ik->i', diff_zdot, Sig_zdot_inv, diff_zdot))
    tau_zdot      = float(np.quantile(m_zdot_emp, coverage))   # empirical τ for ż

    print(f"\nCalibration — {len(Rz):,} residuals, coverage={coverage:.0%}, τ_z={tau:.4f}")
    print(f"TDC dynamic consistency:")
    print(f"  z  joint coverage  : {coverage:.1%}  (by construction,  τ_z={tau:.4f})")
    print(f"  ż  per-dim coverage: {' '.join(f'{c:.0%}' for c in dim_coverage)}")
    print(f"  ż  mean marginal   : {mean_marginal:.1%}  ← verdict: {verdict}")
    print(f"  ż  empirical τ_ż   : {tau_zdot:.4f}  (τ_z={tau:.4f},"
          f" ratio={tau_zdot/tau:.3f} — ideal ≈ 1/√2 ≈ {1/np.sqrt(2):.3f})")

    return mu, Sigma, Sigma_inv, tau


# ---- run calibration, then plot rolling envelope with tolerance ----
mu_cal, Sigma_cal, Sigma_inv_cal, tau_cal = calibrate_envelope(
    det_latent_train, rect, ts=INIT_TS, k=POLY_DEGREE
)

fig2, axes2 = plot_envelope_fit(
    det_latent_test, rect,
    init_ts=INIT_TS, degree=POLY_DEGREE,
    mu=mu_cal, Sigma=Sigma_cal, tau=tau_cal,
    plot_start=PLOT_START, plot_end=PLOT_END,
)
plt.show()
