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

timestamp = "20260527_174105" 

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
X_train_smooth = moving_average(df_train.to_numpy(), 12, axis=0)
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
dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_dataset = TensorDataset(X_val_temporal, torch.zeros(X_val_temporal.shape[0]))
dataloader_val = DataLoader(val_dataset, batch_size=1, shuffle=False)

test_dataset = TensorDataset(X_test_temporal, torch.zeros(X_test_temporal.shape[0]))
dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

neuron_count = len(measurements_columns)
latent_det = 10

latent_stat = 4

AE = hTDCAutoEncoder(neuron_count, latent_det=latent_det, latent_stat=latent_stat)

AE.load_state_dict(torch.load(model_path))

######################### TRAIN DATA ###########################################

rec_errors = []
det_latent_space = []
stat_latent_space = []
central_diffs = []



for datasample, _ in dataloader:
    output, det_latent, stat_latent = AE(datasample[:,0,:])  

    det_latent = det_latent.detach().cpu().squeeze()
    
    det_latent_space.append(det_latent.numpy())
    
    rec_loss = F.mse_loss(datasample[:,0,:], output)
    rec_errors.append(rec_loss.item())


data = np.array(rec_errors)

# Calculate the maximum value
max_value = np.max(data)

# Calculate the 75th percentile
percentile_85 = np.percentile(data, 85)
# Calculate the 75th percentile
percentile_90 = np.percentile(data, 90)
# Calculate the 75th percentile
percentile_95 = np.percentile(data, 95)


# # Convert latent_space to a numpy array for easier manipulation
det_latent_space = np.array(det_latent_space)

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

rect = build_safety_rectangle(det_latent_space, mode="percentile")

print("Safety rectangle (min/max):")
for d in range(det_latent_space.shape[1]):
    print(f"  dim {d:2d}:  [{rect['low'][d]: .4f},  {rect['high'][d]: .4f}]")

out = rectangle_flags(det_latent_space, rect)
print(f"\nTraining points outside rectangle: {out.sum()} / {len(out)} "
      f"({100*out.mean():.2f}%)")

fig, axes = plot_latent_with_rectangle(det_latent_space, rect, slice_range=(0, 3000))
plt.show()
