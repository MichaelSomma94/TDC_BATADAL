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
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(timestamp)
# Create a directory with the timestamp
folder_path = Path(f"./models/TDCAE/{timestamp}")
folder_path.mkdir(parents=True, exist_ok=True)
# Set up logging
logging.basicConfig(filename=rf"{folder_path}\training.log", level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


# Set up logging
logging.basicConfig(
    filename=folder_path.joinpath("training.log").as_posix(),
    level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s'
)

# Load and preprocess training data
train_file = Path("data", "BATADAL_dataset03.csv") # Replace with your training file path
#test_file = Path("data", "coupled_vdp_test.csv")    # Replace with your test file path

df_Train = pd.read_csv(train_file)
#df_test = pd.read_csv(test_file)
time = df_Train["DATETIME"]
measurements_columns = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7', 'F_PU1', 'S_PU1', 'F_PU2', 'S_PU2', 'F_PU3', 'S_PU3', 'F_PU4', 'S_PU4', 'F_PU5', 'S_PU5', 'F_PU6', 'S_PU6', 'F_PU7', 'S_PU7', 'F_PU8', 'S_PU8', 'F_PU9', 'S_PU9', 'F_PU10', 'S_PU10', 'F_PU11', 'S_PU11', 'F_V2', 'S_V2', 'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415', 'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']


df_train = df_Train[measurements_columns]


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
# X_test_temporal = create_temporal_dataset_per_engine(df_test, scaler, colums_to_drop=["engine_id", "time_step", "label"])  # Test set processed separately

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(X_train_temporal, torch.zeros(X_train_temporal.shape[0]))
dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(X_val_temporal, torch.zeros(X_val_temporal.shape[0]))
dataloader_val = DataLoader(val_dataset, batch_size=1, shuffle=False)

# test_dataset = TensorDataset(X_test_temporal, torch.zeros(X_test_temporal.shape[0]))
# dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

neuron_count = len(measurements_columns)
latent_det = 10

latent_stat = 4

mse_loss = []
tdc_loss = []
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


AE = hTDCAutoEncoder(neuron_count, latent_det=latent_det, latent_stat=latent_stat).apply(init_weights)
criterion = TDCLoss(h=1, alpha=50) #100 
#training
epochs = 30
optimizer = torch.optim.Adam(AE.parameters(), lr=0.002)
for epoch in range(epochs):
   
    AE.train()
    for batch_features, _ in dataloader:
        optimizer.zero_grad()  # Zero the gradients
        targets= batch_features

        output, latent_det, latent_stat = AE(targets)

        total_loss, reconstruction_loss, derivative_loss = criterion(output[:, 0, :],targets[:, 0, :], latent_det)

        total_loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        mse_loss.append(reconstruction_loss.item())
        tdc_loss.append(derivative_loss.item())

    # Validation
    AE.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output, _, _ = AE(X_val_temporal[:, 0, :])

        val_loss = F.mse_loss(X_val_temporal[:, 0, :], output)

    print(f'Epoch [{epoch+1}/{epochs}], MSE Loss:  {sum(mse_loss)/len(mse_loss)/32}, Val Loss: {val_loss.item():.4f},\
                Derivative Loss: {sum(tdc_loss)/len(tdc_loss)/32},')
    
    torch.save(
        AE.state_dict(), 
        folder_path.joinpath("trained_model.pth").as_posix(),
    )




