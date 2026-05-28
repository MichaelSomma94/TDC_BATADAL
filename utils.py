import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

class hTDCAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_det, latent_stat):
        super().__init__()
        self.latent_det = latent_det
        self.latent_stat = latent_stat
        latent_dim = latent_det + latent_stat
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
        )
    def forward(self, x):
        latent = self.encoder(x)
        latent_det = latent[:,:self.latent_det]
        latent_stat = latent[:, self.latent_det:]
        recon = self.decoder(latent)
        return recon, latent_det, latent_stat

class TDCLoss(nn.Module):
    def __init__(self, h, alpha):
        super(TDCLoss, self).__init__()
        self.h = h  # Time step size for finite differences
        self.alpha = alpha  # Weight for the derivative loss
        
    def forward(self, inputs, recon, latent):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, inputs)
        
        # Partition the latent space
        half_dim = latent.shape[-1] // 2
        states_current = latent[:, 0, :half_dim]
        derivatives_current = latent[:, 0, half_dim:]
        
        states_next = latent[:, 2, :half_dim]
        states_previous = latent[:, 1, :half_dim]
        
        # Compute central differences for the derivative loss
        d_states_dt = (states_next - states_previous) / (2 * self.h)
        derivative_loss = F.mse_loss(derivatives_current, d_states_dt)
        
        # Total loss
        total_loss = recon_loss + self.alpha * derivative_loss
        
        return total_loss, recon_loss, derivative_loss



class VanillaAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VanillaAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # ----- Encoder -----
        self.encoder_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_layer   = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # ----- Decoder -----
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)  # no activation; match your target scaling
        )

    def encode(self, x):
        h = self.encoder_backbone(x)
        mean   = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        return mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar


def moving_average(values, window_size, axis=0):
    """
    Computes the moving average along a specified axis for an n-dimensional array.
    
    Args:
        values (np.ndarray): Input array.
        window_size (int): The size of the moving window.
        axis (int): Axis along which to compute the moving average. Default is 0.
    
    Returns:
        np.ndarray: Array of the same shape as `values` with the moving average applied along the specified axis.
    """
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    
    # Ensure the axis is valid
    if axis < 0:
        axis += values.ndim
    
    if axis >= values.ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {values.ndim}")
    
    # Compute the cumulative sum along the specified axis
    cumsum = np.cumsum(np.insert(values, 0, 0, axis=axis), axis=axis)
    
    # Compute the moving average along the specified axis
    slicing1 = [slice(None)] * values.ndim
    slicing2 = [slice(None)] * values.ndim
    slicing1[axis] = slice(window_size, None)
    slicing2[axis] = slice(None, -window_size)
    
    moving_averages = (cumsum[tuple(slicing1)] - cumsum[tuple(slicing2)]) / window_size
    
    # Pad the beginning of the result along the specified axis
    padding_shape = list(values.shape)
    padding_shape[axis] = window_size - 1
    padding = np.take(values, range(window_size - 1), axis=axis)
    
    result = np.concatenate((padding, moving_averages), axis=axis)
    
    return result

def create_temporal_dataset(X, scaler):
    """
    Creates a temporal dataset for each engine with [prev, current, next] steps.
    """
    temporal_data = []
    
    # Scale the features
    averaged_scal_features = scaler.transform(X)
    #averaged_scal_features = moving_average(features_scaled, window_size, axis=1)
    # Create [prev, current, next] samples
    current = averaged_scal_features[1:-1]
    prev = averaged_scal_features[:-2]
    next_ = averaged_scal_features[2:]
        
    # Stack the samples together
    temporal_samples = torch.stack([
        torch.tensor(current, dtype=torch.float32),
        torch.tensor(prev, dtype=torch.float32),
        torch.tensor(next_, dtype=torch.float32)
    ], dim=1)

    temporal_data.append(temporal_samples)
    
    # Combine data from all engines
    return torch.cat(temporal_data, dim=0)

def create_temporal_dataset_per_engine(df, scaler, colums_to_drop, window_size=10):
    """
    Creates a temporal dataset for each engine with [prev, current, next] steps.
    """
    temporal_data = []
    
    for engine_id, group in df.groupby("engine_id"):
        # Sort by time_step to ensure temporal order
        group = group.sort_values(by="time_step")
        features = group.drop(columns=colums_to_drop).to_numpy()
        
        # Scale the features
        averaged_scal_features = scaler.transform(features)
        #averaged_scal_features = moving_average(features_scaled, window_size, axis=1)
        # Create [prev, current, next] samples
        current = averaged_scal_features[1:-1]
        prev = averaged_scal_features[:-2]
        next_ = averaged_scal_features[2:]
        
        # Stack the samples together
        temporal_samples = torch.stack([
            torch.tensor(current, dtype=torch.float32),
            torch.tensor(prev, dtype=torch.float32),
            torch.tensor(next_, dtype=torch.float32)
        ], dim=1)

        temporal_data.append(temporal_samples)
    
    # Combine data from all engines
    return torch.cat(temporal_data, dim=0)


def scale_feature(oDf, fillnaMode = 'ffill', save_scaler_path=None):
    """
    fills nan values and scales the feature columns
    output. a numpy array without nans and scaled features
    """
    scaler = MinMaxScaler()
    if fillnaMode == 'interpol':
        oDf.interpolate(method='linear')
    elif fillnaMode =='ffill':
        oDf.ffill(inplace=False) 

    # Split the data into training and validation sets
    X_train, X_val = train_test_split(oDf, test_size=0.1, random_state=42)
    aTrainFeatures = scaler.fit_transform(X_train)
    aValFeatures = scaler.transform(X_val)
    if save_scaler_path:  
        joblib.dump(scaler, save_scaler_path)
    
    return aTrainFeatures, aValFeatures