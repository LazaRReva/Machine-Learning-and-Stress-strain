import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
from tqdm import tqdm
import random
import os

# --- 0. Setup ---
# Create a directory to save the plots
output_dir = 'multi_step_validation_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"Plots will be saved to the '{output_dir}/' directory.")

# Suppress the specific UserWarning from Matplotlib
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')


# --- 1. Data Loading from a Single New Source ---
def load_unified_data(filepath):
    """
    Loads a single MATLAB .mat file containing all cycle data.
    """
    try:
        mat = scipy.io.loadmat(filepath)
        data_key = [k for k in mat.keys() if not k.startswith('__')][0]
        z_filtered_all = mat[data_key][0]
        stacked_data = np.stack([cell_data for cell_data in z_filtered_all], axis=-1)
        return stacked_data
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        return None
    except (KeyError, IndexError):
        print(f"Error: Could not find or parse the expected data structure in {filepath}.")
        return None


# Load all available data from the new unified file
all_strain_data_raw = load_unified_data('Zfiltered_AZ3_unload_again0813.mat')

if all_strain_data_raw is not None:
    # --- DATA SELECTION: Exclude Cycle 100 ---
    all_strain_data = all_strain_data_raw[:, :, :-1]  # Take first 11 cycles (1-90)
    all_cycle_numbers = np.array([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    print(f"Dataset loaded and selected. Using cycles 1-90. Shape: {all_strain_data.shape}")
else:
    print("Script terminated due to data loading failure.")
    exit()

# --- 2. Data Preprocessing and Feature Engineering ---
strain_threshold = 0.002
all_strain_data[all_strain_data < strain_threshold] = np.nan
print(f"Strain values below {strain_threshold} have been set to NaN.")


def calculate_inverse_gradients(strain_data):
    """
    Calculates the RECIPROCAL of the spatial gradient for each point.
    """
    inverse_gradients = np.zeros_like(strain_data)
    epsilon = 1e-8
    for i in range(strain_data.shape[2]):
        field = strain_data[:, :, i]
        diff_up = field - np.roll(field, 1, axis=0);
        diff_up[0, :] = 0
        diff_down = field - np.roll(field, -1, axis=0);
        diff_down[-1, :] = 0
        diff_left = field - np.roll(field, 1, axis=1);
        diff_left[:, 0] = 0
        diff_right = field - np.roll(field, -1, axis=1);
        diff_right[:, -1] = 0

        sum_of_squares = np.nansum(np.stack([diff_up ** 2, diff_down ** 2, diff_left ** 2, diff_right ** 2]), axis=0)
        gradient = np.sqrt(sum_of_squares)

        inverse_gradients[:, :, i] = 1.0 / (gradient + epsilon)

    return inverse_gradients


all_inverse_gradient_data = calculate_inverse_gradients(all_strain_data)
print("Inverse strain gradient calculation complete.")

# --- 3. Preparing Data for TWO Models (Train on 1-70 -> 80) ---
height, width, num_total_cycles = all_strain_data.shape

# --- Data for Model 1: With Inverse Gradient ---
features_with_inv_gradient = np.stack((all_strain_data, all_inverse_gradient_data), axis=-1)
reshaped_features_wg = features_with_inv_gradient.reshape(-1, num_total_cycles, 2)
valid_indices_mask = ~np.isnan(reshaped_features_wg).any(axis=(1, 2))
valid_features_wg = reshaped_features_wg[valid_indices_mask]
valid_indices_array = np.where(valid_indices_mask)[0]
print(f"Found {valid_features_wg.shape[0]} valid spatial points for training and testing.")

# --- Data for Model 2: No Extra Feature (Control) ---
features_no_gradient = all_strain_data[..., np.newaxis]
reshaped_features_ng = features_no_gradient.reshape(-1, num_total_cycles, 1)
valid_features_ng = reshaped_features_ng[valid_indices_mask]

# --- Normalization and Dataloaders for Both ---
scalers_wg = [MinMaxScaler().fit(d) for d in valid_features_wg]
scaled_features_wg = np.array([s.transform(d) for s, d in zip(scalers_wg, valid_features_wg)])
# Train on the first 9 points (cycles 1-70) to predict the 10th (cycle 80)
X_train_wg = torch.tensor(scaled_features_wg[:, :9, :], dtype=torch.float32)
y_train_wg = torch.tensor(scaled_features_wg[:, 9, 0], dtype=torch.float32).unsqueeze(1)
dataloader_wg = DataLoader(TensorDataset(X_train_wg, y_train_wg), batch_size=1024, shuffle=True)

scalers_ng = [MinMaxScaler().fit(d) for d in valid_features_ng]
scaled_features_ng = np.array([s.transform(d) for s, d in zip(scalers_ng, valid_features_ng)])
X_train_ng = torch.tensor(scaled_features_ng[:, :9, :], dtype=torch.float32)
y_train_ng = torch.tensor(scaled_features_ng[:, 9, 0], dtype=torch.float32).unsqueeze(1)
dataloader_ng = DataLoader(TensorDataset(X_train_ng, y_train_ng), batch_size=1024, shuffle=True)


# --- 4. LSTM Model Definition and Training ---
class StrainPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(StrainPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        bn_out = self.bn(last_time_step_out)
        out = self.linear(bn_out)
        return out


def train_model(dataloader, model, model_name):
    NUM_EPOCHS = 200
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    print(f"\n--- Starting Training for {model_name} ---")
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        model.train()
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=False):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Average Loss: {avg_epoch_loss:.6f}")

    print(f"--- Training finished for {model_name} ---")
    return model


model_with_inv_gradient = train_model(dataloader_wg, StrainPredictor(input_size=2), "Model with Inverse Gradient")
model_control = train_model(dataloader_ng, StrainPredictor(input_size=1), "Control Model (No Extra Feature)")


# --- 5. Prediction and Visualization ---
def predict_future(model, initial_sequence, scaler, has_extra_feature):
    model.eval()
    predict_input = torch.tensor(initial_sequence, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction_scaled = model(predict_input).cpu().numpy()[0, 0]

    if has_extra_feature:
        prediction_real = scaler.inverse_transform([[prediction_scaled, 0]])[0, 0]
    else:
        prediction_real = scaler.inverse_transform([[prediction_scaled]])[0, 0]

    return prediction_real


# --- Identify points to analyze ---
max_strain_value = np.nanmax(all_strain_data)
max_indices = np.where(all_strain_data == max_strain_value)
max_y, max_x = max_indices[0][0], max_indices[1][0]
max_point_flat_index = max_y * width + max_x
points_to_plot = [max_point_flat_index]
neighborhood_size = 6
y_min, y_max = max(0, max_y - neighborhood_size), min(height, max_y + neighborhood_size)
x_min, x_max = max(0, max_x - neighborhood_size), min(width, max_x + neighborhood_size)
neighbor_indices = [y * width + x for y in range(y_min, y_max) for x in range(x_min, x_max) if
                    (y * width + x) in valid_indices_array and (y * width + x) != max_point_flat_index]
if len(neighbor_indices) >= 3:
    points_to_plot.extend(random.sample(neighbor_indices, 3))
else:
    points_to_plot.extend(neighbor_indices)

for point_flat_index in points_to_plot:
    try:
        point_valid_index = np.where(valid_indices_array == point_flat_index)[0][0]
        point_y, point_x = np.unravel_index(point_flat_index, (height, width))
    except IndexError:
        continue

    # --- Two-Stage Prediction ---
    # 1. Predict Cycle 80 using 1-70 history
    pred_80_wg = predict_future(model_with_inv_gradient, scaled_features_wg[point_valid_index, :9, :],
                                scalers_wg[point_valid_index], True)
    pred_80_ng = predict_future(model_control, scaled_features_ng[point_valid_index, :9, :],
                                scalers_ng[point_valid_index], False)

    # 2. Predict Cycle 90 using 1-80 history
    pred_90_wg = predict_future(model_with_inv_gradient, scaled_features_wg[point_valid_index, :10, :],
                                scalers_wg[point_valid_index], True)
    pred_90_ng = predict_future(model_control, scaled_features_ng[point_valid_index, :10, :],
                                scalers_ng[point_valid_index], False)

    # Get full history for plotting
    full_history = valid_features_wg[point_valid_index, :, 0]

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(all_cycle_numbers, full_history, 'o-', color='royalblue', label='Actual Evolution Path (Cycles 1-90)')

    # Mark predictions for Cycle 80
    ax.plot(80, pred_80_wg, '^', color='red', markersize=12, label=f'With Inv. Grad. Pred. @ C80: {pred_80_wg:.4f}')
    ax.plot(80, pred_80_ng, 'X', color='purple', markersize=12, label=f'Control Model Pred. @ C80: {pred_80_ng:.4f}')

    # Mark predictions for Cycle 90
    ax.plot(90, pred_90_wg, '^', color='red', markersize=12, mfc='none',
            label=f'With Inv. Grad. Pred. @ C90: {pred_90_wg:.4f}')
    ax.plot(90, pred_90_ng, 'X', color='purple', markersize=12, mfc='none',
            label=f'Control Model Pred. @ C90: {pred_90_ng:.4f}')

    plot_title = f'Multi-Step Validation for Point (y={point_y}, x={point_x})'
    ax.set_title(plot_title, fontsize=16)
    ax.set_xlabel('Cycle Number');
    ax.set_ylabel('Plastic Strain')
    ax.legend();
    ax.grid(True);
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'multistep_val_plot_y{point_y}_x{point_x}.png')
    plt.savefig(save_path, dpi=300);
    plt.close(fig)
    print(f"Saved validation plot to: {save_path}")

# --- 6. Generate and Plot Comparison Heatmaps for Both Cycles ---
for cycle_to_predict in [80, 90]:
    print(f"\nGenerating comparison heatmaps for Cycle {cycle_to_predict}...")
    heatmap_wg = np.full((height, width), np.nan)
    heatmap_ng = np.full((height, width), np.nan)

    # Determine input sequence length based on prediction target
    if cycle_to_predict == 80:
        input_slice = slice(0, 9)  # Use cycles 1-70
    else:  # cycle_to_predict == 90
        input_slice = slice(0, 10)  # Use cycles 1-80

    for i in tqdm(range(len(valid_features_wg)), desc=f"Predicting Cycle {cycle_to_predict} for all points"):
        point_flat_index = valid_indices_array[i]
        point_y, point_x = np.unravel_index(point_flat_index, (height, width))

        point_valid_index = i
        pred_wg = predict_future(model_with_inv_gradient, scaled_features_wg[i, input_slice, :], scalers_wg[i], True)
        pred_ng = predict_future(model_control, scaled_features_ng[i, input_slice, :], scalers_ng[i], False)

        heatmap_wg[point_y, point_x] = pred_wg
        heatmap_ng[point_y, point_x] = pred_ng

    actual_heatmap = all_strain_data[:, :, all_cycle_numbers.tolist().index(cycle_to_predict)]
    error_map_wg = np.abs(heatmap_wg - actual_heatmap)
    error_map_ng = np.abs(heatmap_ng - actual_heatmap)
    error_improvement_map = error_map_ng - error_map_wg

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Ablation Study: Impact of Inverse Strain Gradient on Cycle {cycle_to_predict} Prediction',
                 fontsize=20)
    vmin, vmax = 0, 0.06

    im1 = axes[0, 0].imshow(heatmap_wg, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Prediction with Inverse Gradient', fontsize=14)
    fig.colorbar(im1, ax=axes[0, 0], label='Strain')

    im2 = axes[0, 1].imshow(heatmap_ng, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Prediction (Control Model)', fontsize=14)
    fig.colorbar(im2, ax=axes[0, 1], label='Strain')

    im3 = axes[1, 0].imshow(actual_heatmap, cmap='jet', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f'Actual Experimental Data @ Cycle {cycle_to_predict}', fontsize=14)
    fig.colorbar(im3, ax=axes[1, 0], label='Strain')

    error_vmax = np.nanmax(np.abs(error_improvement_map))
    im4 = axes[1, 1].imshow(error_improvement_map, cmap='coolwarm', vmin=-error_vmax, vmax=error_vmax)
    axes[1, 1].set_title('Error Improvement by Inverse Gradient', fontsize=14)
    fig.colorbar(im4, ax=axes[1, 1], label='Error(Control) - Error(InvGrad)\n(Red = Inv. Grad. Helps)')

    for ax_row in axes:
        for ax in ax_row: ax.axes.get_xaxis().set_visible(False); ax.axes.get_yaxis().set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    heatmap_save_path = os.path.join(output_dir, f'multistep_val_heatmap_C{cycle_to_predict}.png')
    plt.savefig(heatmap_save_path, dpi=300)
    print(f"Saved validation heatmap for Cycle {cycle_to_predict} to: {heatmap_save_path}")
    plt.close(fig)

print("\nAll tasks completed successfully.")
