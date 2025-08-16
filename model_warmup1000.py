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
output_dir = '../robust_ensemble_warmup_plots'
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
    # --- DATA SELECTION: Use cycles 1-90, ignore cycle 100 ---
    all_strain_data = all_strain_data_raw[:, :, :11]
    all_cycle_numbers = np.array([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    print(f"Dataset loaded. Using data up to Cycle 90. Shape: {all_strain_data.shape}")
else:
    print("Script terminated due to data loading failure.")
    exit()

# --- 2. Data Preprocessing and Feature Engineering ---
strain_threshold = 0.002
all_strain_data[all_strain_data < strain_threshold] = np.nan
print(f"Strain values below {strain_threshold} have been set to NaN.")


def calculate_gradients_and_inverse(strain_data):
    """
    Calculates both the spatial gradient and its reciprocal for each point.
    """
    gradients = np.zeros_like(strain_data)
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

        gradients[:, :, i] = gradient
        inverse_gradients[:, :, i] = 1.0 / (gradient + epsilon)

    return gradients, inverse_gradients


all_gradient_data, all_inverse_gradient_data = calculate_gradients_and_inverse(all_strain_data)
print("Gradient and Inverse Gradient calculation complete.")

# --- STABILITY STRATEGY 1: Feature Taming (Clipping) ---
inv_grad_clip_threshold = np.nanpercentile(all_inverse_gradient_data, 99)
all_inverse_gradient_data = np.clip(all_inverse_gradient_data, a_min=None, a_max=inv_grad_clip_threshold)
print(f"Applied feature clipping to Inverse Gradient at 99th percentile value: {inv_grad_clip_threshold:.2f}")

grad_clip_threshold = np.nanpercentile(all_gradient_data, 99)
all_gradient_data = np.clip(all_gradient_data, a_min=None, a_max=grad_clip_threshold)
print(f"Applied feature clipping to Gradient at 99th percentile value: {grad_clip_threshold:.2f}")

# --- 3. Preparing Data for THREE Models ---
height, width, num_total_cycles = all_strain_data.shape

temp_features = np.stack((all_strain_data, all_gradient_data, all_inverse_gradient_data), axis=-1)
reshaped_temp = temp_features.reshape(-1, num_total_cycles, 3)
valid_indices_mask = ~np.isnan(reshaped_temp).any(axis=(1, 2))
valid_indices_array = np.where(valid_indices_mask)[0]
print(f"Found {len(valid_indices_array)} common valid spatial points for all models.")


def prepare_dataloaders(features, valid_mask):
    reshaped_features = features.reshape(-1, num_total_cycles, features.shape[-1])
    valid_features = reshaped_features[valid_mask]

    scalers = [MinMaxScaler().fit(d) for d in valid_features]
    scaled_features = np.array([s.transform(d) for s, d in zip(scalers, valid_features)])

    X_train = torch.tensor(scaled_features[:, :9, :], dtype=torch.float32)
    y_train = torch.tensor(scaled_features[:, 9, 0], dtype=torch.float32).unsqueeze(1)
    dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=True)

    return dataloader, scaled_features, scalers, valid_features


features_inv_grad = np.stack((all_strain_data, all_inverse_gradient_data), axis=-1)
dataloader_ig, scaled_features_ig, scalers_ig, valid_features_ig = prepare_dataloaders(features_inv_grad,
                                                                                       valid_indices_mask)

features_grad = np.stack((all_strain_data, all_gradient_data), axis=-1)
dataloader_g, scaled_features_g, scalers_g, valid_features_g = prepare_dataloaders(features_grad, valid_indices_mask)

features_control = all_strain_data[..., np.newaxis]
dataloader_c, scaled_features_c, scalers_c, valid_features_c = prepare_dataloaders(features_control, valid_indices_mask)


# --- 4. LSTM Model Definition and Ensemble Training ---
class StrainPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(StrainPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        bn_out = self.bn(lstm_out[:, -1, :])
        return self.linear(bn_out)


def train_ensemble(dataloader, input_size, model_name, n_members=10):
    ensemble = []
    print(f"\n--- Starting Ensemble Training for {model_name} ---")
    for i in range(n_members):
        print(f"Training member {i + 1}/{n_members}...")
        model = StrainPredictor(input_size=input_size)
        criterion = nn.MSELoss()

        # --- STABILITY STRATEGY 3: Learning Rate Warmup ---
        NUM_EPOCHS = 200
        WARMUP_EPOCHS = 10
        TARGET_LR = 0.001

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)  # Start with a very low LR
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

        for epoch in range(NUM_EPOCHS):
            # Warmup phase
            if epoch < WARMUP_EPOCHS:
                lr = TARGET_LR * (epoch + 1) / WARMUP_EPOCHS
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            epoch_loss = 0
            model.train()
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            # Plateau scheduler takes over after warmup
            if epoch >= WARMUP_EPOCHS:
                plateau_scheduler.step(epoch_loss / len(dataloader))

        ensemble.append(model)
    print(f"--- Ensemble training finished for {model_name} ---")
    return ensemble


ensemble_inv_grad = train_ensemble(dataloader_ig, 2, "Model with Inverse Gradient")
ensemble_grad = train_ensemble(dataloader_g, 2, "Model with Gradient")
ensemble_control = train_ensemble(dataloader_c, 1, "Control Model")


# --- 5. Ensemble Prediction and Visualization ---
def predict_multistep_future(ensemble, initial_sequence, scaler, has_extra_feature, num_predict_steps):
    full_sequence_scaled = list(initial_sequence)
    for _ in range(num_predict_steps):
        current_sequence_np = np.array(full_sequence_scaled)
        current_sequence = torch.tensor(current_sequence_np, dtype=torch.float32).unsqueeze(0)
        predictions_scaled_step = []
        for model in ensemble:
            model.eval()
            with torch.no_grad():
                predictions_scaled_step.append(model(current_sequence).cpu().numpy()[0, 0])
        next_strain_pred_scaled = np.mean(predictions_scaled_step)
        if has_extra_feature:
            last_extra_feature_scaled = current_sequence[0, -1, 1].item()
            next_features_scaled = np.array([next_strain_pred_scaled, last_extra_feature_scaled])
        else:
            next_features_scaled = np.array([next_strain_pred_scaled])
        full_sequence_scaled.append(next_features_scaled)
    predicted_part_scaled = np.array(full_sequence_scaled)[len(initial_sequence):]
    # Ensure the output shape is always 2D for inverse_transform
    if predicted_part_scaled.ndim == 1:
        predicted_part_scaled = predicted_part_scaled.reshape(-1, 1)

    # Pad with a dummy feature if scaler expects more than 1
    if scaler.n_features_in_ > 1 and predicted_part_scaled.shape[1] == 1:
        dummy_features = np.zeros((predicted_part_scaled.shape[0], scaler.n_features_in_ - 1))
        padded_features = np.hstack((predicted_part_scaled, dummy_features))
        return scaler.inverse_transform(padded_features)

    return scaler.inverse_transform(predicted_part_scaled)


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

target_cycles = [100, 200, 500, 1000]

for point_flat_index in points_to_plot:
    try:
        point_valid_index = np.where(valid_indices_array == point_flat_index)[0][0]
        point_y, point_x = np.unravel_index(point_flat_index, (height, width))
    except IndexError:
        continue

    initial_sequence_ig = scaled_features_ig[point_valid_index, :9, :]
    initial_sequence_g = scaled_features_g[point_valid_index, :9, :]
    initial_sequence_c = scaled_features_c[point_valid_index, :9, :]

    last_known_cycle = all_cycle_numbers[8]
    cycle_step_approx = 10

    pred_2step_ig = predict_multistep_future(ensemble_inv_grad, initial_sequence_ig, scalers_ig[point_valid_index],
                                             True, 2)
    pred_2step_g = predict_multistep_future(ensemble_grad, initial_sequence_g, scalers_g[point_valid_index], True, 2)
    pred_2step_c = predict_multistep_future(ensemble_control, initial_sequence_c, scalers_c[point_valid_index], False,
                                            2)

    num_longterm_steps = int(np.ceil((max(target_cycles) - all_cycle_numbers[-1]) / cycle_step_approx))
    longterm_pred_ig = predict_multistep_future(ensemble_inv_grad, scaled_features_ig[point_valid_index, :11, :],
                                                scalers_ig[point_valid_index], True, num_longterm_steps)
    longterm_pred_g = predict_multistep_future(ensemble_grad, scaled_features_g[point_valid_index, :11, :],
                                               scalers_g[point_valid_index], True, num_longterm_steps)
    longterm_pred_c = predict_multistep_future(ensemble_control, scaled_features_c[point_valid_index, :11, :],
                                               scalers_c[point_valid_index], False, num_longterm_steps)

    known_history = valid_features_ig[point_valid_index, :, 0]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(all_cycle_numbers, known_history, 'o-', color='royalblue', label='Known History (Cycles 1-90)')

    ax.plot(all_cycle_numbers[9], pred_2step_ig[0, 0], '^', color='red', markersize=10)
    ax.plot(all_cycle_numbers[10], pred_2step_ig[1, 0], '^', color='red', markersize=10, label='Inv. Grad. Pred.')
    ax.plot(all_cycle_numbers[9], pred_2step_g[0, 0], '>', color='orange', markersize=10)
    ax.plot(all_cycle_numbers[10], pred_2step_g[1, 0], '>', color='orange', markersize=10, label='Grad. Pred.')
    ax.plot(all_cycle_numbers[9], pred_2step_c[0, 0], 'x', color='purple', markersize=10)
    ax.plot(all_cycle_numbers[10], pred_2step_c[1, 0], 'x', color='purple', markersize=10, label='Control Pred.')

    last_known_cycle_longterm = all_cycle_numbers[-1]
    last_known_strain_longterm = known_history[-1]
    longterm_forecast_cycles = [last_known_cycle_longterm] + [last_known_cycle_longterm + (i + 1) * cycle_step_approx
                                                              for i in range(num_longterm_steps)]

    ax.plot(longterm_forecast_cycles, np.concatenate(([last_known_strain_longterm], longterm_pred_ig[:, 0])), ':',
            color='red')
    ax.plot(longterm_forecast_cycles, np.concatenate(([last_known_strain_longterm], longterm_pred_g[:, 0])), ':',
            color='orange')
    ax.plot(longterm_forecast_cycles, np.concatenate(([last_known_strain_longterm], longterm_pred_c[:, 0])), ':',
            color='purple')

    for tc in target_cycles:
        idx = np.abs(np.array(longterm_forecast_cycles) - tc).argmin()
        if idx > 0:
            ax.plot(longterm_forecast_cycles[idx], longterm_pred_ig[idx - 1, 0], '*', color='red', markersize=12)
            ax.plot(longterm_forecast_cycles[idx], longterm_pred_g[idx - 1, 0], '*', color='orange', markersize=12)
            ax.plot(longterm_forecast_cycles[idx], longterm_pred_c[idx - 1, 0], '*', color='purple', markersize=12)

    plot_title = f'Long-Term Forecast for Point (y={point_y}, x={point_x})'
    ax.set_title(plot_title, fontsize=16)
    ax.set_xlabel('Cycle Number');
    ax.set_ylabel('Plastic Strain')
    ax.legend(loc='best', fontsize='small');
    ax.grid(True);
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'warmup_forecast_plot_y{point_y}_x{point_x}.png')
    plt.savefig(save_path, dpi=300);
    plt.close(fig)
    print(f"Saved long-term forecast plot to: {save_path}")

# --- 6. Generate Final Comparison Heatmaps for Cycle 90 ---
print("\nGenerating final comparison heatmaps for Cycle 90...")
heatmap_ig = np.full((height, width), np.nan)
heatmap_g = np.full((height, width), np.nan)
heatmap_c = np.full((height, width), np.nan)

for i in tqdm(range(len(valid_indices_array)), desc="Predicting all points for heatmaps"):
    point_flat_index = valid_indices_array[i]
    point_y, point_x = np.unravel_index(point_flat_index, (height, width))

    pred_ig_full = predict_multistep_future(ensemble_inv_grad, scaled_features_ig[i, :9, :], scalers_ig[i], True, 2)
    pred_g_full = predict_multistep_future(ensemble_grad, scaled_features_g[i, :9, :], scalers_g[i], True, 2)
    pred_c_full = predict_multistep_future(ensemble_control, scaled_features_c[i, :9, :], scalers_c[i], False, 2)

    heatmap_ig[point_y, point_x] = pred_ig_full[-1, 0]
    heatmap_g[point_y, point_x] = pred_g_full[-1, 0]
    heatmap_c[point_y, point_x] = pred_c_full[-1, 0]

actual_cycle_90_heatmap = all_strain_data[:, :, -1]
epsilon = 1e-8

error_ig = np.abs(heatmap_ig - actual_cycle_90_heatmap)
error_g = np.abs(heatmap_g - actual_cycle_90_heatmap)
error_c = np.abs(heatmap_c - actual_cycle_90_heatmap)

improvement_ig_vs_c = (error_c - error_ig) / (error_c + epsilon) * 100
improvement_g_vs_c = (error_c - error_g) / (error_c + epsilon) * 100
improvement_ig_vs_g = (error_g - error_ig) / (error_g + epsilon) * 100

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle('Pairwise Comparison of Prediction Accuracy Improvement for Cycle 90', fontsize=20)


def plot_improvement_map(ax, data, title, better_label, worse_label):
    im = ax.imshow(data, cmap='coolwarm', vmin=-100, vmax=100)
    ax.set_title(title, fontsize=16)
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08)
    cbar.set_label(f'Error Reduction (%) \n(Red = {better_label} is Better, Blue = {worse_label} is Better)',
                   fontsize=12)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


plot_improvement_map(axes[0], improvement_ig_vs_c, 'Inverse Gradient vs. Control', 'Inv. Grad.', 'Control')
plot_improvement_map(axes[1], improvement_g_vs_c, 'Gradient vs. Control', 'Gradient', 'Control')
plot_improvement_map(axes[2], improvement_ig_vs_g, 'Inverse Gradient vs. Gradient', 'Inv. Grad.', 'Gradient')

plt.tight_layout(rect=[0, 0, 1, 0.95])
heatmap_save_path = os.path.join(output_dir, 'warmup_feature_comparison.png')
plt.savefig(heatmap_save_path, dpi=300)
print(f"Saved final comparison heatmap to: {heatmap_save_path}")
plt.show(block=False)  # Use block=False to allow script to continue

# --- START: ADDED FOR CYCLE 1000 PREDICTION ---
# --- 7. Generate Prediction Heatmaps for Cycle 1000 ---
print("\nGenerating long-term prediction heatmaps for Cycle 1000...")

# Define target and calculate steps
target_cycle_heatmap = 1000
last_known_cycle_num = all_cycle_numbers[-1]
cycle_step_approx = 10  # This should match the step logic used in single point prediction
num_steps_to_1000 = int(np.ceil((target_cycle_heatmap - last_known_cycle_num) / cycle_step_approx))

# Initialize empty heatmaps
heatmap_1000_ig = np.full((height, width), np.nan)
heatmap_1000_g = np.full((height, width), np.nan)
heatmap_1000_c = np.full((height, width), np.nan)

# Loop through all valid points and perform long-term prediction
for i in tqdm(range(len(valid_indices_array)), desc="Predicting Cycle 1000 for all points"):
    point_flat_index = valid_indices_array[i]
    point_y, point_x = np.unravel_index(point_flat_index, (height, width))

    # Use the full history (all 11 known cycles) as the initial sequence for long-term forecast
    initial_sequence_ig = scaled_features_ig[i, :11, :]
    initial_sequence_g = scaled_features_g[i, :11, :]
    initial_sequence_c = scaled_features_c[i, :11, :]

    # Predict up to the target cycle
    pred_1000_ig = predict_multistep_future(ensemble_inv_grad, initial_sequence_ig, scalers_ig[i], True,
                                            num_steps_to_1000)
    pred_1000_g = predict_multistep_future(ensemble_grad, initial_sequence_g, scalers_g[i], True, num_steps_to_1000)
    pred_1000_c = predict_multistep_future(ensemble_control, initial_sequence_c, scalers_c[i], False, num_steps_to_1000)

    # Store the final predicted value
    heatmap_1000_ig[point_y, point_x] = pred_1000_ig[-1, 0]
    heatmap_1000_g[point_y, point_x] = pred_1000_g[-1, 0]
    heatmap_1000_c[point_y, point_x] = pred_1000_c[-1, 0]

# --- Visualize the results ---
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle('Predicted Strain Distribution at Cycle 1000', fontsize=20)

# Determine a common color scale for better comparison
vmin = np.nanmin([np.nanmin(heatmap_1000_ig), np.nanmin(heatmap_1000_g), np.nanmin(heatmap_1000_c)])
vmax = np.nanmax([np.nanmax(heatmap_1000_ig), np.nanmax(heatmap_1000_g), np.nanmax(heatmap_1000_c)])

# Plot for Inverse Gradient Model
im1 = axes[0].imshow(heatmap_1000_ig, cmap='viridis', vmin=vmin, vmax=vmax)
axes[0].set_title('Inverse Gradient Model Prediction', fontsize=16)
fig.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.08).set_label('Predicted Strain')
axes[0].axes.get_xaxis().set_visible(False)
axes[0].axes.get_yaxis().set_visible(False)

# Plot for Gradient Model
im2 = axes[1].imshow(heatmap_1000_g, cmap='viridis', vmin=vmin, vmax=vmax)
axes[1].set_title('Gradient Model Prediction', fontsize=16)
fig.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.08).set_label('Predicted Strain')
axes[1].axes.get_xaxis().set_visible(False)
axes[1].axes.get_yaxis().set_visible(False)

# Plot for Control Model
im3 = axes[2].imshow(heatmap_1000_c, cmap='viridis', vmin=vmin, vmax=vmax)
axes[2].set_title('Control Model Prediction', fontsize=16)
fig.colorbar(im3, ax=axes[2], orientation='horizontal', pad=0.08).set_label('Predicted Strain')
axes[2].axes.get_xaxis().set_visible(False)
axes[2].axes.get_yaxis().set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
heatmap_1000_save_path = os.path.join(output_dir, 'predicted_strain_cycle_1000.png')
plt.savefig(heatmap_1000_save_path, dpi=300)
print(f"Saved Cycle 1000 prediction heatmap to: {heatmap_1000_save_path}")
plt.show()

# --- END: ADDED FOR CYCLE 1000 PREDICTION ---

print("\nAll tasks completed successfully.")