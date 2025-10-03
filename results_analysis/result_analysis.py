#%% Import libraries
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Load refit results data
# Change this path to the specific result file you want to analyze
result_file = '../results/refit/refit-elman-20251002-145339.json'

with open(result_file, 'r') as f:
    data = json.load(f)

print(f"Analyzing results from: {result_file}")
print(f"Model type: {data['model_type']}")

#%% Display hyperparameters table
params_data = []
for key, value in data['params'].items():
    params_data.append({'Parameter': key, 'Value': value})

# Add additional training info
params_data.extend([
    {'Parameter': 'fixed_epochs', 'Value': data['fixed_epochs']},
    {'Parameter': 'model_type', 'Value': data['model_type']}
])

params_df = pd.DataFrame(params_data)
print("\nHyperparameters and Training Configuration:")
print("=" * 50)
print(params_df.to_string(index=False))

#%% Display performance metrics
print("\nPerformance Metrics (Original Scale):")
print("=" * 40)
metrics = data['metrics_original']
for metric, value in metrics.items():
    print(f"{metric.upper()}: {value:.6f}")

#%% Plot loss curve
plt.figure(figsize=(12, 7))
train_loss = data['loss_curve']['train']
epochs = range(1, len(train_loss) + 1)

# Plot training loss with enhanced styling
plt.plot(epochs, train_loss, 'b-', linewidth=2.5, marker='o', markersize=5, 
         label='Training Loss', alpha=0.8)

# Add trend line
z = np.polyfit(epochs, train_loss, 1)
p = np.poly1d(z)
plt.plot(epochs, p(epochs), "r--", alpha=0.6, linewidth=1.5, label='Trend Line')

plt.title(f'{data["model_type"].upper()} - Training Loss Curve', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Loss (MSE)', fontsize=13)
plt.grid(True, alpha=0.4, linestyle='--')
plt.legend(fontsize=12, loc='upper right')

# Enhanced annotations
final_loss = train_loss[-1]
initial_loss = train_loss[0]
min_loss = min(train_loss)
min_epoch = train_loss.index(min_loss) + 1

# Final loss annotation
plt.annotate(f'Final: {final_loss:.6f}', 
             xy=(len(train_loss), final_loss), 
             xytext=(len(train_loss)*0.8, final_loss*1.3),
             arrowprops=dict(arrowstyle='->', color='red', alpha=0.8),
             fontsize=11, color='red', fontweight='bold')

# Best loss annotation
plt.annotate(f'Best: {min_loss:.6f}\n(Epoch {min_epoch})', 
             xy=(min_epoch, min_loss), 
             xytext=(min_epoch*1.3, min_loss*1.5),
             arrowprops=dict(arrowstyle='->', color='green', alpha=0.8),
             fontsize=11, color='green', fontweight='bold')

# Add improvement percentage
improvement = ((initial_loss - final_loss) / initial_loss) * 100
plt.text(0.02, 0.98, f'Loss Reduction: {improvement:.1f}%\nTotal Epochs: {len(train_loss)}', 
         transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.show()

#%% Plot actual vs predicted time series
# Convert timestamps to datetime for better plotting
timestamps = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in data['overlay']['time']]
y_true = data['overlay']['y_true']
y_pred = data['overlay']['y_pred']

plt.figure(figsize=(15, 8))

# Plot full time series
plt.plot(timestamps, y_true, label='Actual', color='blue', linewidth=2, alpha=0.8)
plt.plot(timestamps, y_pred, label='Predicted', color='red', linewidth=2, alpha=0.8)

plt.title(f'{data["model_type"].upper()} - Actual vs Predicted Time Series', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Add performance metrics as text box
textstr = f'RMSE: {metrics["rmse"]:.3f}\nMAE: {metrics["mae"]:.3f}\nSMAPE: {metrics["smape"]:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

#%% Plot residuals analysis
residuals = np.array(y_true) - np.array(y_pred)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Residuals over time
ax1.plot(timestamps, residuals, color='red', alpha=0.7, linewidth=1)
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.8)
ax1.set_title(f'{data["model_type"].upper()} - Residuals Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Add residual statistics
residual_mean = np.mean(residuals)
residual_std = np.std(residuals)
ax1.text(0.02, 0.98, f'Mean: {residual_mean:.3f}\nStd: {residual_std:.3f}', 
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Residuals histogram
ax2.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Residual Value', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%% Scatter plot: Actual vs Predicted
plt.figure(figsize=(10, 8))
plt.scatter(y_true, y_pred, alpha=0.6, s=20, color='blue')

# Perfect prediction line
min_val = min(min(y_true), min(y_pred))
max_val = max(max(y_true), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

plt.title(f'{data["model_type"].upper()} - Actual vs Predicted Scatter Plot', fontsize=14, fontweight='bold')
plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Calculate and display R²
correlation_matrix = np.corrcoef(y_true, y_pred)
r_squared = correlation_matrix[0, 1] ** 2
plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.show()

print(f"\nAnalysis complete for {data['model_type']} model!")
print(f"Total predictions analyzed: {len(y_true)}")
print(f"Date range: {timestamps[0].strftime('%Y-%m-%d')} to {timestamps[-1].strftime('%Y-%m-%d')}")
