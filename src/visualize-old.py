import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# Load the data
csb_ts_data = pd.read_csv("data/TS_NHF1_CPD_CSB_XR_1.bed.csv")
xpc_ts_data = pd.read_csv("data/TS_NHF1_CPD_XPC_XR_1.bed.csv")
csb_nts_data = pd.read_csv("data/NTS_NHF1_CPD_CSB_XR_1.bed.csv")
xpc_nts_data = pd.read_csv("data/NTS_NHF1_CPD_XPC_XR_1.bed.csv")
combined_ts_data = pd.read_csv("data/combined_TS_data.bed.csv")
combined_nts_data = pd.read_csv("data/combined_NTS_data.bed.csv")

# Coefficients
TS_coefficients = {"CSB": 0.18835342, "XPC": 0.81164658}
NTS_coefficients = {"CSB": 0.70594788, "XPC": 0.29405212}

# Create new columns for calculated counts
csb_ts_data['calc_counts'] = csb_ts_data['counts'] * TS_coefficients['CSB']
xpc_ts_data['calc_counts'] = xpc_ts_data['counts'] * TS_coefficients['XPC']
csb_nts_data['calc_counts'] = csb_nts_data['counts'] * NTS_coefficients['CSB']
xpc_nts_data['calc_counts'] = xpc_nts_data['counts'] * NTS_coefficients['XPC']

# Merge the calculated counts data
merged_ts_data = pd.merge(csb_ts_data[['gene_id', 'calc_counts']], xpc_ts_data[['gene_id', 'calc_counts']], on='gene_id', suffixes=('_csb', '_xpc'))
merged_nts_data = pd.merge(csb_nts_data[['gene_id', 'calc_counts']], xpc_nts_data[['gene_id', 'calc_counts']], on='gene_id', suffixes=('_csb', '_xpc'))

# Calculate MSE
mse_csb_ts = mean_squared_error(combined_ts_data['counts'], merged_ts_data['calc_counts_csb'])
mse_xpc_ts = mean_squared_error(combined_ts_data['counts'], merged_ts_data['calc_counts_xpc'])
mse_csb_nts = mean_squared_error(combined_nts_data['counts'], merged_nts_data['calc_counts_csb'])
mse_xpc_nts = mean_squared_error(combined_nts_data['counts'], merged_nts_data['calc_counts_xpc'])

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
x_vals = np.array([0, max(combined_ts_data['counts'].max(), combined_nts_data['counts'].max())])

# Define a function to set equal scaling for x and y axes
def set_equal_scaling(ax, x_data, y_data):
    min_value = min(x_data.min(), y_data.min())
    max_value = max(x_data.max(), y_data.max())
    ax.set_xlim([min_value, max_value])
    ax.set_ylim([min_value, max_value])

# CSB TS
axes[0, 0].scatter(combined_ts_data['counts'], merged_ts_data['calc_counts_csb'], alpha=0.5)
axes[0, 0].plot(x_vals, TS_coefficients["CSB"] * x_vals, 'k--')
axes[0, 0].set_title(f'CSB TS (Coefficient: {TS_coefficients["CSB"]}, MSE: {mse_csb_ts:.2f})')
axes[0, 0].set_xlabel('Combined Counts')
axes[0, 0].set_ylabel('CSB TS Counts * Coefficient')
set_equal_scaling(axes[0, 0], combined_ts_data['counts'], merged_ts_data['calc_counts_csb'])

# CSB NTS
axes[0, 1].scatter(combined_nts_data['counts'], merged_nts_data['calc_counts_csb'], alpha=0.5)
axes[0, 1].plot(x_vals, NTS_coefficients["CSB"] * x_vals, 'k--')
axes[0, 1].set_title(f'CSB NTS (Coefficient: {NTS_coefficients["CSB"]}, MSE: {mse_csb_nts:.2f})')
axes[0, 1].set_xlabel('Combined Counts')
axes[0, 1].set_ylabel('CSB NTS Counts * Coefficient')
set_equal_scaling(axes[0, 1], combined_nts_data['counts'], merged_nts_data['calc_counts_csb'])

# XPC TS
axes[1, 0].scatter(combined_ts_data['counts'], merged_ts_data['calc_counts_xpc'], alpha=0.5)
axes[1, 0].plot(x_vals, TS_coefficients["XPC"] * x_vals, 'k--')
axes[1, 0].set_title(f'XPC TS (Coefficient: {TS_coefficients["XPC"]}, MSE: {mse_xpc_ts:.2f})')
axes[1, 0].set_xlabel('Combined Counts')
axes[1, 0].set_ylabel('XPC TS Counts * Coefficient')
set_equal_scaling(axes[1, 0], combined_ts_data['counts'], merged_ts_data['calc_counts_xpc'])

# XPC NTS
axes[1, 1].scatter(combined_nts_data['counts'], merged_nts_data['calc_counts_xpc'], alpha=0.5)
axes[1, 1].plot(x_vals, NTS_coefficients["XPC"] * x_vals, 'k--')
axes[1, 1].set_title(f'XPC NTS (Coefficient: {NTS_coefficients["XPC"]}, MSE: {mse_xpc_nts:.2f})')
axes[1, 1].set_xlabel('Combined Counts')
axes[1, 1].set_ylabel('XPC NTS Counts * Coefficient')
set_equal_scaling(axes[1, 1], combined_nts_data['counts'], merged_nts_data['calc_counts_xpc'])

plt.tight_layout()
plt.show()

