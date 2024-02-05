import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Define a function that calculates the difference between the original and estimated counts
def residuals(coefs, counts_wt, counts_orig):
    counts_est = counts_wt * coefs
    return counts_orig - counts_est

# Load the data
csb_alpha_data = pd.read_csv(r"new_data\CSB_alpha_eren.bed.csv")
xpc_alpha_data = pd.read_csv(r"new_data\XPC_alpha_eren.bed.csv")
wt_data = pd.read_csv(r"new_data\wt_curated_eren.bed.csv")

# Use least squares to find the coefficients that minimize the residuals for CSB
initial_guess_csb = [1]
result_csb = least_squares(residuals, initial_guess_csb, args=(wt_data['counts'], csb_alpha_data['counts']))
coefs_csb = result_csb.x

# Use least squares to find the coefficients that minimize the residuals for XPC
initial_guess_xpc = [1]
result_xpc = least_squares(residuals, initial_guess_xpc, args=(wt_data['counts'], xpc_alpha_data['counts']))
coefs_xpc = result_xpc.x

# Calculate the estimated counts for CSB and XPC using the coefficients
wt_data['counts_csb_est'] = wt_data['counts'] * coefs_csb
wt_data['counts_xpc_est'] = wt_data['counts'] * coefs_xpc

# Create new dataframes for the estimated CSB and XPC data
csb_est_data = wt_data[['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand', 'counts_csb_est']].copy()
xpc_est_data = wt_data[['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand', 'counts_xpc_est']].copy()

# Rename the counts columns
csb_est_data.rename(columns={'counts_csb_est': 'counts'}, inplace=True)
xpc_est_data.rename(columns={'counts_xpc_est': 'counts'}, inplace=True)

# Save the estimated CSB and XPC data to CSV files
csb_est_data.to_csv(r"new_data\CSB_alpha_eren_estimated_beta.bed.csv", index=False)
xpc_est_data.to_csv(r"new_data\XPC_alpha_eren_estimated_beta.bed.csv", index=False)

# Prints
print('CSB Alpha Coefficients:', coefs_csb)
print('XPC Alpha Coefficients:', coefs_xpc)

# Create a figure with two subplots
fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(6, 8))

# Scatter plot for CSB
axs[0].scatter(wt_data['counts'], csb_alpha_data['counts'], color='#c78c80', label=' Actual')
axs[0].scatter(wt_data['counts'], wt_data['counts_csb_est'], color='#54545f', label='Estimated')
axs[0].set_title('CSB Alpha')
axs[0].legend()

# Scatter plot for XPC
axs[1].scatter(wt_data['counts'], xpc_alpha_data['counts'], color='#c78c80', label='Actual')
axs[1].scatter(wt_data['counts'], wt_data['counts_xpc_est'], color='#54545f', label='Estimated')
axs[1].set_title('XPC Alpha')
axs[1].legend()

plt.tight_layout()
plt.show()