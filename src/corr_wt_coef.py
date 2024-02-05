import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Load the original and estimated data
csb_alpha_data = pd.read_csv(r"new_data\CSB_alpha_eren.bed.csv")
xpc_alpha_data = pd.read_csv(r"new_data\XPC_alpha_eren.bed.csv")
csb_alpha_est_data = pd.read_csv(r"new_data\CSB_alpha_eren_estimated_beta.bed.csv")
xpc_alpha_est_data = pd.read_csv(r"new_data\XPC_alpha_eren_estimated_beta.bed.csv")

# Merge the original and estimated dataframes on the gene_id column
csb_merged = pd.merge(csb_alpha_data, csb_alpha_est_data, on='gene_id', suffixes=('_orig', '_est'))
xpc_merged = pd.merge(xpc_alpha_data, xpc_alpha_est_data, on='gene_id', suffixes=('_orig', '_est'))

# Calculate MSE R2 for CSB alpha counts
mse_csb = mean_squared_error(csb_merged['counts_orig'], csb_merged['counts_est'])
r2_csb = r2_score(csb_merged['counts_orig'], csb_merged['counts_est'])

# Calculate MSE R2 for XPC alpha counts
mse_xpc = mean_squared_error(xpc_merged['counts_orig'], xpc_merged['counts_est'])
r2_xpc = r2_score(xpc_merged['counts_orig'], xpc_merged['counts_est'])

print('CSB Alpha Counts:')
print('Mean Squared Error:', mse_csb)
print('R-squared:', r2_csb)

print('XPC Alpha Counts:')
print('Mean Squared Error:', mse_xpc)
print('R-squared:', r2_xpc)

# Scatter plots to compare the original and estimated counts
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(csb_merged['counts_orig'], csb_merged['counts_est'])
plt.xlabel('Original Counts')
plt.ylabel('Estimated Counts')
plt.title('CSB Alpha Counts: Original vs Estimated')

plt.subplot(1, 2, 2)
plt.scatter(xpc_merged['counts_orig'], xpc_merged['counts_est'])
plt.xlabel('Original Counts')
plt.ylabel('Estimated Counts')
plt.title('XPC Alpha Counts: Original vs Estimated')

plt.tight_layout()
plt.show()