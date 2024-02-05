import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load the data
CSB_data = pd.read_csv(r"new_data\CSB_data.bed.csv")
XPC_data = pd.read_csv(r"new_data\XPC_data.bed.csv")
csb_ts_data = pd.read_csv("data/TS_NHF1_CPD_CSB_XR_1.bed.csv")
xpc_ts_data = pd.read_csv("data/TS_NHF1_CPD_XPC_XR_1.bed.csv")
csb_nts_data = pd.read_csv("data/NTS_NHF1_CPD_CSB_XR_1.bed.csv")
xpc_nts_data = pd.read_csv("data/NTS_NHF1_CPD_XPC_XR_1.bed.csv")

# Extract the 'counts' columns
CSB_counts = CSB_data['counts'].values.reshape(-1, 1)
XPC_counts = XPC_data['counts'].values.reshape(-1, 1)
csb_ts_counts = csb_ts_data['counts'].values
xpc_ts_counts = xpc_ts_data['counts'].values
csb_nts_counts = csb_nts_data['counts'].values
xpc_nts_counts = xpc_nts_data['counts'].values

# Create and train the models
csb_ts_model = LinearRegression().fit(CSB_counts, csb_ts_counts)
xpc_ts_model = LinearRegression().fit(XPC_counts, xpc_ts_counts)
csb_nts_model = LinearRegression().fit(CSB_counts, csb_nts_counts)
xpc_nts_model = LinearRegression().fit(XPC_counts, xpc_nts_counts)

# Create new dataframes with the predicted counts
CSB_data['new_counts_ts'] = csb_ts_model.predict(CSB_counts)
XPC_data['new_counts_ts'] = xpc_ts_model.predict(XPC_counts)

CSB_data['new_counts_nts'] = csb_nts_model.predict(CSB_counts)
XPC_data['new_counts_nts'] = xpc_nts_model.predict(XPC_counts)

# Calculate the predictions
csb_ts_pred = csb_ts_model.predict(CSB_counts)
xpc_ts_pred = xpc_ts_model.predict(XPC_counts)
csb_nts_pred = csb_nts_model.predict(CSB_counts)
xpc_nts_pred = xpc_nts_model.predict(XPC_counts)

# Replace the 'counts' column with the new predicted counts
CSB_data['counts'] = CSB_data['new_counts_ts']
XPC_data['counts'] = XPC_data['new_counts_ts']

# Save the new dataframes to CSV files
CSB_data[['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand', 'counts']].to_csv('new_data/test_CSB_TS_data.bed.csv', index=False)
XPC_data[['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand', 'counts']].to_csv('new_data/test_XPC_TS_data.bed.csv', index=False)

# Replace the 'counts' column with the new predicted counts
CSB_data['counts'] = CSB_data['new_counts_nts']
XPC_data['counts'] = XPC_data['new_counts_nts']

# Save the new dataframes to CSV files
CSB_data[['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand', 'counts']].to_csv('new_data/test_CSB_NTS_data.bed.csv', index=False)
XPC_data[['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand', 'counts']].to_csv('new_data/test_XPC_NTS_data.bed.csv', index=False)

# Calculate the R2 scores
csb_ts_r2 = r2_score(csb_ts_counts, csb_ts_pred)
xpc_ts_r2 = r2_score(xpc_ts_counts, xpc_ts_pred)
csb_nts_r2 = r2_score(csb_nts_counts, csb_nts_pred)
xpc_nts_r2 = r2_score(xpc_nts_counts, xpc_nts_pred)

print("___________________________________________________")
# Print the R2 scores
print("CSB TS R2 Score:", csb_ts_r2)
print("XPC TS R2 Score:", xpc_ts_r2)
print("CSB NTS R2 Score:", csb_nts_r2)
print("XPC NTS R2 Score:", xpc_nts_r2)

# Visualization
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plot for CSB TS
axs[0, 0].scatter(CSB_counts, csb_ts_counts, color='#c78c80', label='Original')
axs[0, 0].scatter(CSB_counts, CSB_data['new_counts_ts'], color='#54545f', label='Predicted')
axs[0, 0].set_title('CSB TS')
axs[0, 0].legend()

# Plot for XPC TS
axs[0, 1].scatter(XPC_counts, xpc_ts_counts, color='#c78c80', label='Original')
axs[0, 1].scatter(XPC_counts, XPC_data['new_counts_ts'], color='#54545f', label='Predicted')
axs[0, 1].set_title('XPC TS')
axs[0, 1].legend()

# Plot for CSB NTS
axs[1, 0].scatter(CSB_counts, csb_nts_counts, color='#c78c80', label='Original')
axs[1, 0].scatter(CSB_counts, CSB_data['new_counts_nts'], color='#54545f', label='Predicted')
axs[1, 0].set_title('CSB NTS')
axs[1, 0].legend()

# Plot for XPC NTS
axs[1, 1].scatter(XPC_counts, xpc_nts_counts, color='#c78c80', label='Original')
axs[1, 1].scatter(XPC_counts, XPC_data['new_counts_nts'], color='#54545f', label='Predicted')
axs[1, 1].set_title('XPC NTS')
axs[1, 1].legend()

plt.tight_layout()
plt.show()