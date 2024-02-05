import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Alpha and Beta values
alpha = .77
beta = .81

# Read the WT data
wt_data = pd.read_csv(r"new_data\CPDs\TOTAL_NHF1_CPD_XR_1_minus.bed_in_windows.bed")

# Getting alpha values from WT
def create_alphas():
    
    # Calculate the estimated counts for CSB and XPC using the coefficients
    wt_data['counts_csb_est'] = wt_data['counts'] * beta
    wt_data['counts_xpc_est'] = wt_data['counts'] * (1 - beta)

    # Create new dataframes for the estimated CSB and XPC data
    csb_est_data = wt_data[['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand', 'counts_csb_est']].copy()
    xpc_est_data = wt_data[['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand', 'counts_xpc_est']].copy()

    # Rename the counts columns
    csb_est_data.rename(columns={'counts_csb_est': 'counts'}, inplace=True)
    xpc_est_data.rename(columns={'counts_xpc_est': 'counts'}, inplace=True)

    # Save the estimated CSB and XPC data to CSV files
    csb_est_data.to_csv(r"new_data\CPDs\test_CSBalpha.bed.csv", index=False)
    xpc_est_data.to_csv(r"new_data\CPDs\test_XPCalpha.bed.csv", index=False)

    # Print confirmation
    print("created CSB/XPC alpha files with beta:", beta)

# Turn alpha values into mutates
def get_mutates():

    # Read alpha values
    csb_alpha_data = pd.read_csv(r"new_data\test_CSBalpha.bed.csv")
    xpc_alpha_data = pd.read_csv(r"new_data\test_XPCalpha.bed.csv")

    # Calculate the estimated counts for CSB and XPC using the coefficients
    csb_alpha_data['counts_est'] = csb_alpha_data['counts'] / alpha
    xpc_alpha_data['counts_est'] = xpc_alpha_data['counts'] / (1 - alpha)
    
    # Create new dataframes for the estimated CSB and XPC data
    csb_data = csb_alpha_data[['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand', 'counts_est']].copy()
    xpc_data = xpc_alpha_data[['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand', 'counts_est']].copy()

    # Rename the counts columns
    csb_data.rename(columns={'counts_est': 'counts'}, inplace=True)
    xpc_data.rename(columns={'counts_est': 'counts'}, inplace=True)

    # Save the estimated CSB and XPC data to CSV files
    csb_data.to_csv(r"new_data\test_CSB.bed.csv", index=False)
    xpc_data.to_csv(r"new_data\test_XPC.bed.csv", index=False)
    
    # Print confirmation
    print("created CSB/XPC files with alpha:", alpha)

def split_ts_nts():
    # Load the data
    CSB_data = pd.read_csv(r"new_data\test_CSB.bed.csv") # Adjusted paths
    XPC_data = pd.read_csv(r"new_data\test_XPC.bed.csv") # Adjusted paths
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

    # Get the coefficients
    csb_ts_coef = csb_ts_model.coef_[0]
    xpc_ts_coef = xpc_ts_model.coef_[0]
    csb_nts_coef = csb_nts_model.coef_[0]
    xpc_nts_coef = xpc_nts_model.coef_[0]

    # Print the coefficients
    print("CSB TS coefficient:", csb_ts_coef)
    print("XPC TS coefficient:", xpc_ts_coef)
    print("CSB NTS coefficient:", csb_nts_coef)
    print("XPC NTS coefficient:", xpc_nts_coef)

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


create_alphas()
get_mutates()
split_ts_nts()