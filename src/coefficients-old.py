import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

# Load datasets
csb_ts_data = pd.read_csv(r"data\TS_NHF1_CPD_CSB_XR_1.bed.csv")
xpc_ts_data = pd.read_csv(r"data\TS_NHF1_CPD_XPC_XR_1.bed.csv")
csb_nts_data = pd.read_csv(r"data\NTS_NHF1_CPD_CSB_XR_1.bed.csv")
xpc_nts_data = pd.read_csv(r"data\NTS_NHF1_CPD_XPC_XR_1.bed.csv")
combined_ts_data = pd.read_csv(r"data\combined_TS_data.bed.csv")
combined_nts_data = pd.read_csv(r"data\combined_NTS_data.bed.csv")

# Merge the datasets on gene_id, start, and end
merged_ts_data = pd.merge(csb_ts_data, xpc_ts_data, on=['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand'], suffixes=('_csb', '_xpc'))
merged_ts_data = pd.merge(merged_ts_data, combined_ts_data, on=['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand'])
merged_nts_data = pd.merge(csb_nts_data, xpc_nts_data, on=['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand'], suffixes=('_csb', '_xpc'))
merged_nts_data = pd.merge(merged_nts_data, combined_nts_data, on=['chrom', 'start', 'end', 'gene_id', 'gene_name', 'strand'])

# Define the feature and target variables
X_ts = merged_ts_data[['counts']]
y_ts = merged_ts_data[['counts_csb', 'counts_xpc']]
X_nts = merged_nts_data[['counts']]
y_nts = merged_nts_data[['counts_csb', 'counts_xpc']]

# Split the data into training and test sets
X_ts_train, X_ts_test, y_ts_train, y_ts_test = train_test_split(X_ts, y_ts, test_size=0.2, random_state=42)
X_nts_train, X_nts_test, y_nts_train, y_nts_test = train_test_split(X_nts, y_nts, test_size=0.2, random_state=42)

# Train the MultiOutputRegressor model
model_ts = MultiOutputRegressor(LinearRegression()).fit(X_ts_train, y_ts_train)
model_nts = MultiOutputRegressor(LinearRegression()).fit(X_nts_train, y_nts_train)

# Print the coefficients
print('TS Coefficients: CSB:', model_ts.estimators_[0].coef_, 'XPC:', model_ts.estimators_[1].coef_)
print('NTS Coefficients: CSB:', model_nts.estimators_[0].coef_, 'XPC:', model_nts.estimators_[1].coef_)
