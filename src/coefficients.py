import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Read datasets
tcr1_data = pd.read_csv(r"data\TCR1.bed.csv")
tcr0_data = pd.read_csv(r"data\TCR0.bed.csv")

gr1_data = pd.read_csv(r"data\NTS_NHF1_CPD_CSB_XR_1.bed.csv")
gr0_data = pd.read_csv(r"data\NTS_NHF1_CPD_XR_1.bed.csv")

# Drop the unnecessary columns from tcr0_data and gr0_data
gr1_data = gr1_data.drop(columns=["rpkm", "log2_rpkm"])
gr0_data = gr0_data.drop(columns=["rpkm", "log2_rpkm"])

# Merge the datasets
tcr_data = pd.merge(tcr1_data, tcr0_data, on=["chrom","start","end","gene_id","gene_name","strand"])
gr_data = pd.merge(gr1_data, gr0_data, on=["chrom","start","end","gene_id","gene_name","strand"])

# Prepare data for linear regression
tcr_X = tcr_data['counts_x'].values.reshape(-1, 1)  # values from tcr1
tcr_y = tcr_data['counts_y']  # values from tcr0

gr_X = gr_data['counts_x'].values.reshape(-1, 1)  # values from gr1
gr_y = gr_data['counts_y']  # values from gr0

# Fit linear regression model for TCR
tcr_regressor = LinearRegression()
tcr_regressor.fit(tcr_X, tcr_y)

# Fit linear regression model for GR
gr_regressor = LinearRegression()
gr_regressor.fit(gr_X, gr_y)

# Print the coefficients
print("TCR Coefficient: ", tcr_regressor.coef_[0])
print("GR Coefficient: ", gr_regressor.coef_[0])

# Make predictions with the TCR model
tcr_predictions = tcr_regressor.predict(tcr_X)

# Plot the original data in blue and the predictions in red
plt.scatter(tcr_X, tcr_y, color='blue', label='Original data')
plt.scatter(tcr_X, tcr_predictions, color='red', label='Predicted data')

plt.title('TCR data')
plt.xlabel('TCR1')
plt.ylabel('TCR0')
plt.legend()

plt.show()

# Make predictions with the GR model
gr_predictions = gr_regressor.predict(gr_X)

# Plot the original data (in blue) and the predictions (in red)
plt.scatter(gr_X, gr_y, color='blue', label='Original data')
plt.scatter(gr_X, gr_predictions, color='red', label='Predicted data')

plt.title('GR data')
plt.xlabel('GR1')
plt.ylabel('GR0')
plt.legend()

plt.show()

# Compute MSE, MAE, and R^2 for the TCR model
tcr_mse = mean_squared_error(tcr_y, tcr_predictions)
tcr_mae = mean_absolute_error(tcr_y, tcr_predictions)
tcr_r2 = r2_score(tcr_y, tcr_predictions)

print('TCR Mean Squared Error:', tcr_mse)
print('TCR Mean Absolute Error:', tcr_mae)
print('TCR R-squared:', tcr_r2)

# Compute MSE, MAE, and R^2 for the GR model
gr_mse = mean_squared_error(gr_y, gr_predictions)
gr_mae = mean_absolute_error(gr_y, gr_predictions)
gr_r2 = r2_score(gr_y, gr_predictions)

print('GR Mean Squared Error:', gr_mse)
print('GR Mean Absolute Error:', gr_mae)
print('GR R-squared:', gr_r2)