import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Load the datasets
csb_data = pd.read_csv("data\TS_NHF1_CPD_XPC_XR_1.bed.csv")
xpc_data = pd.read_csv("data\TS_NHF1_CPD_XPC_XR_1.bed.csv")
input_data = pd.read_csv("data\TS_NHF1_CPD_XR_1.bed.csv")

# Extract 'counts' values as target variables
csb_counts = csb_data["counts"]
xpc_counts = xpc_data["counts"]

# Prepare input data
input_counts = input_data["counts"]
X = input_counts.values.reshape(-1, 1)

# Split data into training and testing setsß
X_train, X_test, y_csb_train, y_csb_test, y_xpc_train, y_xpc_test = train_test_split(X, csb_counts, xpc_counts, test_size=0.2, random_state=42)

# Train the models
csb_model = RandomForestRegressor(random_state=42).fit(X_train, y_csb_train)
xpc_model = RandomForestRegressor(random_state=42).fit(X_train, y_xpc_train)

# Predict counts values for test data
csb_counts_pred = csb_model.predict(X_test)
xpc_counts_pred = xpc_model.predict(X_test)

# Evaluate the performance of the models
csb_mse = mean_squared_error(y_csb_test, csb_counts_pred)
csb_r2 = r2_score(y_csb_test, csb_counts_pred)
xpc_mse = mean_squared_error(y_xpc_test, xpc_counts_pred)
xpc_r2 = r2_score(y_xpc_test, xpc_counts_pred)

print("CSB Model TS: MSE = {}, R2 = {}".format(csb_mse, csb_r2))
print("XPC Model TS: MSE = {}, R2 = {}".format(xpc_mse, xpc_r2))

# Create a figure and a set of subplots
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Scatter plot for CSB model
ax[0].scatter(y_csb_test, csb_counts_pred, alpha=0.5, color='#c78c80' )
ax[0].set_xlabel("Original CSB counts")
ax[0].set_ylabel("Predicted CSB counts")
ax[0].set_title("CSB Model: Original vs Predicted Counts")

# Scatter plot for XPC model
ax[1].scatter(y_xpc_test, xpc_counts_pred, alpha=0.5, color='#c78c80')
ax[1].set_xlabel("Original XPC counts")
ax[1].set_ylabel("Predicted XPC counts")
ax[1].set_title("XPC Model: Original vs Predicted Counts")

# Display the figure
plt.tight_layout()
plt.show()