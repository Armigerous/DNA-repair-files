# Importing necessary modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load the datasets
# This code reads csv files from the "data" directory into pandas DataFrames
csb_ts_data = pd.read_csv(r"data\TS_NHF1_CPD_CSB_XR_1.bed.csv")
xpc_ts_data = pd.read_csv(r"data\TS_NHF1_CPD_XPC_XR_1.bed.csv")
csb_nts_data = pd.read_csv(r"data\NTS_NHF1_CPD_CSB_XR_1.bed.csv")
xpc_nts_data = pd.read_csv(r"data\NTS_NHF1_CPD_XPC_XR_1.bed.csv")
ts_input_data = pd.read_csv(r"data\TS_NHF1_CPD_XR_1.bed.csv")
nts_input_data = pd.read_csv(r"data\NTS_NHF1_CPD_XR_1.bed.csv")  

# Calculate the length of each gene
# This step adds a new column "length" in each DataFrame, calculated as the difference between the "end" and "start" columns
csb_ts_data["length"] = csb_ts_data["end"] - csb_ts_data["start"]
xpc_ts_data["length"] = xpc_ts_data["end"] - xpc_ts_data["start"]
csb_nts_data["length"] = csb_nts_data["end"] - csb_nts_data["start"]
xpc_nts_data["length"] = xpc_nts_data["end"] - xpc_nts_data["start"]
ts_input_data["length"] = ts_input_data["end"] - ts_input_data["start"]
nts_input_data["length"] = nts_input_data["end"] - nts_input_data["start"]  

# Extract 'counts' values as target variables
# This step defines the target variables, which are the 'counts' in each DataFrame
csb_ts_counts = csb_ts_data["counts"]
xpc_ts_counts = xpc_ts_data["counts"]
csb_nts_counts = csb_nts_data["counts"]
xpc_nts_counts = xpc_nts_data["counts"]
 
# Prepare input data
# This step prepares the input data by selecting the 'counts' and 'length' columns from the input data
ts_input_features = ts_input_data[["counts", "length"]]
nts_input_features = nts_input_data[["counts", "length"]]  
X_ts = np.array(ts_input_features)
X_nts = np.array(nts_input_features)  

# Scale the input features to a range between 0 and 1
scaler = MinMaxScaler()
X_ts_scaled = scaler.fit_transform(X_ts)
X_nts_scaled = scaler.transform(X_nts)  

# Split data into training and testing sets
# This step splits the scaled input data and target variables into training and testing sets, with a 80-20 split
X_ts_train, X_ts_test = train_test_split(X_ts_scaled, test_size=0.2, random_state=42)
X_nts_train, X_nts_test = train_test_split(X_nts_scaled, test_size=0.2, random_state=42)
y_csb_ts_train, y_csb_ts_test = train_test_split(csb_ts_counts, test_size=0.2, random_state=42)
y_xpc_ts_train, y_xpc_ts_test = train_test_split(xpc_ts_counts, test_size=0.2, random_state=42)
y_csb_nts_train, y_csb_nts_test = train_test_split(csb_nts_counts, test_size=0.2, random_state=42)
y_xpc_nts_train, y_xpc_nts_test = train_test_split(xpc_nts_counts, test_size=0.2, random_state=42)

# Define the neural network model
# This function defines a neural network model with two hidden layers and one output layer
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=2, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    return model

# Train the CSB_TS model
csb_ts_model = create_model()
csb_ts_model.fit(X_ts_train, y_csb_ts_train, epochs=100, batch_size=16, verbose=0)

# Train the XPC_TS model
xpc_ts_model = create_model()
xpc_ts_model.fit(X_ts_train, y_xpc_ts_train, epochs=100, batch_size=16, verbose=0)

# Train the CSB_NTS model
csb_nts_model = create_model()
csb_nts_model.fit(X_nts_train, y_csb_nts_train, epochs=100, batch_size=16, verbose=0)  

# Train the XPC_NTS model
xpc_nts_model = create_model()
xpc_nts_model.fit(X_nts_train, y_xpc_nts_train, epochs=100, batch_size=16, verbose=0)  

# Predict counts values for test data
csb_ts_counts_pred = csb_ts_model.predict(X_ts_test)
xpc_ts_counts_pred = xpc_ts_model.predict(X_ts_test)
csb_nts_counts_pred = csb_nts_model.predict(X_nts_test)  # Now using NTS test data
xpc_nts_counts_pred = xpc_nts_model.predict(X_nts_test)  # Now using NTS test data

# Evaluate the performance of the models
csb_ts_mse = mean_squared_error(y_csb_ts_test, csb_ts_counts_pred)
csb_ts_r2 = r2_score(y_csb_ts_test, csb_ts_counts_pred)
xpc_ts_mse = mean_squared_error(y_xpc_ts_test, xpc_ts_counts_pred)
xpc_ts_r2 = r2_score(y_xpc_ts_test, xpc_ts_counts_pred)
csb_nts_mse = mean_squared_error(y_csb_nts_test, csb_nts_counts_pred)
csb_nts_r2 = r2_score(y_csb_nts_test, csb_nts_counts_pred)
xpc_nts_mse = mean_squared_error(y_xpc_nts_test, xpc_nts_counts_pred)
xpc_nts_r2 = r2_score(y_xpc_nts_test, xpc_nts_counts_pred)

print("CSB_TS Model: MSE = {}, R2 = {}".format(csb_ts_mse, csb_ts_r2))
print("XPC_TS Model: MSE = {}, R2 = {}".format(xpc_ts_mse, xpc_ts_r2))
print("CSB_NTS Model: MSE = {}, R2 = {}".format(csb_nts_mse, csb_nts_r2))
print("XPC_NTS Model: MSE = {}, R2 = {}".format(xpc_nts_mse, xpc_nts_r2))

# Create a scatter plot for the CSB_TS model's predicted vs. actual counts
plt.subplot(2, 2, 1)
plt.scatter(y_csb_ts_test, csb_ts_counts_pred, color='#c78c80')
plt.title('CSB_TS Model: Predicted vs. Actual counts')
plt.xlabel('Actual Counts')
plt.ylabel('Predicted Counts')

# Create a scatter plot for the XPC_TS model's predicted vs. actual counts
plt.subplot(2, 2, 2)
plt.scatter(y_xpc_ts_test, xpc_ts_counts_pred, color='#c78c80')
plt.title('XPC_TS Model: Predicted vs. Actual counts')
plt.xlabel('Actual Counts')
plt.ylabel('Predicted Counts')

# Create a scatter plot for the CSB_NTS model's predicted vs. actual counts
plt.subplot(2, 2, 3)
plt.scatter(y_csb_nts_test, csb_nts_counts_pred, color='#c78c80')
plt.title('CSB_NTS Model: Predicted vs. Actual counts')
plt.xlabel('Actual Counts')
plt.ylabel('Predicted Counts')

# Create a scatter plot for the XPC_NTS model's predicted vs. actual counts
plt.subplot(2, 2, 4)
plt.scatter(y_xpc_nts_test, xpc_nts_counts_pred, color='#c78c80')
plt.title('XPC_NTS Model: Predicted vs. Actual counts')
plt.xlabel('Actual Counts')
plt.ylabel('Predicted Counts')

# Adjust the layout for better visualization
plt.tight_layout()

# Show the figure with the plots
plt.show()