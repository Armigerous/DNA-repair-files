# Importing necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler 
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization

# Load the datasets
csb_ts_data = pd.read_csv(r"data\TS_NHF1_CPD_CSB_XR_1.bed.csv")
xpc_ts_data = pd.read_csv(r"data\TS_NHF1_CPD_XPC_XR_1.bed.csv")
csb_nts_data = pd.read_csv(r"data\NTS_NHF1_CPD_CSB_XR_1.bed.csv")
xpc_nts_data = pd.read_csv(r"data\NTS_NHF1_CPD_XPC_XR_1.bed.csv")
ts_input_data = pd.read_csv(r"data\TS_NHF1_CPD_XR_1.bed.csv")
nts_input_data = pd.read_csv(r"data\NTS_NHF1_CPD_XR_1.bed.csv")  

# Add a new feature 'length'
ts_input_data['length'] = ts_input_data['end'] - ts_input_data['start']
nts_input_data['length'] = nts_input_data['end'] - nts_input_data['start']

# Prepare input data using 'rpkm', 'counts', and 'length'
ts_input_features = ts_input_data[['rpkm', 'counts', 'length']]
nts_input_features = nts_input_data[['rpkm', 'counts', 'length']] 
X_ts = np.array(ts_input_features)
X_nts = np.array(nts_input_features)  

# Scale the input features to a range between 0 and 1
scaler = RobustScaler()
X_ts_scaled = scaler.fit_transform(X_ts)
X_nts_scaled = scaler.transform(X_nts)  

# Extract 'counts' values as target variables
csb_ts_counts = csb_ts_data["counts"]
xpc_ts_counts = xpc_ts_data["counts"]
csb_nts_counts = csb_nts_data["counts"]
xpc_nts_counts = xpc_nts_data["counts"]

# Histograms for the distribution of 'counts'
fig, axs = plt.subplots(2, 2, figsize=(14,10))
axs[0, 0].hist(csb_ts_counts, bins=50, color='blue', alpha=0.7)
axs[0, 0].set_title('csb_ts_counts')
axs[0, 1].hist(xpc_ts_counts, bins=50, color='red', alpha=0.7)
axs[0, 1].set_title('xpc_ts_counts')
axs[1, 0].hist(csb_nts_counts, bins=50, color='green', alpha=0.7)
axs[1, 0].set_title('csb_nts_counts')
axs[1, 1].hist(xpc_nts_counts, bins=50, color='purple', alpha=0.7)
axs[1, 1].set_title('xpc_nts_counts')
plt.tight_layout()
plt.show()

# Split data into training and testing sets
X_ts_train, X_ts_test, y_csb_ts_train, y_csb_ts_test = train_test_split(X_ts_scaled, csb_ts_counts, test_size=0.2, random_state=42)
X_nts_train, X_nts_test, y_csb_nts_train, y_csb_nts_test = train_test_split(X_nts_scaled, csb_nts_counts, test_size=0.2, random_state=42)
_, _, y_xpc_ts_train, y_xpc_ts_test = train_test_split(X_ts_scaled, xpc_ts_counts, test_size=0.2, random_state=42)
_, _, y_xpc_nts_train, y_xpc_nts_test = train_test_split(X_nts_scaled, xpc_nts_counts, test_size=0.2, random_state=42)

# Neural network model
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation="elu", kernel_regularizer=l2(0.01))) 
    model.add(Dropout(0.3)) 
    model.add(Dense(128, activation="elu", kernel_regularizer=l2(0.01))) 
    model.add(Dropout(0.3)) 
    model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.01)))
    model.add(Dense(32, activation="relu", kernel_regularizer=l2(0.01))) 
    model.add(BatchNormalization())  
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    return model

# Create learning rate reduction
lr_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                 patience=5, 
                                 verbose=1, 
                                 factor=0.5, 
                                 min_lr=0.00001)

# Train the models and save history
csb_ts_model = create_model(X_ts_train.shape[1])
csb_ts_history = csb_ts_model.fit(X_ts_train, y_csb_ts_train, epochs=100, batch_size=64, verbose=0, validation_split=0.2, callbacks=[lr_reduction])

xpc_ts_model = create_model(X_ts_train.shape[1])
xpc_ts_history = xpc_ts_model.fit(X_ts_train, y_xpc_ts_train, epochs=100, batch_size=64, verbose=0, validation_split=0.2, callbacks=[lr_reduction])

csb_nts_model = create_model(X_nts_train.shape[1])
csb_nts_history = csb_nts_model.fit(X_nts_train, y_csb_nts_train, epochs=100, batch_size=64, verbose=0, validation_split=0.2, callbacks=[lr_reduction])  

xpc_nts_model = create_model(X_nts_train.shape[1])
xpc_nts_history = xpc_nts_model.fit(X_nts_train, y_xpc_nts_train, epochs=100, batch_size=64, verbose=0, validation_split=0.2, callbacks=[lr_reduction])  

# Plot training and validation loss over epochs for each model
fig, axs = plt.subplots(2, 2, figsize=(14,10))
axs[0, 0].plot(csb_ts_history.history['loss'], label='Training loss')
axs[0, 0].plot(csb_ts_history.history['val_loss'], label='Validation loss')
axs[0, 0].set_title('CSB TS Model Training and Validation Loss')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()

axs[0, 1].plot(xpc_ts_history.history['loss'], label='Training loss')
axs[0, 1].plot(xpc_ts_history.history['val_loss'], label='Validation loss')
axs[0, 1].set_title('XPC TS Model Training and Validation Loss')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()

axs[1, 0].plot(csb_nts_history.history['loss'], label='Training loss')
axs[1, 0].plot(csb_nts_history.history['val_loss'], label='Validation loss')
axs[1, 0].set_title('CSB NTS Model Training and Validation Loss')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].legend()

axs[1, 1].plot(xpc_nts_history.history['loss'], label='Training loss')
axs[1, 1].plot(xpc_nts_history.history['val_loss'], label='Validation loss')
axs[1, 1].set_title('XPC NTS Model Training and Validation Loss')
axs[1, 1].set_xlabel('Epochs')
axs[1, 1].set_ylabel('Loss')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# Predict counts values for test data
csb_ts_counts_pred = csb_ts_model.predict(X_ts_test)
xpc_ts_counts_pred = xpc_ts_model.predict(X_ts_test)
csb_nts_counts_pred = csb_nts_model.predict(X_nts_test)
xpc_nts_counts_pred = xpc_nts_model.predict(X_nts_test)

# Evaluate the performance of the models
csb_ts_mse = mean_squared_error(y_csb_ts_test, csb_ts_counts_pred)
csb_ts_r2 = r2_score(y_csb_ts_test, csb_ts_counts_pred)
xpc_ts_mse = mean_squared_error(y_xpc_ts_test, xpc_ts_counts_pred)
xpc_ts_r2 = r2_score(y_xpc_ts_test, xpc_ts_counts_pred)
csb_nts_mse = mean_squared_error(y_csb_nts_test, csb_nts_counts_pred)
csb_nts_r2 = r2_score(y_csb_nts_test, csb_nts_counts_pred)
xpc_nts_mse = mean_squared_error(y_xpc_nts_test, xpc_nts_counts_pred)
xpc_nts_r2 = r2_score(y_xpc_nts_test, xpc_nts_counts_pred)

#print the results
print("CSB_TS (TCR) Model: MSE = {}, R2 = {}".format(csb_ts_mse, csb_ts_r2))
print("XPC_TS (GR) Model: MSE = {}, R2 = {}".format(xpc_ts_mse, xpc_ts_r2))
print("CSB_NTS (TCR) Model: MSE = {}, R2 = {}".format(csb_nts_mse, csb_nts_r2))
print("XPC_NTS (GR) Model: MSE = {}, R2 = {}".format(xpc_nts_mse, xpc_nts_r2))

# Scatter plot of the true vs predicted counts
fig, axs = plt.subplots(2, 2, figsize=(14,10))
axs[0, 0].scatter(y_csb_ts_test, csb_ts_counts_pred, alpha=0.7)
axs[0, 0].plot([y_csb_ts_test.min(), y_csb_ts_test.max()], [y_csb_ts_test.min(), y_csb_ts_test.max()], 'k--', lw=3)
axs[0, 0].set_title('CSB TS Model: True vs Predicted Counts')
axs[0, 0].set_xlabel('True Counts')
axs[0, 0].set_ylabel('Predicted Counts')

axs[0, 1].scatter(y_xpc_ts_test, xpc_ts_counts_pred, alpha=0.7)
axs[0, 1].plot([y_xpc_ts_test.min(), y_xpc_ts_test.max()], [y_xpc_ts_test.min(), y_xpc_ts_test.max()], 'k--', lw=3)
axs[0, 1].set_title('XPC TS Model: True vs Predicted Counts')
axs[0, 1].set_xlabel('True Counts')
axs[0, 1].set_ylabel('Predicted Counts')

axs[1, 0].scatter(y_csb_nts_test, csb_nts_counts_pred, alpha=0.7)
axs[1, 0].plot([y_csb_nts_test.min(), y_csb_nts_test.max()], [y_csb_nts_test.min(), y_csb_nts_test.max()], 'k--', lw=3)
axs[1, 0].set_title('CSB NTS Model: True vs Predicted Counts')
axs[1, 0].set_xlabel('True Counts')
axs[1, 0].set_ylabel('Predicted Counts')

axs[1, 1].scatter(y_xpc_nts_test, xpc_nts_counts_pred, alpha=0.7)
axs[1, 1].plot([y_xpc_nts_test.min(), y_xpc_nts_test.max()], [y_xpc_nts_test.min(), y_xpc_nts_test.max()], 'k--', lw=3)
axs[1, 1].set_title('XPC NTS Model: True vs Predicted Counts')
axs[1, 1].set_xlabel('True Counts')
axs[1, 1].set_ylabel('Predicted Counts')

plt.tight_layout()
plt.show()
