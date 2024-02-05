import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
csb_data = pd.read_csv(r"new_data\CSB_data.bed.csv")
xpc_data = pd.read_csv(r"new_data\XPC_data.bed.csv")
tcr_data = pd.read_csv(r"new_data\TCR.bed.csv")
gr_data = pd.read_csv(r"new_data\GR.bed.csv")
wt_data = pd.read_csv(r"new_data\wt_curated_eren.bed.csv")

# Calculate length
csb_data["length"] = csb_data["end"] - csb_data["start"]


# Define input and output data
input_data = pd.concat([csb_data['counts'], xpc_data['counts'], csb_data['length']], axis=1)
output_data = pd.concat([tcr_data['counts'], gr_data['counts']], axis=1)

# Scale input, 0 to 1
scaler = MinMaxScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(input_data_scaled, output_data, test_size=0.2, random_state=42)

# Define the model
def create_model():
    model = Sequential()
    model.add(Dense(192, input_dim=3, activation="relu"))
    model.add(Dense(128, activation="elu"))
    model.add(Dense(64, activation="elu"))
    model.add(Dense(32, activation="elu"))
    model.add(Dense(2, activation="sigmoid"))
    model.compile(loss="mse", optimizer="RMSprop", metrics=["mae"])
    return model

# Create a model
model = create_model()

# Fit the model
model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=20, verbose=0)

# Predict
counts_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, counts_pred)
r2 = r2_score(y_test, counts_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)
