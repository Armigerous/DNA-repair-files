# Mean Squared Error: 19516.453855919703
# R2 Score: 0.8489967462250683
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

# _______________________________________________________________
# Mean Squared Error: 17908.1766542455
# R2 Score: 0.8614403536661395
def create_model():
    model = Sequential()
    model.add(Dense(128, input_dim=2, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    return model

# _______________________________________________________________
# Mean Squared Error: 18965.05140075734
# R2 Score: 0.8532630727556817
def create_model():
    model = Sequential()
    model.add(Dense(128, input_dim=2, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    return model
# _______________________________________________________________
# Mean Squared Error: 30722.501786148565
# R2 Score: 0.7622929980997839
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=2, activation="tanh"))
    model.add(Dense(32, activation="tanh"))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    return model

# _______________________________________________________________
# Mean Squared Error: 129246.56820824853
# R2 Score: -1.017002222547525e-05
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=2, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="sgd", metrics=["mae"])
    return model

# _______________________________________________________________
# Mean Squared Error: 17110.226394142148
# R2 Score: 0.8676142767833052
input_data = pd.concat([xpc_data["counts"], csb_data["counts"], combined_data["length"]], axis=1)

def create_model():
    model = Sequential()
    model.add(Dense(128, input_dim=3, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    return model

# _______________________________________________________________
# Mean Squared Error: 15703.115761518837
# R2 Score: 0.8785014126080838
def create_model():
    model = Sequential()
    model.add(Dense(128, input_dim=3, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="RMSprop", metrics=["mae"])
    return model

# _______________________________________________________________
# Mean Squared Error: 3292.983489858132
# R2 Score: 0.8347346226204443
# Removing outliers using IQR method
Q1 = input_data.quantile(0.25)
Q3 = input_data.quantile(0.75)
IQR = Q3 - Q1
input_data = input_data[~((input_data < (Q1 - 1.5 * IQR)) | (input_data > (Q3 + 1.5 * IQR))).any(axis=1)]
output_data = output_data[input_data.index] 

# _______________________________________________________________
# CSB Alpha Counts:
# Mean Squared Error: 11857.213359810155
# R-squared: 0.934068887769126
# XPC Alpha Counts:
# Mean Squared Error: 11857.213359810155
# R-squared: -0.04961646479121695
# Create a series for the dependent variable (WT counts)
y = merged_data['counts_wt']

# Calculate the average ratio of counts_csb to counts_wt
average_ratio_csb = (merged_data['counts_csb'] / merged_data['counts_wt']).mean()

# Calculate the average ratio of counts_xpc to counts_wt
average_ratio_xpc = (merged_data['counts_xpc'] / merged_data['counts_wt']).mean()
