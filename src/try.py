import pandas as pd

# Read the dataset
data = pd.read_csv(r"new_data\GR.bed.csv")

# Remove the last 2 columns
data = data.iloc[:, :-2]

# Save the modified dataset
data.to_csv(r"new_data\GR.bed.csv", index=False)
