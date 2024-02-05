import pandas as pd

# Load the datasets
data = pd.read_csv(r"new_data\wt_curated_eren.bed.csv")

# Filter data based on the 'strand' column
positive_data = data[data['strand'] == '+']
negative_data = data[data['strand'] == '-']

# Save them as new CSV files
positive_data.to_csv(r"new_data\wt_positive_data.bed.csv", index=False)
negative_data.to_csv(r"new_data\wt_negative_data.bed.csv", index=False)
