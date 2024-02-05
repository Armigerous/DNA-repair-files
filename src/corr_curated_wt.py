import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Directory containing the files
directory = "new_data"

# List of file names
file_names = [
    "wt_curated_eren.bed.csv",
    "WT_data.bed.csv",    
]

# Dictionary to hold file data
data_dict = {}

# Read data from each file into a pandas DataFrame and store it in the dictionary
for file in file_names:
    data_dict[file] = pd.read_csv(f"{directory}/{file}")["counts"]

# Create DataFrame for correlation matrices
pearson_df = pd.DataFrame(index=file_names, columns=file_names)
spearman_df = pd.DataFrame(index=file_names, columns=file_names)

# Calculate correlation
for file1 in file_names:
    for file2 in file_names:
        pearson_corr, _ = stats.pearsonr(data_dict[file1], data_dict[file2])
        spearman_corr, _ = stats.spearmanr(data_dict[file1], data_dict[file2])

        pearson_df.loc[file1, file2] = float(pearson_corr)
        spearman_df.loc[file1, file2] = float(spearman_corr)

# Create heatmaps
plt.figure(figsize=(12, 10))
plt.title("Pearson Correlation Matrix")
sns.heatmap(pearson_df.astype('float'), annot=True, cmap='coolwarm', center=0)
plt.show()

plt.figure(figsize=(12, 10))
plt.title("Spearman Correlation Matrix")
sns.heatmap(spearman_df.astype('float'), annot=True, cmap='coolwarm', center=0)
plt.show()
