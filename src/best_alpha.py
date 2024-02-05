import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read the data
xpc_data = pd.read_csv(r"new_data\XPC_data.bed.csv")
csb_data = pd.read_csv(r"new_data\CSB_data.bed.csv")
wt_data = pd.read_csv(r"new_data\WT_data.bed.csv")

alphas = np.arange(0.01, 1, 0.01)  # Loop from .01 to .99
errors = []

for alpha in alphas:
    formula = (csb_data['counts'] * alpha) + (xpc_data['counts'] * (1 - alpha))
    model = LinearRegression().fit(formula.values.reshape(-1, 1), wt_data['counts'])
    error = model.score(formula.values.reshape(-1, 1), wt_data['counts'])
    errors.append(abs(error))

# Get the alpha with minimum error
max_error_idx = np.argmax(errors)
best_alpha = alphas[max_error_idx]
print(f"The best alpha is: {best_alpha}")

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(alphas, errors, color='#c78c80')
plt.scatter(best_alpha, errors[max_error_idx], color='#54545f')  # best alpha
plt.xlabel("Alpha")
plt.ylabel("R2 Score")
plt.title("Alpha optimization - R2 Score")
plt.grid(True)
plt.show()