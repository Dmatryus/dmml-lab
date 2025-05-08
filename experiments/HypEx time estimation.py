import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

# Load data (replace with your data path)
filepath = r"C:\Projects\HypEx\examples\experiments\performance_test\aa_performance_test_result.csv"
data = pd.read_csv(filepath)
X = data[["n_iterations", "n_rows", "n_columns"]]
Y = data[["time", "M2"]]

# Initialize the model
model = Ridge()

# # Define the scoring metrics
# scoring = {
#     "rmse": make_scorer(
#         mean_squared_error, greater_is_better=False, transformer=np.sqrt
#     ),
#     "r2": make_scorer(r2_score),
#     "mape": make_scorer(mean_absolute_percentage_error),
# }
#
# # Perform cross-validation with multiple metrics
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# results = pd.DataFrame()
#
# for metric_name, metric in scoring.items():
#     scores = cross_val_score(model, X, Y, cv=kf, scoring=metric)
#     results[metric_name] = scores
#
# # Display the cross-validation results table
# print("\nCross-validation results table:")
# print(results)
# print(results.describe())

# Train the model on all data for prediction
model.fit(X, Y)

# Predict on the test set
y_pred = model.predict(X)

# Evaluate the model on the test set with various metrics
final_rmse = np.sqrt(mean_squared_error(Y, y_pred))
final_r2 = r2_score(Y, y_pred)
final_mape = mean_absolute_percentage_error(Y, y_pred)
print(f"\nFinal RMSE: {final_rmse:.4f}")
print(f"Final R2 Score: {final_r2:.4f}")
print(f"Final MAPE: {final_mape:.4f}")

# Time error
time_rmse = np.sqrt(mean_squared_error(Y["time"], y_pred[:, 0]))
time_r2 = r2_score(Y["time"], y_pred[:, 0])
time_mape = mean_absolute_percentage_error(Y["time"], y_pred[:, 0])
print(f"\nTime RMSE: {time_rmse:.4f}")
print(f"Time R2 Score: {time_r2:.4f}")
print(f"Time MAPE: {time_mape:.4f}")

# Memory error
memory_rmse = np.sqrt(mean_squared_error(Y["M2"], y_pred[:, 1]))
memory_r2 = r2_score(Y["M2"], y_pred[:, 1])
memory_mape = mean_absolute_percentage_error(Y["M2"], y_pred[:, 1])
print(f"\nMemory RMSE: {memory_rmse:.4f}")
print(f"Memory R2 Score: {memory_r2:.4f}")
print(f"Memory MAPE: {memory_mape:.4f}")

iterations_states = range(10, 3010, 10)
predicted_data = pd.DataFrame(
    {
        "n_iterations": iterations_states,
        "n_rows": [10_000] * len(iterations_states),
        "n_columns": [20] * len(iterations_states),
    }
)

y_pred = model.predict(predicted_data)
predicted_data["time"] = y_pred[:, 0]
predicted_data["M2"] = y_pred[:, 1]
predicted_data.to_csv("predicted_data.csv", index=False)
