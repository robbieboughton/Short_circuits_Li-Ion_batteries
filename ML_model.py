# # # -*- coding: utf-8 -*-
# # """
# # Created on Wed Feb  5 10:54:55 2025

# # @author: robbi
# # """

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sktime.dists_kernels import FlatDist, ScipyDist
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.distance_based import ProximityForest
from sktime.classification.feature_based import Catch22Classifier
from sktime.dists_kernels import ScipyDist

# os.chdir("..")
# os.chdir("Uni_work/Technical_project/Short_circuits_Li-Ion_batteries")


# file_path = os.path.join("short_circuit", "avg_temperature_data_V_0.1_tdg_900.csv")

# print(os.getcwd())

root_dir = "."

class_folders = {"no_short_circuit1": 0, "short_circuit1": 1}

samples = []
labels = []
voltages = [0.03,0.06, 0.09,0.12,0.15,0.18,0.21,0.24,0.27,0.30000000000000004]
t_dg = [900, 1800, 2700]
types = ["avg_temperature_data", "resistance_data", "temperature_data"]

np.set_printoptions(legacy = '1.13')

all_rows = []

# for j in types:

for folder_name, label in class_folders.items():
    folder_path = os.path.join(root_dir, folder_name)
    
    for i in t_dg:

        for V in voltages:
            file_path = os.path.join(folder_path, f"avg_temperature_data_V_{V}_tdg_{i}.csv")
            current_file = pd.read_csv(file_path)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                for timestep,avg_temp in zip(df.iloc[:,0], df.iloc[:,1]):
                    all_rows.append([label,i,V,timestep,avg_temp])
                # df["Label"] = label
                # all_data.append(df)
            else:
                print(f"Missing file: {file_path}")

final_df = pd.DataFrame(all_rows, columns=["Label", "Dendrite growth time", "Voltage", "Timestep", "Avg Temperature"])

pivot_df = final_df.pivot_table(index=["Label", "Dendrite growth time", "Voltage"], columns="Timestep", values="Avg Temperature")

# metadata = pivot_df.index

# # Reshape temperature data (stack every 3 columns into new rows)
# reshaped_data = pivot_df.to_numpy().reshape(-1, 3)

# # Repeat metadata to match the new shape
# expanded_meta = metadata.repeat(pivot_df.shape[1] // 3)

# # Create DataFrame with metadata as index
# new_df = pd.DataFrame(reshaped_data, index=expanded_meta)

# # Converting to a NumPy 3D array
# X = new_df.values[:, np.newaxis, :]  # Shape (num_samples, num_timesteps, 1)

# # Extract labels
# y = new_df.index.get_level_values("Label").values

# # Splitting the dataset
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True)

# print("final 3d array shape:", X.shape)

window_size = 3

metadata = pivot_df.index

num_windows = pivot_df.shape[1] - (window_size - 1)
sliding_data = []
sliding_index = []

for i in range(num_windows):
    window_data = pivot_df.iloc[:, i:i+window_size].copy()
    window_data.columns = range(window_size)
    sliding_data.append(window_data)
    sliding_index.extend(metadata)

new_df = pd.concat(sliding_data, keys=range(num_windows), names=["Window"]).reset_index(level=0, drop=True)

# Converting to a NumPy 3D array
X = new_df.values[:, np.newaxis, :]  # Shape (num_samples, num_timesteps, 1)

# Extract labels
y = new_df.index.get_level_values("Label").values

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True)

print("final 3d array shape:", X.shape)




# eucl_dist = FlatDist(ScipyDist())
# clf = KNeighborsTimeSeriesClassifier(n_neighbors=8, distance=eucl_dist)

# clf = RocketClassifier(num_kernels=500)

# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)


# print(accuracy_score(y_test, y_pred))

# Initialize classifiers

class_folders = {"no_short_circuit2": 0, "short_circuit2": 1}

samples = []
labels = []
voltages = [0.035,0.065,0.095,0.125, 0.155,0.185,0.215,0.245,0.275,0.30500000000000005]
t_dg = [500, 1500, 2500]
types = ["avg_temperature_data", "resistance_data", "temperature_data"]

np.set_printoptions(legacy = '1.13')

all_rows = []

# for j in types:

for folder_name, label in class_folders.items():
    folder_path = os.path.join(root_dir, folder_name)
    
    for i in t_dg:

        for V in voltages:
            file_path = os.path.join(folder_path, f"avg_temperature_data_V_{V}_tdg_{i}.csv")
            current_file = pd.read_csv(file_path)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                for timestep,avg_temp in zip(df.iloc[:,0], df.iloc[:,1]):
                    all_rows.append([label,i,V,timestep,avg_temp])
                # df["Label"] = label
                # all_data.append(df)
            else:
                print(f"Missing file: {file_path}")

final_df2 = pd.DataFrame(all_rows, columns=["Label", "Dendrite growth time", "Voltage", "Timestep", "Avg Temperature"])

pivot_df2 = final_df2.pivot_table(index=["Label", "Dendrite growth time", "Voltage"], columns="Timestep", values="Avg Temperature")

test_row = pivot_df2.iloc[[59]]

window_size = 3

metadata = test_row.index

num_windows = test_row.shape[1] - (window_size - 1)
sliding_data = []
sliding_index = []

for i in range(num_windows):
    window_data = test_row.iloc[:, i:i+window_size].copy()
    window_data.columns = range(window_size)
    sliding_data.append(window_data)
    sliding_index.extend(metadata)

new_df = pd.concat(sliding_data, keys=range(num_windows), names=["Window"]).reset_index(level=0, drop=True)

# Converting to a NumPy 3D array
X_test_row = new_df.values[:, np.newaxis, :]  # Shape (num_samples, num_timesteps, 1)

# Extract labels
y_test_row = new_df.index.get_level_values("Label").values

classifiers = {
    # "Rocket Classifier": RocketClassifier(),
    "Time Series Forest": TimeSeriesForestClassifier(),
#     # "Shapelet Transform": ShapeletTransformClassifier(),
#     # "Catch22 Classifier": Catch22Classifier(),
#     # # "Proximity Forest": ProximityForest()
}

# Train and evaluate classifiers
results = {}
for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(x_train, y_train)
    y_pred = clf.predict(X_test_row)
    acc = accuracy_score(y_test_row, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# Print final results
print("\nFinal Accuracy Scores:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")
    
print("Final innacurate index: ")
print(np.max(np.where(y_pred == 0)[0]))
print("Highest temperature without a short circuit: ")
print(pivot_df.iloc[:30].values.max())
print("Temperatures in window of last incorrect prediction: ")
print(X_test_row[np.max(np.where(y_pred == 0)[0])])
    
# # results = {}
# # for name, clf in classifiers.items():
# #     print(f"Training {name}...")
# #     clf.fit(x_train, y_train)
# #     y_pred = clf.predict(X_test_row)
# #     acc = accuracy_score(y_test_row, y_pred)
# #     results[name] = acc
# #     print(f"{name} Accuracy: {acc:.4f}")

# # # Print final results
# # print("\nFinal Accuracy Scores:")
# # for name, acc in results.items():
# #     print(f"{name}: {acc:.4f}")
    
# # bin_size = 4
# # num_bins = len(y_pred) // bin_size

# # ones_counts = [np.sum(y_pred[i * bin_size: (i + 1) * bin_size]) for i in range(num_bins)]

# # group_indices = np.arange(1, num_bins + 1)

# # plt.figure(figsize=(10, 5))
# # plt.bar(group_indices, ones_counts, color='blue', alpha=0.7, edgecolor='black')

# # plt.xlabel('Group of 4')
# # plt.ylabel('Number of 1s in Group')
# # plt.title('Number of 1s in Each 4-row Group of y_pred')
# # plt.xticks(group_indices, rotation=90)
# # plt.yticks(range(5))
# # plt.ylim(0, 4) 
# # plt.grid(axis='y', linestyle='--', alpha=0.7)

# # plt.show()
    
# # Number of trials
# num_trials = 50

# # Initialize classifier
# clf = TimeSeriesForestClassifier()

# # Store prediction counts
# num_windows = X_test_row.shape[0]
# prediction_counts = np.zeros(num_windows)

# last_inaccuracy = np.zeros(num_trials)
# highest_no_sc_temp = np.zeros(num_trials)
# last_inaccuracy_temp = np.zeros(num_trials)

# # Run 100 trials
# for _ in range(num_trials):
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
#     clf.fit(x_train, y_train)
    
#     y_pred = clf.predict(X_test_row)

#     prediction_counts += (y_pred == 0)
    
#     print("Final innacurate index: ")
#     print(np.max(np.where(y_pred == 0)[0]))
#     last_inaccuracy[_] = np.max(np.where(y_pred == 0)[0])
#     print("Highest temperature without a short circuit: ")
#     print(pivot_df.iloc[:30].values.max())
#     highest_no_sc_temp[_] = pivot_df.iloc[:30].values.max()
#     print("Temperatures in window of last incorrect prediction: ")
#     print(X_test_row[np.max(np.where(y_pred == 0)[0])])
#     last_inaccuracy_temp[_] = np.mean(X_test_row[np.max(np.where(y_pred == 0)[0])])
    
#     print(f"Trial number: {_}")

# average_temperatures = X_test_row.mean(axis=(1, 2))

# # Plot results
# plt.figure(figsize=(10, 5))
# plt.bar(average_temperatures, prediction_counts, width=0.5, color='blue', alpha=0.7, edgecolor='black')
# plt.xlabel('Average Temperature of Window')
# plt.ylabel('Number of Times Predicted 0 Over 100 Trials')
# plt.title('Prediction Frequency of 0s for Different Temperature Windows')
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

# # Identify misclassified indices
# misclassified_indices = np.where(y_pred != y_test_row)[0]

# # Extract the temperature values for misclassified and correctly classified windows
# misclassified_temps = X_test_row[misclassified_indices, :, :].flatten()
# correctly_classified_temps = X_test_row[np.where(y_pred == y_test_row)[0], :, :].flatten()

# # Plot histograms to compare temperature distributions
# plt.figure(figsize=(10, 5))
# plt.hist(misclassified_temps, bins=30, alpha=0.6, label="Misclassified", color='red')
# plt.hist(correctly_classified_temps, bins=30, alpha=0.6, label="Correctly Classified", color='green')
# plt.xlabel("Temperature")
# plt.ylabel("Frequency")
# plt.legend()
# plt.title("Temperature Distribution: Misclassified vs. Correctly Classified Windows")
# plt.show()