# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:54:55 2025

@author: robbi
"""

import os
import glob
import re
import pandas as pd
import numpy as np
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


# os.chdir("Uni_work/Technical_project/Short_circuits_Li-Ion_batteries")


# file_path = os.path.join("short_circuit", "avg_temperature_data_V_0.1_tdg_900.csv")

print(os.getcwd())

root_dir = "."

class_folders = {"no_short_circuit": 0, "short_circuit": 1}

samples = []
labels = []
voltages = [0.1,0.2,0.30000000000000004,0.4,0.5,0.6,0.7000000000000001,0.8,0.9,1.0]
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

metadata = pivot_df.index

# Reshape temperature data (stack every 3 columns into new rows)
reshaped_data = pivot_df.to_numpy().reshape(-1, 3)

# Repeat metadata to match the new shape
expanded_meta = metadata.repeat(pivot_df.shape[1] // 3)

# Create DataFrame with metadata as index
new_df = pd.DataFrame(reshaped_data, index=expanded_meta)

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
classifiers = {
    "Rocket Classifier": RocketClassifier(num_kernels=500),
    "Time Series Forest": TimeSeriesForestClassifier(n_estimators=100),
    # "Shapelet Transform": ShapeletTransformClassifier(),
    "Catch22 Classifier": Catch22Classifier(),
    # # "Proximity Forest": ProximityForest()
}

# Train and evaluate classifiers
results = {}
for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# Print final results
print("\nFinal Accuracy Scores:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")