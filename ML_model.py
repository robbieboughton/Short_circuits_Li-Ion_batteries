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

from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sklearn.pipeline import make_pipeline
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

for folder_name, label in class_folders.items():
    folder_path = os.path.join(root_dir, folder_name)

    for V in voltages:
        file_path = os.path.join(folder_path, f"avg_temperature_data_V_{V}_tdg_900.csv")
        current_file = pd.read_csv(file_path)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            for timestep,avg_temp in zip(df.iloc[:,0], df.iloc[:,1]):
                all_rows.append([label,V,timestep,avg_temp])
            # df["Label"] = label
            # all_data.append(df)
        else:
            print(f"Missing file: {file_path}")

final_df = pd.DataFrame(all_rows, columns=["Label", "Voltage", "Timestep", "Avg Temperature"])

final_df.set_index(['Label', 'Voltage', 'Timestep'], inplace=True)

# final_df = final_df[['Avg Temperature']]

    


# for folder_name, label in class_folders.items():
#     folder_path = os.path.join(root_dir, folder_name)

#     file_paths = glob.glob(os.path.join(folder_path, "*.csv"))
    
#     sample_files = {}
#     for file in file_paths:
#         basename = os.path.basename(file)

#         match = re.match(r"(.+)_V_([0-9\.]+)_tdg_([0-9]+)", basename)
#         if match:
#             data_type = match.group(1)
#             voltage = match.group(2)    
#             tdg = match.group(3)        
#             key = (voltage, tdg)
#             if key not in sample_files:
#                 sample_files[key] = {}
#             sample_files[key][data_type] = file


#     for key, files_dict in sample_files.items():
#         required_data = {"avg_temperature_data"}
#         if required_data.issubset(files_dict.keys()):
#             sample = {}
#             for dt in required_data:
#                 # Read the CSV file into a pandas Series. Adjust header parameter if needed.
#                 ts = pd.read_csv(files_dict[dt], header=None).iloc[:, 0]
#                 print(ts.head())
#                 sample[dt] = ts
#             samples.append(sample)
#             labels.append(label)

# # Create a nested DataFrame.
# # Each row corresponds to a battery pack sample and each cell in the data columns is a pd.Series.
# X = pd.DataFrame(samples)
# y = np.array(labels)

# print(f"Total samples: {len(X)}")

# print(f"Shape of X: {X.shape}")
# print(f"Length of y: {len(y)}")


# # Split the data into training and testing sets.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Convert list of Series objects into a proper 2D NumPy array
# X_train_np = np.stack(X_train['avg_temperature_data'].values)
# X_test_np = np.stack(X_test['avg_temperature_data'].values)

# X_train_np = X_train_np[:, 1:] 
# X_test_np = X_test_np[:, 1:]  

# X_train_np = X_train_np.astype(np.float64)
# X_test_np = X_test_np.astype(np.float64)


# ###############################################
# # Classifier 1: ROCKET transformer + RidgeClassifierCV
# ###############################################
# # Build the pipeline.
# rocket = Rocket(random_state=42, num_kernels=10000)
# ridge_clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
# rocket_pipeline = make_pipeline(rocket, ridge_clf)

# # Train the pipeline.
# rocket_pipeline.fit(X_train_np, y_train)
# # Predict on test set.
# y_pred_rocket = rocket_pipeline.predict(X_test_np)
# accuracy_rocket = accuracy_score(y_test, y_pred_rocket)
# print(f"ROCKET + RidgeClassifierCV accuracy: {accuracy_rocket:.3f}")

# ###############################################
# # Classifier 2: TimeSeriesForestClassifier
# ###############################################
# # Instantiate the TimeSeriesForestClassifier.
# tsf = TimeSeriesForestClassifier(random_state=42)

# # Train the classifier.
# tsf.fit(X_train_np, y_train)
# # Predict on test set.
# y_pred_tsf = tsf.predict(X_test_np)
# accuracy_tsf = accuracy_score(y_test, y_pred_tsf)
# print(f"TimeSeriesForestClassifier accuracy: {accuracy_tsf:.3f}")
