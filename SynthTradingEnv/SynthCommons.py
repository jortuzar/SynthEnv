# import sys
# import os
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
import json
import zipfile
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def load_json(path, iszip=False):
    if iszip:
        zf = zipfile.ZipFile(path)
        data = json.load(zf.open(zipfile.ZipFile.namelist(zf)[0]))
    else:
        with open(path) as f:
            data = json.load(f)
    return np.asarray(data)


def load_npz(path):
    loaded = np.load(path)
    return loaded['arr']


def get_acc(y_p, y, p=0.5, p_max=1):
    if p > 0.5:
        p = 0.5
    correct = 0
    totP = 0.00000000001
    for i in range(len(y_p)):
        if (y_p[i] > 1 - p) or (y_p[i] <= p):
            totP += 1

        if (y_p[i] > 1 - p and y[i] == 1) or (y_p[i] <= p and y[i] == 0):
            correct += 1

    return correct / totP, totP


def get_acc2(y_p, y, p=0.5, p_max=1):
    p = p * p_max
    if p > p_max / 2:
        p = p_max / 2
    correct = 0
    totP = 0.00000000001
    for i in range(len(y_p)):
        if (y_p[i] > p_max - p) or (y_p[i] <= p):
            totP += 1

        if (y_p[i] > p_max - p and y[i] == 1) or (y_p[i] <= p and y[i] == 0):
            correct += 1

    return correct / totP, totP


def merge_dicts(dict1, dict2):
    merged_dictionary = {}

    for key in dict1:
        if key in dict2:
            new_value = dict1[key] + dict2[key]
        else:
            new_value = dict1[key]

        merged_dictionary[key] = new_value

    for key in dict2:
        if key not in merged_dictionary:
            merged_dictionary[key] = dict2[key]

    return merged_dictionary


# NORMALIZE FEATURES (NO TIME STEPS)
def norm_features(features, n_label_cols, test_scaler=None):
    price_info = features[:, 0:n_label_cols]
    feature_info = features[:, n_label_cols:]
    if test_scaler is None:
        scaler = MinMaxScaler()  # StandardScaler()
    else:
        scaler = test_scaler
    scaler.fit(feature_info)
    feature_info = scaler.transform(feature_info)
    return np.array(feature_info), np.array(price_info), scaler

# NORMALIZE DATA
def norm_data(data_for_norm, return_scaler=False, test_scaler=None):
    # USE scaler.inverse_transform(normalized) to reverese normalization
    if test_scaler is None:
        scaler = MinMaxScaler()  # StandardScaler()
    else:
        scaler = test_scaler
    if (data_for_norm.ndim == 3):
        reshapedInputData = data_for_norm.reshape(data_for_norm.shape[0],
                                                  data_for_norm.shape[1] * data_for_norm.shape[2])
        if test_scaler is None: scaler.fit(reshapedInputData)
        scaledInputData = scaler.transform(reshapedInputData)
        scaledInputDataReshaped = scaledInputData.reshape(data_for_norm.shape[0], data_for_norm.shape[1],
                                                          data_for_norm.shape[2])
        if return_scaler:
            return scaledInputDataReshaped, scaler
        else:
            return scaledInputDataReshaped
    elif (data_for_norm.ndim == 2):
        reshapedInputData = data_for_norm
        if test_scaler is None: scaler.fit(reshapedInputData)
        scaledInputData = scaler.transform(reshapedInputData)
        if return_scaler:
            return np.array(scaledInputData), scaler
        else:
            return np.array(scaledInputData)
    elif (data_for_norm.ndim == 1):
        reshapedInputData = data_for_norm.reshape(data_for_norm.shape[0], 1)
        if test_scaler is None: scaler.fit(reshapedInputData)
        scaledInputData = scaler.transform(reshapedInputData)
        scaledInputDataReshaped = scaledInputData.reshape(data_for_norm.shape[0], )
        if return_scaler:
            return np.array(scaledInputDataReshaped), scaler
        else:
            return np.array(scaledInputDataReshaped)


# CONVERT FEATURE LIST TABLE. OUTPUT IS ALWAYS ON COLUMN 0
def convert_to_3D(featureList_, timesteps, has_label_on_first_col=False):
    featureList_ = np.array(featureList_)
    timestep_set = []
    features_set = []
    labels = []
    featureList_length = featureList_.shape[0]
    num_features = featureList_.shape[1]

    for i in range(timesteps, featureList_length):
        for j in range(i - timesteps, i):
            # inputDataRaw[:,1:inputDataRaw.shape[1]][0].tolist()
            if has_label_on_first_col:
                features_set.append(featureList_[:, 1:featureList_.shape[1]][j])
            else:
                features_set.append(featureList_[j])
        timestep_set.append(features_set)
        features_set = []

        if has_label_on_first_col:
            labels.append(featureList_[i - 1, 0])
        else:
            labels.append(featureList_[i, 0])

    timestep_set = np.array(timestep_set)
    labels = np.array(labels)

    return timestep_set, labels


# CONVERT FLAT FEATURES TO TIMESTEPPED FEATURES.
# EXTRACTS FIRST N COLS AS THE Y INFO Price, Bid, Ask
def to_3D(featureList_, timesteps, n_label_cols=3, normalize_features=True, test_scaler=None):
    featureList_ = np.array(featureList_)
    timestep_set = []
    features_set = []
    labels = []
    featureList_length = featureList_.shape[0]
    num_features = featureList_.shape[1]
    scaler = None

    if test_scaler is not None:
        scaler = test_scaler

    if timesteps > 1:
        for i in range(timesteps, featureList_length):
            for j in range(i - timesteps, i):
                # inputDataRaw[:,1:inputDataRaw.shape[1]][0].tolist()
                features_set.append(featureList_[:, n_label_cols:featureList_.shape[1]][j])

            timestep_set.append(features_set)
            features_set = []
            labels.append(featureList_[i - 1, 0:n_label_cols])
    elif timesteps == 1:
        labels = featureList_[:, 0:n_label_cols]
        timestep_set = featureList_[:, n_label_cols:]

    timestep_set = np.array(timestep_set)
    if normalize_features:
        if test_scaler is not None:
            timestep_set = norm_data(timestep_set, return_scaler=False, test_scaler=test_scaler)
        else:
            timestep_set, scaler = norm_data(timestep_set, return_scaler=True)
    labels = np.array(labels)
    # Reshape to (rows, timesteps x features).
    if timesteps > 1:
        timestep_set = timestep_set.reshape(timestep_set.shape[0], timestep_set.shape[1] * timestep_set.shape[2])

    return timestep_set, labels, scaler


def get_binary_output_i(x_data, i_start, n_target_points):
    start_price = x_data[i_start][0]
    output = -1
    for i in range(i_start, len(x_data)):
        if abs(x_data[i][0] - start_price) >= n_target_points:
            if x_data[i][0] - start_price >= 0:
                output = 1
                break
            else:
                output = 0
                break
    return output, i - i_start


def get_binary_outputs(x_data, n_target_points, time_steps, strip_price_col=True):
    features = []
    labels = []
    for i in range(len(x_data)):
        output, steps = get_binary_output_i(x_data, i, n_target_points)
        # print('steps:', steps)
        # print('output:', output)
        if output != -1:
            tmp = []
            tmp.append(output)
            # Take all columns after 0
            tmp = tmp + x_data[:, 1:x_data.shape[1]][i].tolist()
            # tmp.append(steps)
            # print('temp:', tmp)
            features.append(tmp)
            # print('features:', features)
            labels.append(output)
    features = np.array(features)
    labels = np.array(labels)

    features, labels = convert_to_3D(features, time_steps, True)

    return features, labels
