import numpy as np
import torch as torch
from torch.utils.data import Dataset
import random


def create_data(num_risk_vals, data_size, noise_factor):
    """
    Generate synthetic data for a binary classification problem with specified parameters.

    Parameters:
    - num_risk_vals (int): Number of risk values to generate for each data point.
    - data_size (int): Total number of data points to generate.
    - noise_factor (float): Factor controlling the amount of noise in the data.

    Returns:
    - torch.Tensor: Tensor containing the generated data with shape (data_size, num_risk_vals*2).
    - torch.Tensor: Tensor containing corresponding labels with shape (data_size, 2).
                   Labels are binary, with the first column indicating the high-risk class (1 or 0),
                   and the second column indicating the low-risk class (1 or 0).
    """
    data = np.zeros((data_size, num_risk_vals*2))
    labels = np.zeros((data_size, 2))

    for i in range(data_size):
        for k in range(num_risk_vals):
            high_seed = np.random.uniform(0.5-noise_factor, 1.0)
            low_seed = np.random.uniform(0.0, 0.5+noise_factor)
            data[i][k] = np.random.uniform(high_seed, 1.0)
            data[i][k+num_risk_vals] = np.random.uniform(0.0, low_seed)

    data[int(data_size/2)::] = np.flip(data[int(data_size/2)::])
    labels[int(data_size/2)::, 0] = 1.
    labels[0:int(data_size / 2), 1] = 1.
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


class TrolleyDataset(Dataset):
    def __init__(self, risk_values, labels):
        self.risk_values = risk_values
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.risk_values[item], self.labels[item]


def convert_persona(input_values, list_length, age_weight, gender_weight, child_weight):
    age = input_values[0]
    gender = input_values[1]  # 0 -> Male, 1 -> Female
    child = input_values[2]

    # age mapping
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    if age >= 40:
        age_value = sigmoid(((age_weight-0.5)*(40-age))/4)
    elif age < 40:
        age_value = sigmoid(((-age_weight+0.5)*(age-40))/4)
    # print("Age: ", age_value)

    # gender mapping
    if gender==0:
        gender_value = gender_weight
    elif gender==1:
        gender_value = 1 - gender_weight
    # print("Gender: ", gender_value)

    # children mapping
    if child==0:
        child_value = child_weight
    elif child==1:
        child_value = 1 - child_weight
    # print("Child: ", child_value)

    noise_factor = 0.02

    #only works for list lengths that are divisble by 3
    size = int(np.floor(list_length/3))
    age_data = np.random.uniform(age_value-noise_factor, age_value+noise_factor, size)
    gender_data = np.random.uniform(gender_value - noise_factor, gender_value + noise_factor, size)
    child_data = np.random.uniform(child_value - noise_factor, child_value + noise_factor, size)

    mapped = list(age_data) + list(gender_data) + list(child_data)
    #print(age_data)
    #print(gender_data)
    #print(child_data)
    random.shuffle(mapped)
    return mapped