import numpy as np
import torch as torch
from torch.utils.data import Dataset
import random

def create_data(num_risk_vals, data_size, noise_factor):
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
    age_value = 0.5

    gender = input_values[1] #0 -> Male #1 -> Female
    gender_value = 0.5

    child = input_values[2]
    child_value = 0.5

    #age mapping
    if age_weight==0.5:
        age_value = 0.5
    if age_weight < 0.5:
        age_offset = 1.5-age_weight
        if age < 50:
            age_value = (age * (1-(age_offset-1))) / 100
        else:
            age_value =  (age * age_offset) / 100

    if age_weight > 0.5:
        age_offset = 1+age_weight-0.5 #1.1
        if age < 50:
            age_value =  ((100-age) * age_offset) / 100
        else:
            age_value = ((100-age) * (1-(age_offset-1))) / 100

    #print("Age: ", age_value)

    #gender mapping
    if gender_weight == 0.5:
        gender_value = 0.5
    if gender_weight < 0.5:
        if gender == 0:
            gender_value = 0 + gender_weight
        else:
            gender_value = 1 - gender_weight

    if gender_weight > 0.5:
        if gender == 0:
            gender_value = 0 + gender_weight
        else:
            gender_value = 1 - gender_weight

    #print("Gender: ", gender_value)

    #children mapping
    if child_weight == 0.5:
        child_value = 0.5
    if child_weight < 0.5:
        if child == 0:
            child_value = 0 + child_weight
        else:
            child_value = 1 - child_weight

    if child_weight > 0.5:
        if child == 0:
            child_value = 0 + child_weight
        else:
            child_value = 1 - child_weight


    #print("Child: ", child_value)
    #print("")

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
    mapped = random.sample(mapped, len(mapped))
    return mapped