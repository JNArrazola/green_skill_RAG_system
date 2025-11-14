import torch 
import os 
import torch
import numpy as np

for file in os.listdir("../predictions"):
    if file.endswith(".pt"):
        filepath = os.path.join("../predictions", file)
        data = torch.load(filepath)
        data_np = data.numpy()
        print(data_np.shape)