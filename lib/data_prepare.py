import torch
import pandas as pd
import numpy as np
import os
from .utils import print_log, StandardScaler

def read_numpy(data_path, transpose=False, log=None):
    if data_path.endswith("npy"):
        data = np.load(data_path).astype(np.float32)
    elif data_path.endswith("npz"):
        data = np.load(data_path)["data"].astype(np.float32)
    else:
        raise TypeError("Unsupported file type.")
    if transpose:
        data = data.T
    print_log("Original data shape", data.shape, log=log)
    return data
