import numpy as np
import pandas as pd
import torch

def mytranspose(data):
    if isinstance(data, np.ndarray):
        return data.T
    elif isinstance(data, list):
        return np.array(data).T
    elif isinstance(data, pd.DataFrame):
        return data.transpose()
    elif isinstance(data, pd.Series):
        return data.to_frame().T
    elif isinstance(data, torch.Tensor):
        return data.T
    else:
        raise TypeError("Unsupported data type for transposition")
