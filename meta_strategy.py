import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VIX_data = pd.read_csv("data/VIX.csv")
VIX_futures_data = pd.read_csv("data/VIX_futures.csv")
# TODO: combine VIX with VIX_futures
# TODO: maybe feature engineer VIX and VIX_futures together