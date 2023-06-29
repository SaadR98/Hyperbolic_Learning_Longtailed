# Hyperbolic_Learning_Longtailed

Packages used in this project:
- os
- pickle
- numpy as np
- pandas as pd
- random

- geoopt
- from geoopt import PoincareBallExact

- torch
- torch.nn as nn
- torch.optim as optim
- torchvision
- torchvision.models as models
- from torchvision.datasets import ImageFolder
- from torch.utils.data import DataLoader

- torch.nn.functional as F
- torchvision.transforms as transforms
- torchvision.datasets as datasets
from torch.optim.lr_scheduler - StepLR, MultiStepLR

Instructions: 
1) Load the data, define the IMBALance factor and the file paths
2) Load the prototypes
   - Check which prototypes one you need
   - Download the self-made prototypes if using those
3) Choose the model
4) Run the training and validation loop
5) Save the model and use it for analysis
