import random
import numpy as np
import torch
from .default import *
from . import dataset
from . import model
from . import workflow

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
