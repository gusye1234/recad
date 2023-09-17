import random
import numpy as np
import torch
from .default import *
from . import dataset
from . import model
from . import workflow
from . import utils
from .main import main

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

__version__ = '0.0.1'
__author__ = 'Jianbai Ye'
__url__ = 'https://github.com/gusye1234/recad'
