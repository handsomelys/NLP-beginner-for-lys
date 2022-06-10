import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import numpy as np
import pkuseg
import argparse
from tqdm import trange,tqdm
import os
from utils import read_corpus,batch_iter
from vocab import Vocab
from model import CNN
import math
from sklearn.metrics import f1_score