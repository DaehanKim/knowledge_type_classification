import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from kor2vec import Kor2Vec
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import *

def loader():
	word_emb = Kor2Vec.load('kor2vec03010231.checkpoint.ep0')
	ld = LOADER(word_emb)
	for batch in ld.train:
		print(batch[0],batch[1])
		break
	for batch in ld.train:
		print(batch[0],batch[1])
		break