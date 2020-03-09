import torch
import random
from torch.utils.data import DataLoader

# custom module
from config import *

class LOADER:
	def __init__(self, word_emb):
		self.tensorize = word_emb
		self.zero_pad = lambda seq : torch.cat([seq, torch.zeros(MAX_SEQ_LEN-seq.size(0),WORD_DIM)], dim=0)
		self.trim = lambda seq : seq[:MAX_SEQ_LEN,:].contiguous()
		self.train, self.val, self.test = self.get_data_loader(test_ratio=TEST_RATIO)

	def get_data_loader(self, test_ratio):
		with open('txt_clf_data.tsv','rt',encoding = 'utf8') as f:
			data = f.readlines()
		data = [item.strip().split('\t') for item in data]
		data = [[self.tensorize.embedding(item[0]), int(item[1])] for item in data if len(item) == 2] # into torch.tensor
		data = [(self.trim(item[0]), item[1]) if item[0].size(0)>MAX_SEQ_LEN else (self.zero_pad(item[0]), item[1]) for item in data] # if seq len is too long, trim it. otherwise, pad the sequence.

		random.seed(SEED)
		random.shuffle(data)

		test_idx = int(len(data) * TEST_RATIO)
		train = DataLoader(data[test_idx*2:], batch_size = BATCH_SIZE)
		val = DataLoader(data[test_idx:test_idx*2], batch_size = BATCH_SIZE)
		test = DataLoader(data[:test_idx], batch_size = BATCH_SIZE)

		return train, val, test