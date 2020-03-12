import torch
import random
import numpy as np
from torch.utils.data import DataLoader

# custom module
from config import *
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

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

		X = np.array(data)[:, 0]
		grp = np.array(data)[:, -1]

		sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_RATIO * 2, random_state=SEED)

		for train_idx, test_idx in sss.split(X, grp):  # split train set and (test set + valid set)
			trainX = X[train_idx]
			testX_sample = X[test_idx]
			trainY = grp[train_idx]
			testY_sample = grp[test_idx]

		sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)

		for test_idx, valid_idx in sss.split(testX_sample, testY_sample):  # split test set and valid set
			testX = testX_sample[test_idx]
			validX = testX_sample[valid_idx]
			testY = testY_sample[test_idx]
			validY = testY_sample[valid_idx]

		trainM = np.vstack([trainX, trainY]).T
		validM = np.vstack([validX, validY]).T
		testM = np.vstack([testX, testY]).T

		trainM = [[self.tensorize.embedding(item[0]), int(item[1])] for item in trainM if len(item) == 2] # into torch.tensor
		trainM = [(self.trim(item[0]), item[1]) if item[0].size(0) > MAX_SEQ_LEN else (self.zero_pad(item[0]), item[1]) for item in trainM] # if seq len is too long, trim it. otherwise, pad the sequence.

		validM = [[self.tensorize.embedding(item[0]), int(item[1])] for item in validM if len(item) == 2]  # into torch.tensor
		validM = [(self.trim(item[0]), item[1]) if item[0].size(0) > MAX_SEQ_LEN else (self.zero_pad(item[0]), item[1]) for item in validM]  # if seq len is too long, trim it. otherwise, pad the sequence.

		testM = [[self.tensorize.embedding(item[0]), int(item[1])] for item in testM if	len(item) == 2]  # into torch.tensor
		testM = [(self.trim(item[0]), item[1]) if item[0].size(0) > MAX_SEQ_LEN else (self.zero_pad(item[0]), item[1]) for item in testM]  # if seq len is too long, trim it. otherwise, pad the sequence.

		# random.seed(SEED)
		# random.shuffle(data)

		# test_idx = int(len(data) * TEST_RATIO)
		# train = DataLoader(data[test_idx * 2:], batch_size=BATCH_SIZE)
		# val = DataLoader(data[test_idx:test_idx * 2], batch_size=BATCH_SIZE)
		# test = DataLoader(data[:test_idx], batch_size=BATCH_SIZE)

		train = DataLoader(trainM, batch_size=BATCH_SIZE)
		val = DataLoader(validM, batch_size=BATCH_SIZE)
		test = DataLoader(testM, batch_size=BATCH_SIZE)



		return train, val, test