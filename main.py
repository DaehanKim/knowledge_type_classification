# GRU attention knowledge type classification
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from kor2vec import Kor2Vec
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Hyperparameters
NUM_EPOCH = 20
BATCH_SIZE = 32
MAX_SEQ_LEN = 100

LEARNING_RATE = 0.01
WORD_DIM = 128
HIDDEN_DIM = 64
NUM_LAYER = 1
BIDIRECTIONAL = False
TEST_RATIO = 0.15
SEED = 20200306


# to be implemented : dropout, learning rate schedule
# 					: generate test, pred pairs
DROPOUT_RATE = 0.5


# Define cls model class
class GRU_ATT(nn.Module):
	def __init__(self):
		super(GRU_ATT, self).__init__()
		self.rnn = nn.GRU(WORD_DIM,HIDDEN_DIM,NUM_LAYER, batch_first = True, bidirectional = BIDIRECTIONAL)
		self.att_layer = nn.Linear(HIDDEN_DIM, 1)
		self.fc = nn.Sequential(nn.Linear(HIDDEN_DIM, 5), nn.LogSoftmax(dim=1)) # definition, process, property, example, the others
		self.hidden = None

	def forward(self, x):
		# x : batch x seq_len x word_dim
		out, ht = self.rnn(x, self.hidden[:,:x.size(0),:])
		att_weight = F.softmax(self.att_layer(out), dim = 2)
		att_applied = (att_weight * out).sum(1)
		logit = self.fc(att_applied)
		return logit

	def init_hidden(self):
		return nn.Parameter(torch.zeros(NUM_LAYER*(int(BIDIRECTIONAL)+1), BATCH_SIZE, HIDDEN_DIM)) 

class GRU_ATT_WRAP:
	def __init__(self, model):
		self.model = model
		self.model.hidden = self.model.init_hidden()
		self.optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
		self.scheduler = ReduceLROnPlateau(self.optimizer, mode = 'min')
		self.criterion = nn.NLLLoss()

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


def unittest_loader():
	word_emb = Kor2Vec.load('kor2vec03010231.checkpoint.ep0')
	ld = LOADER(word_emb)
	for batch in ld.train:
		print(batch[0],batch[1])
		break
	for batch in ld.train:
		print(batch[0],batch[1])
		break

# train/ test logic
def train_epoch(model_wrap, train_loader, val_loader):
	model_wrap.model.train()
	# do train
	for batch in train_loader:
		model_wrap.optimizer.zero_grad()
		out = model_wrap.model(batch[0])
		loss = model_wrap.criterion(out, batch[1])
		loss.backward(retain_graph=True)
		model_wrap.optimizer.step()

	# do validation to do early stopping
	acc, loss_ = test(model_wrap, val_loader)
	print('val - accuracy : {:.4f} | loss : {:.6f}'.format(acc, loss_))



def test(model_wrap, test_loader):
	model_wrap.model.eval()
	
	correct = 0.
	running_loss = 0.
	num_sample = 0.
	for batch in test_loader:
		out = model_wrap.model(batch[0])
		loss = model_wrap.criterion(out, batch[1])
		correct += (out.max(dim=1)[1] == batch[1]).sum()
		running_loss += loss.item()
		num_sample += out.size(0)

	return correct/num_sample, running_loss/num_sample


def main():

	# kor2vec model load
	word_emb = Kor2Vec.load('kor2vec03010231.checkpoint.ep0')
	rnn_clf = GRU_ATT()
	model = GRU_ATT_WRAP(rnn_clf)
	loader = LOADER(word_emb)

	for epoch in range(NUM_EPOCH):
		train_epoch(model, loader.train, loader.val)
		acc, loss_ = test(model, loader.test)
	print('test - accuracy : {:.4f} | loss : {:.6f}'.format(acc, loss_))	


if __name__ == '__main__':
	# unittest_loader()
	main()