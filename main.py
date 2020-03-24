# GRU attention knowledge type classification
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from kor2vec import Kor2Vec
from collections import Counter
from sklearn.metrics import confusion_matrix

# custom modules
from loader import *
from model import *
from config import *

# train/ test logic
def train_epoch(model_wrap, test_raw, train_loader, val_loader):
	model_wrap.model.train()

	for batch in train_loader:
		model_wrap.optimizer.zero_grad()
		out = model_wrap.model(batch[0])
		loss = model_wrap.criterion(out, batch[1])
		loss.backward(retain_graph=True)
		model_wrap.optimizer.step()
		model_wrap.scheduler.step(loss.item())

	# do validation to do early stopping
	acc, loss_ = test(model_wrap, test_raw, val_loader)
	print('val - accuracy : {:.4f} | loss : {:.6f}'.format(acc, loss_))



def test(model_wrap, test_raw, test_loader, print_confusion_matrix=False, print_test_set=False):
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

	if print_confusion_matrix:
		print(*cm(model_wrap, test_raw, test_loader, print_test_set), sep='\n')


	return correct/num_sample, running_loss/num_sample


def cm(model_wrap, test_raw, test_loader, print_test_set=False):
	model_wrap.model.eval()
	
	pred_y, true_y = [], []

	for batch in test_loader:
		out = model_wrap.model(batch[0])
		pred_y.append(out.max(dim=1)[1])
		true_y.append(batch[1])

	# For printing
	# -------------------------------------------------------------------------------------------------------------#
	if print_test_set:
		relevant_cols = ['정의', '과정', '성질', '예', '흥미유발']
		label_map = dict(zip(range(5), relevant_cols))
		printing = []
		index = 0
		for batch in test_raw:
			printing.append(batch)
			index += 1
		for i in range(len(printing)):
			for j in range(len(printing[i])):
				if pred_y[i][j] != true_y[i][j]:
					print(printing[i][j]+" 예측: "+label_map[int(pred_y[i][j])]+", 실제: "+label_map[int(true_y[i][j])])
	# -------------------------------------------------------------------------------------------------------------#

	existing_types = Counter(np.concatenate(pred_y + true_y)).keys()
	print(Counter(np.concatenate(true_y)))

	return existing_types, confusion_matrix(np.concatenate(pred_y), np.concatenate(true_y))

def main():

	# kor2vec model load
	word_emb = Kor2Vec.load('kor2vec03010231.checkpoint.ep0')
	rnn_clf = GRU_ATT()
	model = GRU_ATT_WRAP(rnn_clf)
	loader = LOADER(word_emb)

	for epoch in range(NUM_EPOCH):
		train_epoch(model, loader.test_raw, loader.train, loader.val)
	acc, loss_ = test(model, loader.test_raw, loader.test, print_confusion_matrix=True, print_test_set=True)
	print('test - accuracy : {:.4f} | loss : {:.6f}'.format(acc, loss_))	


if __name__ == '__main__':
	main()