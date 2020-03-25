import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# custom module
from config import *

# Define cls model class
class GRU_ATT(nn.Module):
	def __init__(self):
		super(GRU_ATT, self).__init__()
		self.rnn = nn.GRU(WORD_DIM,HIDDEN_DIM,NUM_LAYER, batch_first = True, bidirectional = BIDIRECTIONAL)
		self.att_layer = nn.Linear(HIDDEN_DIM*(int(BIDIRECTIONAL)+1), 1)
		self.fc = nn.Sequential(nn.Linear(HIDDEN_DIM*(int(BIDIRECTIONAL)+1), 5), nn.LogSoftmax(dim=1)) # definition, process, property, example, the others
		self.hidden = None

	def forward(self, x):
		# x : batch x seq_len x word_dim
		mask = ~torch.eq(x[:,:,0], torch.zeros(*x.size()[:-1]))
		x = F.dropout(x, p = DROPOUT_RATE, training=self.training)
		out, ht = self.rnn(x, self.hidden[:,:x.size(0),:])
		att_weight = F.softmax(self.att_layer(out), dim = 1)
		att_weight = att_weight*mask.unsqueeze(2) / (att_weight*mask.unsqueeze(2)).sum(1,keepdim=True)
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
		self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=10)
		self.criterion = nn.NLLLoss()