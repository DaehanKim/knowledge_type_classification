import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
		# get seq len, sort
		mask = ~torch.eq(x[:,:,0], torch.zeros(*x.size()[:-1]))
		seq_len = mask.sum(1).long()
		mask = mask[:,:seq_len.max()]
		seq_lengths, perm_idx = seq_len.sort(0, descending=True)
		# print(seq_lengths)
		x = x[perm_idx]
		# exit()
		# print(x.size())
		x = F.dropout(x, p = DROPOUT_RATE, training=self.training)		
		packed_input = pack_padded_sequence(x, seq_lengths.cpu().numpy(), batch_first = True)
		# print(packed_input.data.size())
		packed_out, ht = self.rnn(packed_input, self.hidden[:,:x.size(0),:])
		padded_output, input_sizes = pad_packed_sequence(packed_out, batch_first= True)
		# print(input_sizes)
		# print(padded_output.size())
		att_logits =  self.att_layer(padded_output)  
		att_weight = F.softmax(att_logits, dim = 1)
		# print(att_weight.size(), mask.size())
		att_weight = att_weight*mask.unsqueeze(2) / (att_weight*mask.unsqueeze(2)).sum(1,keepdim=True)
		# print(att_weight[:10,:,0])
		# exit()
		# + ~mask.unsqueeze(2)*SOFTMAX_PAD_CONSTANT
		att_applied = (att_weight * padded_output).sum(1)
		logit = self.fc(att_applied)
		return logit, perm_idx

	def init_hidden(self):
		return nn.Parameter(torch.zeros(NUM_LAYER*(int(BIDIRECTIONAL)+1), BATCH_SIZE, HIDDEN_DIM)) 

class GRU_ATT_WRAP:
	def __init__(self, model):
		self.model = model
		self.model.hidden = self.model.init_hidden()
		self.optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
		self.scheduler = ReduceLROnPlateau(self.optimizer, mode = 'min', patience=10)
		self.criterion = nn.NLLLoss()