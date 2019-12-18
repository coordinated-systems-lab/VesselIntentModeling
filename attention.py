from __future__ import print_function
import sys
sys.dont_write_bytecode=True

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything()

class spatialAttention(nn.Module):
	def __init__(self,domain_parameter,hidden_size,delta_bearing,delta_heading):
		super(spatialAttention,self).__init__()
		self.delta_bearing=delta_bearing
		self.delta_heading=delta_heading
		self.domain = nn.Parameter(torch.FloatTensor(int(360/delta_heading),int(360/delta_bearing)).to(device),requires_grad=True)
		self.relu=nn.Softplus()
		self.out = nn.Tanh()
	def forward(self, hidden_state, distance_matrix, bearing_matrix, heading_matrix,sequence_mask):
		num_vessels = hidden_state.size(0)
		weights = self.compute_weights(distance_matrix, bearing_matrix, heading_matrix, sequence_mask)
		weighted_hidden = torch.matmul(weights.unsqueeze(2),hidden_state.permute(1,0,2).unsqueeze(1).repeat(1,num_vessels,1,1)).squeeze(2)
		weighted_hidden = self.out(weighted_hidden)
		weighted_hidden.data.masked_fill_(mask=~sequence_mask.unsqueeze(-1).expand_as(weighted_hidden),value=float(0))
		return weighted_hidden
	def compute_weights(self, distance_matrix, bearing_matrix, heading_matrix, sequence_mask):
		idx1, idx2 = torch.floor(heading_matrix/self.delta_heading), torch.floor(bearing_matrix/self.delta_bearing)
		idx1, idx2 = idx1.clamp(0, int(360/self.delta_heading)-1), idx2.clamp(0, int(360/self.delta_bearing)-1)
		weights_mask = sequence_mask.unsqueeze(-1).expand(distance_matrix.size())
		weights_mask = torch.mul(weights_mask, weights_mask.transpose(1,2))
		distance_matrix.data.masked_fill_(mask=~weights_mask, value=float(1e+24))
		weights=self.relu(self.domain[idx1.long(),idx2.long()]-distance_matrix)
		return weights


class temporalAttention(nn.Module):
	def __init__(self,hidden_size,sequence_length):
		super(temporalAttention,self).__init__()
		self.softmax=nn.Softmax(dim=1)
		self.sequence_length=sequence_length
		self.linear=nn.Sequential(nn.Linear(2*hidden_size,hidden_size),nn.Tanh())
		#self.general_weights = nn.Parameter(Variable(torch.FloatTensor(hidden_size,hidden_size).to(device),requires_grad=True))
	def compute_score(self,hidden_encoder,hidden_decoder,sequence_mask):
		score = torch.bmm(hidden_encoder, hidden_decoder.unsqueeze(-1)).squeeze(-1)
		score.data.masked_fill_(mask=~sequence_mask, value=float(0))
		score = self.softmax(score)
		return score
	def forward(self,hidden_encoder,hidden_decoder,sequence_mask):
		if hasattr(self,'general_weights'):
			score = self.compute_score(hidden_encoder@self.general_weights,hidden_decoder,sequence_mask)
		else:
			score = self.compute_score(hidden_encoder,hidden_decoder,sequence_mask)
		context_vector=torch.bmm(score.unsqueeze(1), hidden_encoder)
		attention_features = [hidden_decoder, context_vector.squeeze(1)]
		attention_input = torch.cat(attention_features, dim=1)
		output = self.linear(attention_input)
		output.data.masked_fill_(mask=~sequence_mask[:,-1].unsqueeze(-1).expand_as(output),value=float(0))
		return output, score

