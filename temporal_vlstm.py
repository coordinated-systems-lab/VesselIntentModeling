import sys
sys.dont_write_bytecode=True
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import time
import math
from utils import *

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything()

class modularLSTM(nn.Module):
	def __init__(self,feature_size,hidden_size):
		super(modularLSTM,self).__init__()
		self.lstm=nn.LSTMCell(feature_size,hidden_size)
	def forward(self,x,h_t,c_t):
		return self.lstm(x, (h_t,c_t))

class softAttention(nn.Module):
	def __init__(self,hidden_size,sequence_length):
		super(softAttention,self).__init__()
		self.softmax=nn.Softmax(dim=1)
		self.sequence_length=sequence_length
		self.linear=nn.Linear(2*hidden_size,hidden_size)
		self.tanh=nn.Tanh()
	def compute_score(self,hidden_encoder,hidden_decoder):
		score=torch.FloatTensor(hidden_encoder.size(0),self.sequence_length).to(device)
		for i in range(self.sequence_length):
			score[:,i,...]=torch.bmm(hidden_encoder[:,i,...].unsqueeze(1),hidden_decoder.unsqueeze(2)).view(hidden_encoder.size(0))
		return score
	def forward(self,hidden_encoder,hidden_decoder,sequence_mask):
		alpha = self.softmax(self.compute_score(hidden_encoder,hidden_decoder)*sequence_mask)
		context_vector=torch.bmm(alpha.unsqueeze(1),hidden_encoder).squeeze(1)
		return self.tanh(self.linear(torch.stack((context_vector,hidden_decoder)).view(hidden_encoder.size(0),-1))),context_vector

	
class temporal_vlstm(nn.Module):
	def __init__(self,args):
		super(temporal_vlstm,self).__init__()
		self.lstm=nn.ModuleList([modularLSTM(args.feature_size, args.hidden_size) for _ in range(args.maxVessels)])
		self.sequence_length=args.sequence_length
		self.prediction_length=args.prediction_length
		self.hidden_size=args.hidden_size
		self.maxVessels=args.maxVessels
		self.min_dist=args.min_dist
		self.output_size=args.output_size
		self.feature_size=args.feature_size
		self.softAttention=nn.ModuleList([softAttention(args.hidden_size,args.sequence_length) for _ in range(args.maxVessels)])
		self.linear=nn.Linear(self.hidden_size,self.feature_size)
	def forward(self,sequence,dist_matrix,rb_matrix,cog_matrix,seq_mask,maxval,minval,maxVessels,return_context=False):
		if return_context:
			contextVectorDict={}
		maxval, minval = maxval.unsqueeze(1),minval.unsqueeze(1)
		maxval, minval = maxval.expand(sequence[:,:,-1,...].size()),minval.expand(sequence[:,:,-1,...].size())
		batchSize=sequence.size(0)
		h_t=[Variable(torch.zeros(batchSize,self.hidden_size),requires_grad=True).to(device) for _ in range(maxVessels)]
		c_t=[Variable(torch.zeros(batchSize,self.hidden_size),requires_grad=True).to(device) for _ in range(maxVessels)]
		encodedInput=torch.FloatTensor(batchSize,maxVessels,self.sequence_length,self.hidden_size).to(device)
		for i in range(self.sequence_length):
			for j in range(maxVessels):
				h_t[j],c_t[j]=self.lstm[j](sequence[:,j,i,...],h_t[j],c_t[j])
				encodedInput[:,j,i,...]=h_t[j]
		output=torch.FloatTensor(batchSize,maxVessels,self.prediction_length,self.feature_size).to(device)
		for i in range(self.prediction_length):
			if(i==0):
				previous_seq=torch.tensor(sequence[:,:,-1,...].data).float().to(device)
				previous_seq*=(maxval-minval)
				previous_seq+=minval
			else:
				previous_seq=torch.tensor(output[:,:,(i-1),...].data).float().to(device)
				previous_seq*=(maxval-minval)
				previous_seq+=minval
			for j in range(maxVessels):
				h_t[j],context_vector_j=self.softAttention[j](encodedInput[:,j,...],h_t[j],seq_mask[:,j,...])
				if return_context:
					contextVectorDict[(j,i)] = context_vector_j 
				h_t[j],c_t[j]=self.lstm[j](previous_seq[:,j,...],h_t[j],c_t[j])
				output[:,j,i,...]=F.relu(self.linear(h_t[j]))
		if not (return_context):
			return output[...,:self.output_size].clamp_(0,1)
		else:
			return output[...,:self.output_size].clamp_(0,1), contextVectorDict


