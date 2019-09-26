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

torch.manual_seed(10)

class modularLSTM(nn.Module):
	def __init__(self,feature_size,hidden_size):
		super(modularLSTM,self).__init__()
		self.lstm=nn.LSTMCell(feature_size,hidden_size)
	def forward(self,x,h_t,c_t):
		return self.lstm(x, (h_t,c_t))

class vlstm(nn.Module):
	def __init__(self,args):
		super(vlstm,self).__init__()
		self.lstm=nn.ModuleList([modularLSTM(args.feature_size, args.hidden_size) for _ in range(args.maxVessels)])
		self.sequence_length=args.sequence_length
		self.prediction_length=args.prediction_length
		self.hidden_size=args.hidden_size
		self.maxVessels=args.maxVessels
		self.min_dist=args.min_dist
		self.output_size=args.output_size
		self.delta_rb=args.delta_rb
		self.feature_size=args.feature_size
		self.linear=nn.Linear(self.hidden_size,self.feature_size)
	def forward(self,sequence,dist_matrix,rb_matrix,cog_matrix,seq_mask,maxval,minval,maxVessels):
		maxval,minval=maxval.to(device),minval.to(device)
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
				h_t[j],c_t[j]=self.lstm[j](previous_seq[:,j,...],h_t[j],c_t[j])
				output[:,j,i,...]=F.relu(self.linear(h_t[j])).clamp_(0,1)
		return output[...,:self.output_size]


