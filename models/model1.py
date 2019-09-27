from __future__ import print_function
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
		h_t,c_t = self.lstm(x, (h_t,c_t))
		return (h_t,c_t)

class hardwiredAttention(nn.Module):
	def __init__(self,min_dist,delta_rb,delta_cog):
		super(hardwiredAttention,self).__init__()
		self.domain=nn.Parameter(torch.FloatTensor(int(360/delta_cog),int(360/delta_rb)).to(device),requires_grad=True)
		self.relu=nn.ReLU()
	def forward(self,h_t,r_mat,d_mat,mask,cog_mat):
		orig_size=r_mat.size()
		mul1 = mask.unsqueeze(-1).expand(orig_size)*mask.unsqueeze(-1).expand(orig_size).transpose(1,2)
		mul2 = self.domain[cog_mat.contiguous().view(-1).long(),r_mat.contiguous().view(-1).long()]
		wts=self.relu(mul2.view(orig_size)-d_mat)
		weights=torch.mul(mul1,wts)
		weights/=weights.data.max()
		weightedHidden=torch.matmul(weights.unsqueeze(2),h_t.transpose(0,1).unsqueeze(1)).squeeze(2)
		return weightedHidden

class softAttention(nn.Module):
	def __init__(self,hidden_size,sequence_length):
		super(softAttention,self).__init__()
		self.softmax=nn.Softmax(dim=1)
		self.sequence_length=sequence_length
		self.linear=nn.Linear(2*hidden_size,hidden_size)
		self.tanh=nn.Tanh()
	def compute_score(self,hidden_encoder,hidden_decoder):
		score = torch.matmul(hidden_encoder.unsqueeze(2),hidden_decoder.unsqueeze(1).repeat(1,hidden_encoder.size(1),1).unsqueeze(-1)).view(hidden_encoder.size(0),hidden_encoder.size(1))
		return score
	def forward(self,hidden_encoder,hidden_decoder,sequence_mask):
		alpha=self.softmax(self.compute_score(hidden_encoder,hidden_decoder)*sequence_mask)
		context_vector=torch.matmul(alpha.unsqueeze(1),hidden_encoder).squeeze(1)
		return self.tanh(self.linear(torch.stack((context_vector,hidden_decoder)).view(hidden_encoder.size(0),-1))),context_vector
	
class model1(nn.Module):
	def __init__(self,args):
		super(model1,self).__init__()
		self.lstm=nn.ModuleList([modularLSTM(args.feature_size, args.hidden_size) for _ in range(args.maxVessels)])
		self.sequence_length=args.sequence_length
		self.prediction_length=args.prediction_length
		self.hidden_size=args.hidden_size
		self.maxVessels=args.maxVessels
		self.min_dist=args.min_dist
		self.output_size=args.output_size
		self.delta_cog=args.delta_cog
		self.delta_rb=args.delta_rb
		self.feature_size=args.feature_size
		self.softAttention=nn.ModuleList([softAttention(args.hidden_size,args.sequence_length) for _ in range(args.maxVessels)])
		self.hardwiredAttention=hardwiredAttention(self.min_dist,self.delta_rb,self.delta_cog)
		self.linear = nn.Linear(self.hidden_size,self.feature_size)
		self.relu=nn.ReLU()
	def forward(self,sequence,dist_matrix,rb_matrix,cog_matrix,seq_mask,maxval,minval,maxVessels,return_context=False):
		if return_context:
			contextVectorDict={}
		maxval,minval = reshape_normalizing_parameters(maxval,minval,sequence[:,:,-1,...].size())
		batchSize=sequence.size(0)
		h_t=[Variable(torch.zeros(batchSize,self.hidden_size),requires_grad=True).to(device) for _ in range(maxVessels)]
		c_t=[Variable(torch.zeros(batchSize,self.hidden_size),requires_grad=True).to(device) for _ in range(maxVessels)]
		encodedInput=torch.FloatTensor(batchSize,maxVessels,self.sequence_length,self.hidden_size).to(device)
		for i in range(self.sequence_length):
			weightedHidden=self.hardwiredAttention(torch.stack(h_t),rb_matrix[:,:,i,...],dist_matrix[:,:,i,...],seq_mask[:,:,i],cog_matrix[:,:,i,:])
			for j in range(maxVessels):
				h_t[j],c_t[j]=self.lstm[j](sequence[:,j,i,...],weightedHidden[:,j,...],c_t[j])
				encodedInput[:,j,i,...]=h_t[j]
		output=torch.FloatTensor(batchSize,maxVessels,self.prediction_length,self.feature_size).to(device)
		for i in range(self.prediction_length):
			if(i==0):
				weightedHidden=self.hardwiredAttention(torch.stack(h_t),rb_matrix[:,:,-1,...],dist_matrix[:,:,-1,...],seq_mask[:,:,-1,...],cog_matrix[:,:,-1,:])
				previous_seq=get_previous_sequence(sequence[:,:,-1,...].data,maxval,minval)
			else:
				previous_seq=get_previous_sequence(output[:,:,(i-1),...].data,maxval,minval)
				dist_mat,rb_mat,cog_mat = get_feature_matrices(previous_seq,self.delta_rb,self.delta_cog,1)
				weightedHidden=self.hardwiredAttention(torch.stack(h_t),rb_mat,dist_mat,seq_mask[:,:,-1,...],cog_mat)
			for j in range(maxVessels):
				h_t[j],c_t[j]=self.lstm[j](previous_seq[:,j,...],weightedHidden[:,j,...],c_t[j]) 
				h_t[j], context_vector_j = self.softAttention[j](encodedInput[:,j,...],h_t[j],seq_mask[:,j,:]) 
				if return_context:
					contextVectorDict[(j,i)] = context_vector_j 
				output[:,j,i,...] = self.relu(self.linear(h_t[j]))
		if not (return_context):
			return output[...,:self.output_size]
		else:
			return output[...,:self.output_size], contextVectorDict


def reshape_normalizing_parameters(maxval,minval,size):
	with torch.no_grad():
		val1,val2=maxval.unsqueeze(1).expand(size), minval.unsqueeze(1).expand(size)
	return val1,val2

def get_previous_sequence(seq_data,maxval,minval):
	with torch.no_grad():
		previous_sequence=torch.tensor(seq_data).float().to(device)
		previous_sequence*=(maxval-minval)
		previous_sequence+=minval
	return previous_sequence
