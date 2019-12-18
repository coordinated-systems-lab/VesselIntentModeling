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
from attention import *

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything()

class modularLSTM(nn.Module):
	def __init__(self,feature_size,hidden_size):
		super(modularLSTM,self).__init__()
		self.encoder=nn.LSTMCell(feature_size,hidden_size)
	def forward(self,x,h_t,c_t):
		h_t, c_t = self.encoder(x, (h_t,c_t))
		return h_t,c_t

class vanilla_lstm(nn.Module):
	def __init__(self,sequence_length,prediction_length,feature_size,hidden_size,delta_bearing,delta_heading,domain_parameter,normalize_domain_weights=False):
		super(vanilla_lstm,self).__init__()
		self.encoder = modularLSTM(feature_size, hidden_size)
		self.sequence_length=sequence_length
		self.prediction_length=prediction_length
		self.hidden_size=hidden_size
		self.feature_size=feature_size
		self.out = nn.Sequential(nn.Linear(self.hidden_size,self.feature_size))
	def init_states(self, batch_size, num_vessels):
		h_t = [Variable(torch.zeros(batch_size,self.hidden_size),requires_grad=True).to(device) for _ in range(num_vessels)]
		c_t = [Variable(torch.zeros(batch_size,self.hidden_size),requires_grad=True).to(device) for _ in range(num_vessels)]
		return h_t,c_t
	def forward(self,sequence,dist_matrix,bearing_matrix,heading_matrix,seq_mask,op_mask):
		alignmentVector = {}
		batch_size, num_vessels,_,_ = list(sequence.size())
		h_t, c_t = self.init_states(batch_size, num_vessels)
		for i in range(self.sequence_length):
			for j in range(num_vessels):
				h_t[j],c_t[j] = self.encoder(sequence[:,j,i,...],h_t[j],c_t[j])
		output = torch.FloatTensor(batch_size, num_vessels, self.prediction_length, self.feature_size).to(device)
		previous_sequence = sequence[:,:,-1,...].clone()
		for i in range(self.prediction_length):
			current_sequence = []
			for j in range(num_vessels):
				h_t[j], c_t[j] = self.encoder(previous_sequence[:,j,...],h_t[j],c_t[j])
				output_ped = self.out(h_t[j])
				current_sequence+=[output_ped.unsqueeze(1)]
			previous_sequence = torch.cat(current_sequence,dim=1)
			output[:,:,i,...] = previous_sequence	
		return output, alignmentVector

class temporal_model(nn.Module):
	def __init__(self,sequence_length,prediction_length,feature_size,hidden_size,delta_bearing,delta_heading,domain_parameter):
		super(temporal_model,self).__init__()
		self.encoder = modularLSTM(feature_size, hidden_size)
		self.decoder = modularLSTM(feature_size,hidden_size)
		self.sequence_length=sequence_length
		self.prediction_length=prediction_length
		self.hidden_size=hidden_size
		self.feature_size=feature_size
		self.temporalAttention = temporalAttention(hidden_size,sequence_length)
		self.delta_bearing=delta_bearing
		self.delta_heading=delta_heading
		self.out = nn.Sequential(nn.Linear(self.hidden_size,self.feature_size),nn.ReLU())
	def init_states(self, batch_size, num_vessels):
		h_t = [Variable(torch.zeros(batch_size,self.hidden_size),requires_grad=True).to(device) for _ in range(num_vessels)]
		c_t = [Variable(torch.zeros(batch_size,self.hidden_size),requires_grad=True).to(device) for _ in range(num_vessels)]
		return h_t, c_t
	def encode(self, h_t, c_t, sequence, distance, bearing, heading, input_mask, num_vessels): 
		for j in range(num_vessels):
			hidden_ped, c_t[j] = self.encoder(sequence[:,j,...],h_t[j], c_t[j])
			hidden_ped.data.masked_fill_(mask=~input_mask[:,j].unsqueeze(-1).expand_as(hidden_ped),value=float(0))
			h_t[j] = hidden_ped
		return torch.stack(h_t), h_t, c_t
	def decode(self,h_t,c_t,previous_sequence,encoded_input,input_mask,alignmentVector,prediction_timestep,batch_size,num_vessels,op_mask):
		prediction = torch.FloatTensor(batch_size, num_vessels, self.feature_size).to(device)
		prediction_mask = op_mask.unsqueeze(-1).expand(prediction.size())
		for j in range(num_vessels):
			attended_ped, alignment_vector = self.temporalAttention(encoded_input[:,j,...],h_t[j],input_mask[:,j,:])
			hidden_ped, c_t[j] = self.decoder(previous_sequence[:,j,...],attended_ped,c_t[j])
			hidden_ped.data.masked_fill_(mask=~op_mask[:,j].unsqueeze(-1).expand_as(hidden_ped),value=float(0))
			h_t[j] = hidden_ped
			prediction[:,j,...] = self.out(h_t[j])
			alignmentVector[(j,prediction_timestep)] = alignment_vector
		prediction.data.masked_fill_(mask=~prediction_mask,value=float(0))
		return h_t, c_t, prediction, alignmentVector 
	def forward(self,sequence,dist_matrix,bearing_matrix,heading_matrix,seq_mask,op_mask):
		alignmentVector = {}
		batch_size, num_vessels,_,_ = list(sequence.size())
		h_t, c_t = self.init_states(batch_size, num_vessels)
		encoded_input = []
		for i in range(self.sequence_length):
			encoded_hidden, h_t, c_t = self.encode(h_t, c_t, sequence[:,:,i,...], dist_matrix[:,:,i,...], bearing_matrix[:,:,i,...], heading_matrix[:,:,i,...], seq_mask[:,:,i,...],num_vessels)
			encoded_input+=[encoded_hidden.unsqueeze(2)]
		encoded_input = torch.cat(encoded_input,dim=2)
		encoded_input = encoded_input.transpose(0,1)
		previous_sequence = sequence[:,:,-1,...]
		output = torch.FloatTensor(batch_size, num_vessels, self.prediction_length, self.feature_size).to(device)
		for i in range(self.prediction_length):
			h_t, c_t, previous_sequence, alignmentVector = self.decode(h_t,c_t,previous_sequence,encoded_input,seq_mask,alignmentVector,i,batch_size,num_vessels,seq_mask[:,:,-1])
			output[:,:,i,...] = previous_sequence
			hidden_state = torch.stack(h_t)
		return output, alignmentVector

class spatial_model(nn.Module):
	def __init__(self,sequence_length,prediction_length,feature_size,hidden_size,delta_bearing,delta_heading,domain_parameter):
		super(spatial_model,self).__init__()
		self.encoder = modularLSTM(feature_size, hidden_size)
		self.decoder = modularLSTM(feature_size,hidden_size)
		self.sequence_length=sequence_length
		self.prediction_length=prediction_length
		self.hidden_size=hidden_size
		self.feature_size=feature_size
		self.spatialAttention = spatialAttention(domain_parameter,hidden_size,delta_bearing,delta_heading)
		self.delta_bearing=delta_bearing
		self.delta_heading=delta_heading
		self.out = nn.Sequential(nn.Linear(self.hidden_size,self.feature_size),nn.ReLU())
	def init_states(self, batch_size, num_vessels):
		h_t = [Variable(torch.zeros(batch_size,self.hidden_size),requires_grad=True).to(device) for _ in range(num_vessels)]
		c_t = [Variable(torch.zeros(batch_size,self.hidden_size),requires_grad=True).to(device) for _ in range(num_vessels)]
		return h_t, c_t
	def encode(self, h_t, c_t, sequence, distance, bearing, heading, input_mask, num_vessels): 
		hidden_state = torch.stack(h_t)
		weighted_hidden = self.spatialAttention(hidden_state, distance, bearing, heading, input_mask)
		for j in range(num_vessels):
			hidden_ped, c_t[j] = self.encoder(sequence[:,j,...],weighted_hidden[:,j,...], c_t[j])
			hidden_ped.data.masked_fill_(mask=~input_mask[:,j].unsqueeze(-1).expand_as(hidden_ped),value=float(0))
			h_t[j] = hidden_ped
		return weighted_hidden, h_t, c_t
	def decode(self,h_t,c_t,previous_sequence,weighted_hidden,encoded_input,input_mask,alignmentVector,prediction_timestep,batch_size,num_vessels,op_mask):
		prediction = torch.FloatTensor(batch_size, num_vessels, self.feature_size).to(device)
		prediction_mask = op_mask.unsqueeze(-1).expand(prediction.size())
		for j in range(num_vessels):
			hidden_ped, c_t[j] = self.decoder(previous_sequence[:,j,...],weighted_hidden[:,j,...],c_t[j])
			hidden_ped.data.masked_fill_(mask=~op_mask[:,j].unsqueeze(-1).expand_as(hidden_ped),value=float(0))
			h_t[j] = hidden_ped
			prediction[:,j,...] = self.out(h_t[j])
		prediction.data.masked_fill_(mask=~prediction_mask,value=float(0))
		return h_t, c_t, prediction, alignmentVector 
	def forward(self,sequence,dist_matrix,bearing_matrix,heading_matrix,seq_mask,op_mask):
		alignmentVector = {}
		batch_size, num_vessels,_,_ = list(sequence.size())
		h_t, c_t = self.init_states(batch_size, num_vessels)
		encoded_input = []
		for i in range(self.sequence_length):
			encoded_hidden, h_t, c_t = self.encode(h_t, c_t, sequence[:,:,i,...], dist_matrix[:,:,i,...], bearing_matrix[:,:,i,...], heading_matrix[:,:,i,...], seq_mask[:,:,i,...],num_vessels)
			encoded_input+=[encoded_hidden.unsqueeze(2)]
		encoded_input = torch.cat(encoded_input,dim=2)
		previous_sequence = sequence[:,:,-1,...]
		output = torch.FloatTensor(batch_size, num_vessels, self.prediction_length, self.feature_size).to(device)
		weighted_hidden=self.spatialAttention(torch.stack(h_t),dist_matrix[:,:,-1,...],bearing_matrix[:,:,-1,...],heading_matrix[:,:,-1,...],seq_mask[:,:,-1])
		for i in range(self.prediction_length):
			h_t, c_t, previous_sequence, alignmentVector = self.decode(h_t,c_t,previous_sequence,weighted_hidden,encoded_input,seq_mask,alignmentVector,i,batch_size,num_vessels,seq_mask[:,:,-1])
			output[:,:,i,...] = previous_sequence
			if not (i==0):
				distance, bearing, heading = get_features(previous_sequence,1,output[:,:,i-1,...])
			else:
					distance, bearing, heading = get_features(previous_sequence,1,sequence[:,:,-1,...])
			hidden_state = torch.stack(h_t)
			weighted_hidden = self.spatialAttention(hidden_state, distance, bearing, heading, seq_mask[:,:,-1])
		return output, alignmentVector


class spatial_temporal_model(nn.Module):
	def __init__(self,sequence_length,prediction_length,feature_size,hidden_size,delta_bearing,delta_heading,domain_parameter):
		super(spatial_temporal_model,self).__init__()
		self.encoder = modularLSTM(feature_size, hidden_size)
		self.decoder = modularLSTM(feature_size,hidden_size)
		self.sequence_length=sequence_length
		self.prediction_length=prediction_length
		self.hidden_size=hidden_size
		self.feature_size=feature_size
		self.temporalAttention = temporalAttention(hidden_size,sequence_length)
		self.spatialAttention = spatialAttention(domain_parameter,hidden_size,delta_bearing,delta_heading)
		self.delta_bearing=delta_bearing
		self.delta_heading=delta_heading
		self.out = nn.Sequential(nn.Linear(self.hidden_size,self.feature_size),nn.ReLU())
	def init_states(self, batch_size, num_vessels):
		h_t = [Variable(torch.zeros(batch_size,self.hidden_size),requires_grad=True).to(device) for _ in range(num_vessels)]
		c_t = [Variable(torch.zeros(batch_size,self.hidden_size),requires_grad=True).to(device) for _ in range(num_vessels)]
		return h_t, c_t
	def encode(self, h_t, c_t, sequence, distance, bearing, heading, input_mask, num_vessels): 
		hidden_state = torch.stack(h_t)
		weighted_hidden = self.spatialAttention(hidden_state, distance, bearing, heading, input_mask)
		for j in range(num_vessels):
			hidden_ped, c_t[j] = self.encoder(sequence[:,j,...],weighted_hidden[:,j,...], c_t[j])
			hidden_ped.data.masked_fill_(mask=~input_mask[:,j].unsqueeze(-1).expand_as(hidden_ped),value=float(0))
			h_t[j] = hidden_ped
		return weighted_hidden, h_t, c_t
	def decode(self,h_t,c_t,previous_sequence,weighted_hidden,encoded_input,input_mask,alignmentVector,prediction_timestep,batch_size,num_vessels,op_mask):
		prediction = torch.FloatTensor(batch_size, num_vessels, self.feature_size).to(device)
		prediction_mask = op_mask.unsqueeze(-1).expand(prediction.size())
		for j in range(num_vessels):
			attended_weighted_ped, alignment_vector = self.temporalAttention(encoded_input[:,j,...],weighted_hidden[:,j,...],input_mask[:,j,:])
			hidden_ped, c_t[j] = self.decoder(previous_sequence[:,j,...],attended_weighted_ped,c_t[j])
			hidden_ped.data.masked_fill_(mask=~op_mask[:,j].unsqueeze(-1).expand_as(hidden_ped),value=float(0))
			h_t[j] = hidden_ped
			prediction[:,j,...] = self.out(h_t[j])
			alignmentVector[(j,prediction_timestep)] = alignment_vector
		prediction.data.masked_fill_(mask=~prediction_mask,value=float(0))
		return h_t, c_t, prediction, alignmentVector 
	def forward(self,sequence,dist_matrix,bearing_matrix,heading_matrix,seq_mask,op_mask):
		alignmentVector = {}
		batch_size, num_vessels,_,_ = list(sequence.size())
		h_t, c_t = self.init_states(batch_size, num_vessels)
		encoded_input = []
		for i in range(self.sequence_length):
			encoded_hidden, h_t, c_t = self.encode(h_t, c_t, sequence[:,:,i,...], dist_matrix[:,:,i,...], bearing_matrix[:,:,i,...], heading_matrix[:,:,i,...], seq_mask[:,:,i,...],num_vessels)
			encoded_input+=[encoded_hidden.unsqueeze(2)]
		encoded_input = torch.cat(encoded_input,dim=2)
		previous_sequence = sequence[:,:,-1,...]
		output = torch.FloatTensor(batch_size, num_vessels, self.prediction_length, self.feature_size).to(device)
		weighted_hidden=self.spatialAttention(torch.stack(h_t),dist_matrix[:,:,-1,...],bearing_matrix[:,:,-1,...],heading_matrix[:,:,-1,...],seq_mask[:,:,-1])
		for i in range(self.prediction_length):
			h_t, c_t, previous_sequence, alignmentVector = self.decode(h_t,c_t,previous_sequence,weighted_hidden,encoded_input,seq_mask,alignmentVector,i,batch_size,num_vessels,seq_mask[:,:,-1])
			output[:,:,i,...] = previous_sequence
			if not (i==0):
				distance, bearing, heading = get_features(previous_sequence,1,output[:,:,i-1,...])
			else:
					distance, bearing, heading = get_features(previous_sequence,1,sequence[:,:,-1,...])
			hidden_state = torch.stack(h_t)
			weighted_hidden = self.spatialAttention(hidden_state, distance, bearing, heading, seq_mask[:,:,-1])
		return output, alignmentVector


