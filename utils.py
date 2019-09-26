from __future__ import print_function
import sys
from scipy import stats
sys.dont_write_bytecode=True
import scipy.ndimage
import torch
from torch.autograd import Variable
import math
import numpy as np
import random
from termcolor import colored
import matplotlib
matplotlib.use("agg")
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import glob
import os

deg2rad = math.pi/180
radius_earth = 3440.1

def seed_everything(seed=10):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	torch.initial_seed()
	torch.manual_seed(seed)

seed_everything()

class progressbar(object):
	def __init__(self,total,width=50,output=sys.stderr,sym='='):
		self.total=total
		self.width=width
		self.output=output
		self.symbol=sym
		self.current=0
	def __call__(self,loss=0):
		self.current+=1
		filled_len=int(round(self.width*self.current / float(self.total)))
		percents = round(100.0 * self.current / float(self.total),1)
		if(loss==0):
			bar = '[' +self.symbol * (filled_len-1) + '>' + ' ' * (self.width - filled_len) + ']' + ' ' + str(self.current) + '/' + str(self.total)
		else:
			bar = '[' +self.symbol * (filled_len-1) + '>' + ' ' * (self.width - filled_len) + ']' + ' '+ str(self.current) + '/' + str(self.total) + 'loss: ' + str(np.round(loss.item(),3))
		self.output.write('\r {}'.format(bar))
		sys.stdout.flush()

def transfer_weights(transfer_net,net):
	params1 = dict(transfer_net.named_parameters())
	dict_params = dict(net.named_parameters())
	for name in dict_params:
		if name in params1.keys():
			dict_params[name].data.copy_(params1[name])
 
def get_haversine_distance(lat1,lon1,lat2,lon2):
	dlat = abs(lat1-lat2)
	dlon = abs(lon1-lon2)
	al = torch.sin(dlat/2)**2+torch.cos(lat1)*torch.cos(lat2)*(torch.sin(dlon/2)**2)
	d=torch.atan2(torch.sqrt(al),torch.sqrt(1-al))
	d*=(2*6371)
	return d

def get_absolute_bearing(lat,lon):
	x = radius_earth*torch.cos(lat)*torch.cos(lon)
	y = radius_earth*torch.cos(lat)*torch.cos(lon)
	return torch.atan2(y,x)*(180/math.pi)

def get_relative_bearing(heading,lat,lon):
	abs_bearing = get_absolute_bearing(lat,lon)
	return heading-abs_bearing
	
def get_feature_matrices(sample,delta_rb,delta_cog,neighbors_dim=0):
	if not (neighbors_dim==1):
		sample=sample.transpose(0,1)
	init_bearing=0
	theta_1 = sample[...,-1].unsqueeze(-1)
	lat_1 = sample[...,0]
	lon_1 = sample[...,1]
	lat_1 = lat_1*deg2rad
	lon_1 = lon_1*deg2rad
	lat_1 = lat_1.unsqueeze(-1)
	lon_1 = lon_1.unsqueeze(-1)
	lat_1 = lat_1.expand(sample.size(0),sample.size(1),sample.size(1))
	lat_2 = lat_1.transpose(1,2)
	lon_1 = lon_1.expand(sample.size(0),sample.size(1),sample.size(1))
	lon_2 = lon_1.transpose(1,2)
	dlon = lon_2-lon_1
	dlat = lat_2-lat_1
	x = torch.sin(dlon)*torch.cos(lat_2)
	y = torch.cos(lat_1)*torch.sin(lat_2)-torch.sin(lat_1)*torch.cos(lat_2)*torch.cos(dlon)
	bearing=(180/math.pi)*torch.atan2(x,y)
	dist = torch.sqrt(x**2+y**2)*radius_earth
	bearing[bearing<0]+=360
	theta_1 = theta_1.expand(sample.size(0),sample.size(1),sample.size(1))
	theta_1[theta_1<0]+=360
	theta_2=theta_1.transpose(1,2)
	bearing-=theta_1
	bearing[dist==0] = 0 
	cog_matrix=theta_2-theta_1
	cog_matrix[cog_matrix<0]+=360
	bearing[bearing<0]+=360
	bearing = torch.floor(bearing/delta_rb)
	cog_matrix=torch.floor(cog_matrix/delta_cog)
	bearing.clamp_(0,int(360/delta_rb)-1)
	cog_matrix.clamp_(0,int(360/delta_cog)-1)
	# print(torch.unique(cog_matrix))
	# input("- - - - - - - - - - - - - - - - - - - - - -")
	if not (neighbors_dim==1):
		dist, bearing, cog_matrix = dist.transpose(0,1),bearing.transpose(0,1),cog_matrix.transpose(0,1)
	return dist, bearing, cog_matrix

def convert_tensor_to_numpy(tensor_):
	return np.array(tensor_.clone().detach().cpu().numpy())

def varied_normal_init(tensor, a=2.0, b=1.0, std=1.0):
	nn.init.normal_(tensor[:int(0.5*tensor.size(0))], a, std)
	nn.init.normal_(tensor[int(0.5*tensor.size(0)):], b, std)

def varied_constant_init(tensor, a=3.0, b=2.0):
	nn.init.constant_(tensor[:int(0.5*tensor.size(0))], a)
	nn.init.constant_(tensor[int(0.5*tensor.size(0)):], b)

def unnormalize(seq,maxval,minval):
	device=seq.get_device()
	numftr=seq.size(-1)
	maxval, minval = maxval[...,:numftr].unsqueeze(1).expand(seq.size(0),seq.size(1),numftr).unsqueeze(2).expand(seq.size(0),seq.size(1),seq.size(2),numftr), minval[...,:numftr].unsqueeze(1).expand(seq.size(0),seq.size(1),numftr).unsqueeze(2).expand(seq.size(0),seq.size(1),seq.size(2),numftr)
	seq_ = torch.FloatTensor(seq.size()).to(device)
	seq_ = seq*(maxval-minval)
	seq_+=minval
	return seq_

def log_loss_value(loss,args):
	df = pd.read_csv('loss.csv',header=0)
	if df.empty or df.loc[((df['sequence length']==args.sequence_length) & (df['prediction length']==args.prediction_length))].empty:
		df_current={}
		df_current['sequence length']=args.sequence_length
		df_current['prediction length']=args.prediction_length
		df_current['best validation loss']=loss
		df_current['hidden size'] = args.hidden_size
		df = df.append(df_current,ignore_index=True)
		df.to_csv('loss.csv',index=False)
	else:
		df_ = df.loc[((df['sequence length']==args.sequence_length) & (df['prediction length']==args.prediction_length))]
		if (df_['best validation loss']>loss).all():
			df = df[~df.isin(df_)]
			df_current={}
			df_current['sequence length']=args.sequence_length
			df_current['prediction length']=args.prediction_length
			df_current['best validation loss']=loss
			df_current['hidden size'] = args.hidden_size
			df = df.append(df_current,ignore_index=True)
			df.to_csv('loss.csv',index=False)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def domain_initialization(domain,delta_rb,delta_cog,param_domain):
	with torch.no_grad():
		domain[:,:int(int(360/delta_rb)/2)].data.copy_(domain[:,:int(int(360/delta_rb)/2)].uniform_(param_domain).sort(descending=True)[0])
		domain[:,int(int(360/delta_rb)/2):].data.copy_(domain[:,int(int(360/delta_rb)/2):].uniform_(param_domain).sort()[0])
