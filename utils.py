from __future__ import print_function
import sys
import os
sys.dont_write_bytecode=True
import torch
import numpy as np
import math
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import random
from termcolor import colored
from geographic_utils import *


# seed = 10
# seed = 300
# seed - 10
def seed_everything(seed=100):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	torch.initial_seed()
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


seed_everything()

def get_features(sample, neighbors_dim, previous_sample=None):
	if not (neighbors_dim==1):
		sample = sample.transpose(0,1)
	theta_1 = get_heading(sample,previous_sample)
	theta_1 = theta_1.unsqueeze(-1)
	lat_1, lon_1 = sample[...,0], sample[...,1]
	lat_1, lon_1 = lat_1.unsqueeze(-1).expand(sample.size(0),sample.size(1),sample.size(1)), lon_1.unsqueeze(-1).expand(sample.size(0),sample.size(1),sample.size(1))
	lat_2, lon_2 = lat_1.transpose(1,2), lon_1.transpose(1,2)
	distance = equirectangular_distance(lat_1, lon_1, lat_2, lon_2)
	bearing = absolute_bearing(lat_1, lon_1, lat_2, lon_2)
	theta_1 = theta_1.expand(sample.size(0),sample.size(1),sample.size(1))
	theta_2 = theta_1.transpose(1,2)
	bearing = bearing-theta_1
	bearing[distance==distance.min()]=0
	bearing[bearing<0]+=360
	heading = theta_2-theta_1
	heading[heading<0]+=360
	if not (neighbors_dim==1):
		distance, bearing, heading = distance.transpose(0,1), bearing.transpose(0,1), heading.transpose(0,1)
	return distance, bearing, heading

class Plotter(object):
	def __init__(self,args):
		super(Plotter,self).__init__()
		plot_dir='plots/'
		self.train_plots = plot_dir+str(args.model)+'/'+'hsz_'+str(args.hidden_size)+'_bsz_' + str(args.batch_size) + '_lr_'+str(args.learning_rate) + '_feature_size_'+str(args.feature_size)+'_sequence_length_'+str(args.sequence_length)+'_prediction_length_'+str(args.prediction_length)+'_delta_vals_'+str(args.delta_bearing)+'_'+str(args.delta_heading)+'_'+str(args.criterion_type)+'.png'
		if not os.path.isdir(plot_dir+str(args.model)):
			os.makedirs(plot_dir+str(args.model))
		print(colored("Saving learning curve at %s" %(self.train_plots),"blue"))
		self.train=[]
		self.valid=[]
	def update(self,train_loss,valid_loss):
		self.train.append(train_loss)
		self.valid.append(valid_loss)
		fig = plt.figure()
		plt.plot(self.train, 'r-', label='mean displacement error (training)')
		plt.plot(self.valid, 'b-',label='mean displacement error (validation)')
		plt.legend()
		plt.xlabel("Epoch")
		plt.savefig(self.train_plots)
		plt.close()


