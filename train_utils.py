from __future__ import print_function
import sys
import os
sys.dont_write_bytecode=True
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import *
from geographic_utils import *

seed_everything()

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def get_dirs(args):
	net_dir = str(args.model)+'/'
	data_dir = 'data/%02d/'%(args.zone)
	assert(os.path.isdir(data_dir))
	if not os.path.isdir(net_dir):
		os.makedirs(net_dir)
	return net_dir, data_dir

def get_batch(batch):
	batch = [tensor.to(device) for tensor in batch]
	if not len(batch[0].size())==4:
		batch = [tensor.unsqueeze(0) for tensor in batch]
	return batch

def predict(batch,net):
	sequence,target,dist_matrix,bearing_matrix,heading_matrix,ip_mask,op_mask,vessels = get_batch(batch)
	sequence, dist_matrix, bearing_matrix, heading_matrix = Variable(sequence, requires_grad=True), Variable(dist_matrix, requires_grad=True), Variable(bearing_matrix, requires_grad=True), Variable(heading_matrix, requires_grad=True)
	pred, temporal_attention_dict = net(sequence, dist_matrix,bearing_matrix,heading_matrix,ip_mask,op_mask)
	assert(not torch.isnan(pred).any())
	target_mask = op_mask.unsqueeze(-1).expand(target.size())
	target.data.masked_fill_(mask=~target_mask, value=float(0))
	pred.data.masked_fill_(mask=~target_mask, value=float(0))
	return pred, target, sequence[...,:2],temporal_attention_dict, vessels
