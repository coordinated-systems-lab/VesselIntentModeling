from __future__ import print_function
import sys
sys.dont_write_bytecode=True
import utm
import math
import torch.nn as nn
import numpy as np
from utils import *
from termcolor import colored

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

radius_earth = 3440.1

# need to uncomment one of these
#min_lat, max_lat, min_lon, max_lon = 32, 33, -118, -117 
min_lat, max_lat, min_lon, max_lon = 32, 35, -120, -117
min_lat , max_lat, min_lon, max_lon  = (math.pi/180)*min_lat, (math.pi/180)*max_lat, (math.pi/180)*min_lon, (math.pi/180)*max_lon

class displacement_error(nn.Module):
	def __init__(self,args,method="mean",interaction=False):
		super(displacement_error,self).__init__()
		self.dist = equirectangular_distance
		self.mse_loss = nn.MSELoss()
		self.delta_bearing = args.delta_bearing
		self.delta_heading = args.delta_heading
		self.interaction = interaction
		if method=="mean":
			self.error_func = mean_displacement_error
		elif method=="final":
			self.error_func = final_displacement_error
		elif method=="mean_squared":
			self.error_func = mean_squared_displacement_error
		self.feature_size=args.feature_size
	def forward(self, prediction, target, vessel_count):
		prediction_length=prediction.size(2)
		lat1, lon1, lat2, lon2 = prediction[...,0],prediction[...,1], target[...,0], target[...,1]
		lat1, lon1 = scale_values(lat1, lon1)
		lat2, lon2 = scale_values(lat2, lon2)
		lat1, lon1, lat2, lon2 = map(rad2deg, (lat1, lon1, lat2, lon2))
		dist = self.dist(prediction[...,0],prediction[...,1], target[...,0], target[...,1])
		dist_error = self.error_func(dist,vessel_count,prediction_length)
		if self.interaction:
			orient_error = self.compare_mat(prediction,target)
			error = (dist_error + orient_error)
		else:
			error = dist_error	
		return error, dist_error
	def get_features(self, prediction):
		lat_1 = prediction[...,0].unsqueeze(-1).expand(prediction.size(0), prediction.size(1), prediction.size(2), prediction.size(1))
		lon_1 =  prediction[...,1].unsqueeze(-1).expand(prediction.size(0), prediction.size(1), prediction.size(2), prediction.size(1))
		lat_2 = lat_1.transpose(1,3)
		lon_2 = lat_2.transpose(1,3)
		bearing = absolute_bearing(lat_1, lon_1, lat_2, lon_2)
		heading = get_heading(prediction) # batch_size x num_neighbors x prediction_length
		distance = equirectangular_distance(lat_1, lon_1, lat_2, lon_2)
		heading = heading.unsqueeze(-1).expand(bearing.size())
		bearing = bearing - heading 
		bearing[distance==distance.min()]=0
		heading = heading.transpose(1,3) - heading
		bearing[bearing<0]+=360
		heading[heading<0]+=360
		return distance, bearing, heading
	def compare_mat(self, prediction, target):
		dist1, brng1, hdng1 = self.get_features(prediction)
		dist2, brng2, hdng2 = self.get_features(target)
		dist_loss = self.mse_loss(dist2,dist1)
		brng_loss = self.mse_loss(brng2, brng1)
		hdng_loss = self.mse_loss(hdng2, hdng1)
		sum_mat_loss = dist_loss
		sum_mat_loss = (dist_loss + brng_loss + hdng_loss)/3
		return sum_mat_loss
		
def mean_displacement_error(distance, vessel_count, prediction_length):
	return distance.sum()/(vessel_count.sum()*prediction_length)

def final_displacement_error(distance, vessel_count, prediction_length):
	return distance[...,-1].sum()/(vessel_count.sum())

def mean_squared_displacement_error(distance, vessel_count, prediction_length):
	return (distance**2).sum()/(vessel_count.sum()*prediction_length)

def deg2rad(point):
	return (math.pi/180)*point

def rad2deg(point):
	point = (180/math.pi)*point
	return point

def haversine_distance(lat1, lon1, lat2, lon2):
	lat1, lon1 = scale_values(lat1, lon1)
	lat2, lon2 = scale_values(lat2, lon2)
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	a = (torch.sin(dlat/2))**2 + torch.cos(lat1)*torch.cos(lat2)*(torch.sin(dlon/2))**2
	a = a+(1e-24)
	c = 2*torch.atan2(torch.sqrt(a),torch.sqrt(1-a))
	d = radius_earth*c
	return d

def scale_values(lat, lon):
	lat = (max_lat-min_lat)*lat + min_lat
	lon = (max_lon-min_lon)*lat + min_lon
	return lat, lon

def absolute_bearing(lat1, lon1, lat2, lon2):
	lat1, lon1 = scale_values(lat1, lon1)
	lat2, lon2 = scale_values(lat2, lon2)
	dlon = lon2-lon1
	y = torch.sin(dlon)*torch.cos(lat2)
	x = torch.cos(lat1)*torch.sin(lat2) - torch.sin(lat1)*torch.cos(lat2)*torch.cos(dlon)
	x = x + (1e-15)
	y = y + (1e-15)
	bearing = torch.atan2(y,x)
	bearing = rad2deg(bearing)
	bearing[bearing<0]+=360
	return bearing

def get_heading(sample, prev_sample=None):
	n = sample.size(1)
	if (prev_sample is None):
		if len(sample.size())==3:
			prev_sample = sample[:-1,...]
			sample = sample[1:,...]
			heading = absolute_bearing(prev_sample[...,0],prev_sample[...,1],sample[...,0],sample[...,1])
			heading = torch.cat((heading[0,...].clone().unsqueeze(0),heading),dim=0)
		else:
			prev_sample = sample[:,:,:-1,...]
			sample = sample[:,:,1:,...]
			heading = absolute_bearing(prev_sample[...,0],prev_sample[...,1],sample[...,0],sample[...,1])
			heading = torch.cat((heading[:,:,0,...].clone().unsqueeze(2),heading),dim=2)
	else:
		heading = absolute_bearing(prev_sample[...,0],prev_sample[...,1],sample[...,0],sample[...,1])
	return heading

def fill_zeros(heading):
	heading_np = heading.detach().cpu().numpy()
	neighbors = heading_np.shape[1]
	slen = heading_np.shape[0]	
	for n in range(neighbors):
		if len(heading_np[:,n]==0)==0:
			continue
		idx = np.arange(slen)
		idx[heading_np[:,n]==0]=0
		idx = np.maximum.accumulate(idx,axis=0)
		heading_np[:,n] = heading_np[idx,n]
		if (heading_np[:,n]==0).any():
			idx = np.arange(slen)
			idx[heading_np[:,n]==0]=0
			idx = np.minimum.accumulate(idx[::-1],axis=0)
			heading_np[:,n] = heading_np[idx[::-1],n]	
	return torch.from_numpy(heading_np).float()

def equirectangular_distance(lat1, lon1, lat2, lon2):
	lat1, lon1 = scale_values(lat1, lon1)
	lat2, lon2 = scale_values(lat2, lon2)
	dlon = lon2-lon1
	dlat = lat2-lat1
	dist = (dlat)**2 + (dlon*torch.cos((lat1+lat2)/2))**2
	dist = radius_earth*torch.sqrt(dist+(1e-24))
	return dist


