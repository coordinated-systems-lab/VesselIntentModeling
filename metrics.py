import warnings
warnings.filterwarnings("ignore")
import os
import sys
sys.dont_write_bytecode=True
import torch
import torch.nn as nn
import math
import numpy as np
from data import *
from utils import *
from termcolor import colored

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything()

# mean of square of error
class MSE(nn.Module):
	def __init__(self):
		super(MSE,self).__init__()
		self.loss_criterion=nn.MSELoss()
	def forward(self,pred,targets):
		return self.loss_criterion(pred,targets)

# MSE on displacement error calculated in km
class ADE(nn.Module):
	def __init__(self):
		super(ADE,self).__init__()
		self.deg2rad = math.pi/180
		self.radius_earth = 3440.1
		self.eps=1e-24
	def haversine_distance(self,a1,a2):
		a1_lat = a1[...,0].to(device)*self.deg2rad
		a2_lat = a2[...,0].to(device)*self.deg2rad
		a1_lon = a1[...,1].to(device)*self.deg2rad
		a2_lon = a2[...,1].to(device)*self.deg2rad
		x = torch.sin(a2_lon-a1_lon)*torch.cos(a1_lat+a2_lat)
		y = torch.cos(a1_lat)*torch.sin(a2_lat)-torch.sin(a1_lat)*torch.cos(a2_lat)*torch.cos(a2_lon-a1_lon)
		dist=torch.sqrt(x**2+y**2+self.eps)*self.radius_earth
		return dist
	def get_denom_val(self,targets):
		t = targets.clone().view(-1)
		return len(t[~(t==0)])/2
	def forward(self,pred,targets):
		err=torch.FloatTensor(1).to(device).fill_(0.0)
		valid_len = self.get_denom_val(targets)
		dist=self.haversine_distance(pred,targets).to(device)
		dist.pow_(2)
		dist=dist.view(-1)
		if not (valid_len==0):
			err=dist.sum()/valid_len
		return err

class rootADE(nn.Module):
	def __init__(self):
		super(rootADE,self).__init__()
		self.ade = ADE()
	def forward(self,pred,targets):
		return torch.sqrt(self.ade(pred,targets))

# final displacement error calculated in km
class FDE(nn.Module):
	def __init__(self):
		super(FDE,self).__init__()
		self.deg2rad=math.pi/180
		self.radius_earth=3440.1
	def haversine_distance(self,a1,a2):
		a1_lat = a1[...,0].to(device)*self.deg2rad
		a2_lat = a2[...,0].to(device)*self.deg2rad
		a1_lon = a1[...,1].to(device)*self.deg2rad
		a2_lon = a2[...,1].to(device)*self.deg2rad
		x = (a2_lon-a1_lon)*torch.cos((a1_lat+a2_lat)/2)
		y = (a2_lat-a1_lat)
		dist=torch.sqrt(x**2+y**2)*self.radius_earth
		return dist
	def get_denom_val(self,targets):
                t = targets.clone().view(-1)
                return len(t[~(t==0)])/2
	def forward(self,pred,targets):
		err=torch.FloatTensor(1).to(device).fill_(0.0)
		valid_len = self.get_denom_val(targets)
		dist = self.haversine_distance(pred,targets).to(device)
		dist = dist.view(-1)
		if not (valid_len==0):
			err=dist.sum()/valid_len
		return err
