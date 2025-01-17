from __future__ import print_function
import sys
import os
sys.dont_write_bytecode=True
import torch
import math
import pandas as pd
import numpy as np
import glob
from termcolor import colored
from torch.utils.data import Dataset, DataLoader, random_split
from utils import *
from geographic_utils import *

class trajectory_dataset(Dataset):
	def __init__(self, data_dir, sequence_length, prediction_length, feature_size):
		super(trajectory_dataset, self).__init__()
		self.filenames = glob.glob(data_dir+'*.csv')
		self.sequence_length=sequence_length
		self.prediction_length=prediction_length
		self.feature_size=feature_size
		self.len = 0
		self.sequences = {}
		self.masks = {}
		self.vesselCount = {}
	#	self.shift=100
#		self.shift=self.sequence_length+self.prediction_length
		self.shift = self.sequence_length
		for f, filename in enumerate(self.filenames):
			df = self.load_df(filename)
			df = self.normalize(df)
			if not df.empty:
				self.get_file_samples(df,f)
	def load_df(self, filename):
		df = pd.read_csv(filename, header=0, usecols=['BaseDateTime','MMSI','LAT','LON','SOG','Heading'],parse_dates=['BaseDateTime'])
		df.sort_values(['BaseDateTime'],inplace=True)
		return df 
	def normalize(self, df):
		df['LAT'] = (math.pi/180)*df['LAT']
		df['LON'] = (math.pi/180)*df['LON']
		df = df.loc[(df['LAT']<=max_lat)&(df['LAT']>=min_lat)&(df['LON']<=max_lon)&(df['LON']>=min_lon)]
		if not df.empty:
			df['LAT'] = (df['LAT']-min_lat)/(max_lat-min_lat)
			df['LON'] = (df['LON']-min_lon)/(max_lon-min_lon)
			df['SOG'] = df['SOG']/22
			df['Heading'] = df['Heading']/360
			print(df['SOG'].min(),df['SOG'].max())
		return df
	def get_file_samples(self, df,f):
		j = 0
		timestamps = df['BaseDateTime'].unique()
		while not (j+self.sequence_length+self.prediction_length)>len(timestamps):
			frame_timestamps = timestamps[j:j+self.sequence_length+self.prediction_length]
			frame = df.loc[df['BaseDateTime'].isin(frame_timestamps)]
			if self._condition_time(frame_timestamps):
				cond_val, vessels = self._condition_vessels(frame)
				if cond_val:
					sys.stdout.write(colored("\rfile: {}/{} Sample: {} Num Vessels: {}".format(f+1,len(self.filenames),self.len,vessels),"blue"))
					self.sequences[self.len], self.masks[self.len], self.vesselCount[self.len] = self.get_sequence(frame)
					self.len+=1
			j+=self.shift	
	def _condition_time(self, timestamps):
		condition_satisfied=True
		diff_timestamps = np.amax(np.diff(timestamps).astype('float'))
		if diff_timestamps/(6e+10) > 1 or diff_timestamps/(8.64e+13)>=1:
			condition_satisfied = False
		return condition_satisfied
	def _condition_vessels(self, frame):
		condition_satisfied = True
		frame_timestamps = frame['BaseDateTime'].unique()[:self.sequence_length]
		frame = frame.loc[frame['BaseDateTime'].isin(frame_timestamps)]
		total_vessels = len(frame['MMSI'].unique())
		valid_vessels = [v for v in frame['MMSI'].unique() if not \
		abs(frame.loc[frame['MMSI']==v]['LAT'].diff()).max()<(1e-04) \
		and not abs(frame.loc[frame['MMSI']==v]['LON'].diff()).max()<(1e-04) and len(frame.loc[frame['MMSI']==v])==self.sequence_length]
		if (len(valid_vessels)<total_vessels) or total_vessels<=3:
			condition_satisfied=False
		return condition_satisfied,total_vessels
	def get_sequence(self,frame):
		frame = frame.values
		frameIDs = np.unique(frame[:,0]).tolist()
		input_frame = frame[np.isin(frame[:,0],frameIDs[:self.sequence_length])]
		vessels = np.unique(input_frame[:,1]).tolist()
		sequence=torch.FloatTensor(len(vessels),len(frameIDs),frame.shape[-1]-2)
		mask=torch.BoolTensor(len(vessels),len(frameIDs))
		for v, vessel in enumerate(vessels):
			vesselTraj = frame[frame[:,1]==vessel]
			vesselTrajLen = np.shape(vesselTraj)[0]
			vesselIDs = np.unique(vesselTraj[:,0])
			maskVessel = np.ones(len(frameIDs))
			if vesselTrajLen<(self.sequence_length+self.prediction_length):
				missingIDs=[f for f in frameIDs if not f in vesselIDs]
				maskVessel[vesselTrajLen:].fill(0.0)
				paddedTraj = np.zeros((len(missingIDs),np.shape(vesselTraj)[1]))
				vesselTraj=np.concatenate((vesselTraj,paddedTraj),axis=0)
				vesselTraj[vesselTrajLen:,0] = missingIDs
				vesselTraj[vesselTrajLen:,1]=vessel*np.ones((len(missingIDs)))
				sorted_idx = vesselTraj[:,0].argsort()
				vesselTraj = vesselTraj[sorted_idx,:]
				maskVessel = maskVessel[sorted_idx]
				vesselTraj[:,2:] = fillarr(vesselTraj[:,2:])
			vesselTraj = vesselTraj[:,2:]
			sequence[v,:] = torch.from_numpy(vesselTraj.astype('float32'))
			mask[v,:] = torch.from_numpy(maskVessel.astype('float32')).bool()
		vessel_count = torch.tensor(len(vessels))
		return sequence, mask, vessel_count
	def __getitem__(self, idx):
		idx = int(idx.numpy()) if not isinstance(idx,int) else idx
		sequence, mask, vessel_count = self.sequences[idx], self.masks[idx], self.vesselCount[idx]
		ip = sequence[:,:self.sequence_length,...]
		op = sequence[:,self.sequence_length:,...]
		distance_matrix, bearing_matrix, heading_matrix = get_features(ip,0)
		ip_mask = mask[:,:self.sequence_length]
		op_mask = mask[:,self.sequence_length:]
		ip = ip[...,:self.feature_size]
		op = op[...,:self.feature_size]
		return {'input': ip, \
			'output': op, \
			'distance_matrix': distance_matrix, \
			'bearing_matrix': bearing_matrix, \
			'heading_matrix': heading_matrix, \
			'input_mask': ip_mask, \
			'output_mask': op_mask, \
			'vessels': vessel_count}
	def __len__(self):
		return self.len

def fillarr(arr):
	for i in range(arr.shape[1]):
		idx = np.arange(arr.shape[0])
		idx[arr[:,i]==0] = 0
		np.maximum.accumulate(idx, axis = 0, out=idx)
		arr[:,i] = arr[idx,i]
		if (arr[:,i]==0).any():
			idx[arr[:,i]==0] = 0
			np.minimum.accumulate(idx[::-1], axis=0)[::-1]
			arr[:,i] = arr[idx,i]
	return arr	

def pad_sequence(sequences, f, _len, padding_value=0.0):
	dim_ = sequences[0].size(1)
	if 'matrix' in f:
		out_dims = (len(sequences),_len,dim_,_len)
	elif 'mask' in f:
		out_dims = (len(sequences),_len,dim_)
	else:
		out_dims = (len(sequences),_len,dim_,sequences[0].size(-1))
	out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
	for i, tensor in enumerate(sequences):
		length=tensor.size(0)
		if 'matrix' in f:
			out_tensor[i,:length,:,:length]=tensor
		else:
			out_tensor[i,:length,...]=tensor
	return out_tensor

class collate_function:
	def __call__(self, batch):
		batch_size=len(batch)
		features = list(batch[0].keys())
		_len = max([b['input'].size(0) for b in batch])
		output_batch = []
		for f in features:
			if 'vessels' in f:
				output_feature=torch.stack([torch.tensor(b[f]).float() for b in batch])
			else:
				output_feature = pad_sequence([b[f] for b in batch],f,_len)
			output_batch.append(output_feature)
		return tuple(output_batch)

def load_data(data_dir, args):
	dataset_dir = 'datasets/%02d/'%(args.zone)
	train_dir = dataset_dir+'train/'
	val_dir = dataset_dir+'val/'
	test_dir = dataset_dir+'test/'
	if not os.path.isdir(dataset_dir):
		os.makedirs(dataset_dir)
		os.makedirs(train_dir)
		os.makedirs(val_dir)
		os.makedirs(test_dir)
	if args.split_data or len(os.listdir(train_dir))==0:
		data = trajectory_dataset(data_dir, args.sequence_length, args.prediction_length, args.feature_size)
		data_size=len(data)
		valid_size = int(np.floor(0.1*data_size))
		test_size=valid_size
		train_size = data_size-valid_size-test_size
		traindataset, validdataset, testdataset = random_split(data, [train_size, valid_size, test_size])
		torch.save(traindataset, train_dir+"%02d_%02d.pt"%(args.sequence_length,args.prediction_length))
		torch.save(validdataset,val_dir+"%02d_%02d.pt"%(args.sequence_length,args.prediction_length))
		torch.save(testdataset,test_dir+"%02d_%02d.pt"%(args.sequence_length,args.prediction_length))	
	else:
		traindataset = torch.load(train_dir+"%02d_%02d.pt"%(args.sequence_length,args.prediction_length))
		validdataset = torch.load(val_dir+"%02d_%02d.pt"%(args.sequence_length,args.prediction_length))
		testdataset = torch.load(test_dir+"%02d_%02d.pt"%(args.sequence_length,args.prediction_length)) 
	return traindataset, validdataset, testdataset


