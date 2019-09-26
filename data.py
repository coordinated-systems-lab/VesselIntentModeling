import sys
sys.dont_write_bytecode=True
import torch
import pickle
from torch.utils.data import Dataset,DataLoader
import pandas as pd 
import numpy as np 
from utils import *
from multiprocessing import Process
import time

device=torch.device("cuda" if torch.cuda.is_available else "cpu")

seed_everything()

class dataset(Dataset):
	def __init__(self,filenames,args):
		super(dataset,self).__init__()
		self.files = filenames
		self.len = 0
		self.maxval={}
		self.minval={}
		self.file_idx={}
		self.timestamps={}
		self.sequences={}
		self.masks={}
		self.vesselCount={}
		self.sequence_length=args.sequence_length
		self.prediction_length=args.prediction_length
		self.maxVessels=args.maxVessels
		self.delta_rb = args.delta_rb
		self.combinations={}
		self.delta_cog = args.delta_cog
		self.feature_size=args.feature_size
		self.output_size=args.output_size
		self.idx=0
		#self.shift=1
		#self.shift=int(self.sequence_length/2)
		self.shift=int(self.sequence_length+self.prediction_length)
		for f,filename in enumerate(filenames):
			df=self.load_data(filename)
			maxval,minval=self.normalizing_parameters(df)
			self.maxval[f],self.minval[f]=torch.from_numpy(maxval),torch.from_numpy(minval)
			self.maxval[f].requires_grad_(False)
			self.minval[f].requires_grad_(False)
			self.timestamps[f] = np.unique(df['BaseDateTime'].values)
			self.get_sequences(df,f)
		self.len=self.idx
		sys.stdout.write("\nTotal number of samples: %d" %(self.len))
		max_vcount = max([v for k,v in self.vesselCount.items()])
		print("\nmax vessels per sample: %d" %(max_vcount))
	def __len__(self):
		return self.len
	def load_data(self,filename):
		data=pd.read_csv(filename,header=0,parse_dates=['BaseDateTime'],usecols=['BaseDateTime','MMSI','LAT','LON','SOG','Heading'])
		data.sort_values(['BaseDateTime'],inplace=True)
		data=data[['BaseDateTime','MMSI','LAT','LON','SOG','Heading']]
		return data
	def normalizing_parameters(self,df):
		return np.amax(df.values[:,2:2+self.feature_size],axis=0).astype('float32'),np.amin(df.values[:,2:2+self.feature_size],axis=0).astype('float32')
	def get_sequences(self,df,f):
		j=0
		timestamps=self.timestamps[f]
		while not (j+self.sequence_length+self.prediction_length)>len(timestamps):
			frameTimestamps=timestamps[j:j+self.sequence_length+self.prediction_length]
			if not ((np.amax(np.diff(frameTimestamps).astype('float'))/(6e+10))>1 or (np.amax(np.diff(frameTimestamps).astype('float'))/(8.64e+13))>=1):
				frame=df.loc[df['BaseDateTime'].isin(frameTimestamps)]
				frameVessels = [v for v in frame['MMSI'].unique() if len(frame.loc[frame['MMSI']==v]['BaseDateTime'].unique())>=2]
				if ((frame['Heading']>=0).all() and (frame['Heading']<=360).all()) and len(frameVessels)>1 and len(frameVessels)<=self.maxVessels:
					sys.stdout.write("\rprocessing file: %d/%d            sample: %d\t" %(f+1,len(self.files),self.idx))
					if not self.get_sequence(frame,f)[0] is None:
						self.sequences[self.idx],self.masks[self.idx],self.vesselCount[self.idx]=self.get_sequence(frame,f)
						self.file_idx[self.idx]=f 
						self.idx+=1
			j+=self.shift
	def get_sequence(self,frame,f):
		frame=frame.values
		frameIDs=np.unique(frame[:,0]).tolist()
		input_frame = frame[np.isin(frame[:,0],frameIDs[:self.sequence_length])]
		frameVessels = [v for v in np.unique(input_frame[:,1]).tolist() if len(frame[frame[:,1]==v])>=2]
		sequence=torch.FloatTensor(len(frameVessels),len(frameIDs),self.feature_size)
		mask=torch.FloatTensor(len(frameVessels),len(frameIDs))
		for v, vessel in enumerate(frameVessels):
			vesselTraj = frame[frame[:,1]==vessel]
			vesselTrajlen=np.shape(vesselTraj)[0]
			vesselIDs=np.unique(vesselTraj[:,0])
			maskVessel=np.ones(len(frameIDs))
			if vesselTrajlen<(self.sequence_length+self.prediction_length):
				missingIDs=[f for f in frameIDs if not f in vesselIDs]
				maskVessel[vesselTrajlen:].fill(0.0)
				paddedTraj=np.zeros((len(missingIDs),np.shape(vesselTraj)[1]))
				vesselTraj=np.concatenate((vesselTraj,paddedTraj),axis=0)
				vesselTraj[vesselTrajlen:,0]=missingIDs
				vesselTraj[vesselTrajlen:,1]=vessel*np.ones((len(missingIDs)))
				sorted_idx = vesselTraj[:,0].argsort()
				vesselTraj=vesselTraj[sorted_idx,:]
				maskVessel=maskVessel[sorted_idx]
				vesselTraj[:,2:] = interpolate(vesselTraj[:,2:])
				if maskVessel[0]==0 or maskVessel[-1]==0:
					mask_init = np.where(maskVessel!=0)[0][0]
					mask_end = np.where(maskVessel!=0)[0][-1]
					maskVessel[mask_init:mask_end].fill(1)
				else:
					maskVessel=np.ones(len(frameIDs))
				vesselTraj=vesselTraj[:,2:]
				if not (np.amax(np.diff(vesselIDs)).seconds==60):
					vesselTraj = interpolate(vesselTraj)
			else:
				vesselTraj=vesselTraj[:,2:]
			sequence[v,:]=torch.from_numpy(vesselTraj[:,:self.feature_size].astype('float32'))
			mask[v,:]=torch.from_numpy(maskVessel.astype('float32'))
		dist_matrix,rb_matrix,cog_matrix=get_feature_matrices(sequence[:,:self.sequence_length],self.delta_rb,self.delta_cog,0)
		self.update_combinations_dict(rb_matrix,cog_matrix)
		invalid = 0
		for key, value in self.combinations.items():
			if value > 5000 and not key=='tensor([[0., 0.]])' and invalid==0:
				invalid=1
				break
		if invalid==1:
			self.remove_combination(rb_matrix,cog_matrix)
			return None,None,None
		else:
			minval = self.minval[f]
			maxval = self.maxval[f]
			for ftr in range(sequence.size(-1)):
				sequence[...,ftr].clamp_(min=minval[...,ftr],max=maxval[...,ftr])
			for key in sorted(self.combinations.keys()):
				print(key, ":: ", self.combinations[key]) 
			#print(self.combinations)
			print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
			return sequence,mask,len(frameVessels)
	def normalize_data(self,seq,maxval,minval):
		eps=1e-12
		seq=seq-minval
		seq=seq+eps
		seq=seq/(maxval-minval+eps)
		return seq
	def update_combinations_dict(self,rb_matrix,cog_matrix):
		combinations = torch.stack((rb_matrix.contiguous().view(-1),cog_matrix.contiguous().view(-1)))
		combinations=torch.unique(combinations,dim=1).transpose(0,1).view(-1,2)
		for row in combinations.chunk(combinations.size(0),dim=0):
			if not str(row) in list(self.combinations.keys()):
				self.combinations[str(row)]=1
			else:
				self.combinations[str(row)]+=1
	def remove_combination(self,rb_matrix,cog_matrix):
		combinations = torch.stack((rb_matrix.contiguous().view(-1),cog_matrix.contiguous().view(-1)))
		combinations=torch.unique(combinations,dim=1).transpose(0,1).view(-1,2)
		for row in combinations.chunk(combinations.size(0),dim=0):
			self.combinations[str(row)]-=1
	def __getitem__(self,idx):
		if not isinstance(idx,int):
			idx=int(idx.numpy())
		fnum = self.file_idx[idx]
		maxval, minval = self.maxval[fnum], self.minval[fnum]
		sequence,mask,frameVessels=self.sequences[idx],self.masks[idx],self.vesselCount[idx]
		ip=sequence[:,:self.sequence_length,...]
		op=sequence[:,self.sequence_length:,...]
		dist_matrix,rb_matrix,cog_matrix=get_feature_matrices(ip,self.delta_rb,self.delta_cog,0)
		ip,op=self.normalize_data(ip,maxval,minval),self.normalize_data(op,maxval,minval)
		numVessels=sequence.size(0)
		if (numVessels<self.maxVessels):
			ip, op, dist_matrix, rb_matrix, cog_matrix, mask = self.pad_data(ip, op, dist_matrix, rb_matrix, cog_matrix, mask, (self.maxVessels-numVessels))
		elif (numVessels>self.maxVessels):
			ip=ip[:self.maxVessels,...,:self.feature_size]
			op = op[:self.maxVessels,...,:self.output_size]
			dist_matrix=dist_matrix[:self.maxVessels,...,:self.maxVessels]
			rb_matrix=rb_matrix[:self.maxVessels,...,:self.maxVessels]
			cog_matrix=cog_matrix[:self.maxVessels,...,:self.maxVessels]
			mask=mask[:self.maxVessels,...]
		ip_mask = mask[:,:self.sequence_length]
		op_mask=mask[:,self.sequence_length:]
		return ip[...,:self.feature_size], op[...,:self.output_size], dist_matrix, rb_matrix, cog_matrix, ip_mask, op_mask, frameVessels, maxval, minval
	def pad_data(self,ip,op,dist_matrix,rb_matrix,cog_matrix,mask,padnum):
		paddedIP=torch.zeros(padnum,ip.size(1),ip.size(2))
		paddedOP=torch.zeros(padnum,op.size(1),op.size(2))
		paddedDist1=torch.zeros(padnum,ip.size(1),dist_matrix.size(2))
		paddedDist2=torch.zeros(self.maxVessels,ip.size(1),padnum)
		ip=torch.cat((ip,paddedIP),dim=0)
		op=torch.cat((op,paddedOP),dim=0)
		dist_matrix=torch.cat((dist_matrix,paddedDist1),dim=0)
		dist_matrix=torch.cat((dist_matrix,paddedDist2),dim=2)
		rb_matrix=torch.cat((rb_matrix,paddedDist1),dim=0)
		rb_matrix=torch.cat((rb_matrix,paddedDist2),dim=2)
		cog_matrix=torch.cat((cog_matrix,paddedDist1),dim=0)
		cog_matrix=torch.cat((cog_matrix,paddedDist2),dim=2)
		mask=torch.cat((mask,torch.zeros(padnum,mask.size(1))),dim=0)
		return ip, op[...,:self.output_size], dist_matrix, rb_matrix, cog_matrix, mask

def interpolate(arr):
	cols = arr.shape[1]
	arr = pd.DataFrame(arr,index=None)
	for c in range(cols):
		arr.iloc[:,c].replace(to_replace=0,inplace=True,method='ffill')
		arr.iloc[:,c].replace(to_replace=0,inplace=True,method='bfill')
	return arr.values
