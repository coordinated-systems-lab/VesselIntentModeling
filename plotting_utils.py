from __future__ import print_function
import sys
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

# function returns basemap 
def initialize_plot_env(urcrnrlat, llcrnrlat, llcrnrlon, urcrnrlon):
	m = Basemap(projection="cyl",area_thresh=.1,urcrnrlat=urcrnrlat,llcrnrlat=llcrnrlat,llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon)
	parallels = [llcrnrlat,urcrnrlat]
	meridians=[llcrnrlon,urcrnrlon]
	m.drawparallels(parallels,labels=[False,True,False,False])
	m.drawmeridians(meridians,labels=[True,False,False,False])
	return m

# function to convert a torch tensor to numpy
def convert_tensor_to_numpy(tensor_):
	return np.array(tensor_.clone().detach().cpu().numpy())

# plot hardwired attentions for  a frame
def plot_hardwired_attention(sequence, mask, dist_matrix, rb_matrix, vessel_domain, plot_dir, v_id=None):
	if not os.path.isdir(plot_dir):
		os.makedirs(plot_dir)
	if not os.path.isdir(plot_dir+'hardwired_attention/'):
		os.makedirs(plot_dir+'hardwired_attention/')
	numVessels = sequence.size(1)
	orig_size = rb_matrix.size()
	wts = mask.unsqueeze(-1).expand(mask.size(0),mask.size(1), dist_matrix.size(1)).bmm(F.relu(vessel_domain.index_select(0, rb_matrix.contiguous().view(-1).long()).view(orig_size)-dist_matrix))
	wts/=(wts.max()+1e-9)
	x_axis = np.arange(sequence.size(-2))
	if not v_id:
		v_id = random.choice(range(numVessels))
	fig, ax = plt.subplots()
	dist_mat_v = dist_matrix[:,v_id,...]
	wts_v = wts[:,v_id,...]
	dist_mat_v, wts_v = dist_mat_v.squeeze(0), wts_v.squeeze(0)
	dist_mat_v, wts_v = convert_tensor_to_numpy(dist_mat_v), convert_tensor_to_numpy(wts_v)
	flag=0
	for v1 in range(numVessels):
		if not (v1==v_id):
			dist_mat_v1 = np.reshape(dist_mat_v[:,v1],(-1))
			wts_v1 = np.reshape(wts_v[:,v1],(-1))
			if (flag==0):
				ax.plot(dist_mat_v1, 'r-', 'distance from vessel of interest')
				ax.plot(wts_v1,'b--', 'hardwired weight')
				flag=1
			else:
				ax.plot(dist_mat_v1, 'r-')
				ax.plot(wts_v1,'b--')
	ax.legend()
	ax.title("hardwired attention weights")
	plt.savefig(plot_dir+'attention/hardwired_attention/'+str(v_id)+'.png')

# plot temporal attention for a frame
def plot_soft_attention(sequence, target, prediction, plot_dir, soft_attention):
	if not os.path.isdir(plot_dir):
		os.makedirs(plot_dir)
	if not os.path.isdir(plot_dir+'soft_attention/'):
		os.makedirs(plot_dir+'soft_attention/')
	sequence, target, prediction = convert_tensor_to_numpy(sequence), convert_tensor_to_numpy(target), convert_tensor_to_numpy(prediction)
	ground_truth = np.concatenate((sequence,target),axis=0)
	prediction=np.concatenate((sequence,prediction),axis=0)
	ground_truth = ground_truth[~(ground_truth[...,:]==0).any()]
	prediction = prediction[~(prediction[...,:]==0).any()]
	lat1, lon1 = np.reshape(ground_truth[...,0],(-1)), np.reshape(ground_truth[...,1],(-1))
	lat2, lon2 = np.reshape(prediction[...,0],(-1)), np.reshape(prediction[...,1],(-1))
	llcrnrlat, urcrnrlat = np.amin([np.amin(lat1),np.amin(lat2)]), np.amax([np.amax(lat1),np.amax(lat2)])
	llcrnrlon, urcrnrlon = np.amin([np.amin(lon1),np.amin(lon2)]), np.amax([np.amax(lon1),np.amax(lon2)])
	m = initialize_plot_env(urcrnrlat, llcrnrlat, llcrnrlon, urcrnrlon)
	soft_attention = convert_tensor_to_numpy(soft_attention)
	lat, lon = sequence[...,0], sequence[...,1]
	lat, lon = np.reshape(lat, (-1)), np.reshape(lon, (-1))
	x, y = m(lon, lat)
	len_ = np.shape(target)[-2]+1
	lat1, lon1 = lat1[-len_:], lon1[-len_:]
	lat2, lon2 = lat2[-len_:], lon2[-len_:]
	x1, y1 = m(lon1, lat1)
	x2, y2 = m(lon2, lat2)
	segments = [x,y]
	fig, ax = plt.subplots()
	ax.set_title("soft attention")
	norm = plt.Normalize(soft_attention.min(),soft_attention.max())
	lc = matplotlib.collections.LineCollection(segments, cmap='viridis', norm=norm)
	lc.set_array(soft_attention)
	lc.set_linewidth(2)
	line = ax.add_collection(lc)
	line2 = ax.plot(x1, y1, 'r-', 'ground truth')
	line3 = ax.plot(x2, y2, 'b--', 'prediction')
	plt.savefig(plot_dir+'soft_attention/soft_attn.png')

# plot trajectory of a vessel in frame
def plot_vessel(sequence, target, prediction, plot_dir, infer=False):
	if not os.path.isdir(plot_dir):
		os.makedirs(plot_dir)
	if not os.path.isdir(plot_dir+'predictions/'):
		os.makedirs(plot_dir+'predictions/')
	sequence, target, prediction = convert_tensor_to_numpy(sequence), convert_tensor_to_numpy(target), convert_tensor_to_numpy(prediction)
	print(sequence)
	ground_truth = np.concatenate((sequence,target),axis=0)
	prediction=np.concatenate((sequence,prediction),axis=0)
	ground_truth = ground_truth[~(ground_truth[...,:]==0).any()]
	prediction = prediction[~(prediction[...,:]==0).any()]
	lat1, lon1 = np.reshape(ground_truth[...,0],(-1)), np.reshape(ground_truth[...,1],(-1))
	lat2, lon2 = np.reshape(prediction[...,0],(-1)), np.reshape(prediction[...,1],(-1))
	llcrnrlat, urcrnrlat = np.amin([np.amin(lat1),np.amin(lat2)]), np.amax([np.amax(lat1),np.amax(lat2)])
	llcrnrlon, urcrnrlon = np.amin([np.amin(lon1),np.amin(lon2)]), np.amax([np.amax(lon1),np.amax(lon2)])
	print(urcrnrlat, llcrnrlat, llcrnrlon, urcrnrlon)
	m = initialize_plot_env(urcrnrlat, llcrnrlat, llcrnrlon-0.1, urcrnrlon)
	x1, y1 = m(lon1, lat1)
	x2, y2 = m(lon2, lat2)
	fig, ax = plt.subplots()
	ax.set_title("predicted vs. ground truth trajectory of a vessel")
	line1 = ax.plot(x1,y1,'ro','ground truth trajectory')
	line2 = ax.plot(x2,y2,'yo','predicted trajectory')
	ax.legend()
	if infer:
		num_pred = glob.glob(plot_dir+'predictions/*.png')
		plt.savefig(plot_dir+'predictions/predicted_trajectory_'+str(num_pred+1)+'.png')
	else:
		plt.savefig(plot_dir+'predictions/predicted_trajectory.png')

# plot trajectories for all vessels in a frame/batch 
def plot_batch(sequence, target, prediction, plot_dir, loss, model_type, infer=False):
	if not os.path.isdir(plot_dir):
		os.makedirs(plot_dir)
	if not os.path.isdir(plot_dir+'predictions/'):
		os.makedirs(plot_dir+'predictions/')
	numVessels = sequence.size(1)
	plot_dir = plot_dir + 'predictions/' + str(model_type) + '/' + str(numVessels) + '/' 
	if not os.path.isdir(plot_dir):
		os.makedirs(plot_dir)
	sequence, target, prediction = sequence.squeeze(0), target.squeeze(0),prediction.squeeze(0)
	sequence_length=sequence.size(1)
	prediction_length=target.size(1)
	num_v = sequence.size(0)
	sequence, target, prediction = convert_tensor_to_numpy(sequence), convert_tensor_to_numpy(target), convert_tensor_to_numpy(prediction)
	ground_truth = np.concatenate((sequence,target),axis=1)
	prediction=np.concatenate((sequence,prediction),axis=1)
	lat_values1 = np.reshape(ground_truth[...,0],(-1))
	lat_values2 = np.reshape(prediction[...,0],(-1))
	lat_values1 = np.append(lat_values1,lat_values2)
	llcrnrlat, urcrnrlat = np.amin(lat_values1[np.nonzero(lat_values1)]), np.amax(lat_values1[np.nonzero(lat_values1)])
	lon_values1 = np.reshape(ground_truth[...,1],(-1))
	lon_values2 = np.reshape(prediction[...,1],(-1))
	lon_values1=np.append(lon_values1,lon_values2)
	llcrnrlon, urcrnrlon = np.amin(lon_values1[np.nonzero(lon_values1)]), np.amax(lon_values1[np.nonzero(lon_values1)])
	fig, ax = plt.subplots()
	m = initialize_plot_env(urcrnrlat+0.001, llcrnrlat-0.01, llcrnrlon-0.05, urcrnrlon+0.001)
	ax.set_title("Vessel Intent Modeling\nSequence Length " + str(sequence_length) + " Prediction Length " + str(prediction_length) + "\nNumber of Vessels in Frame: " + str(num_v) + "\nADE loss: " + str(np.round(loss,5)))
	for v in range(numVessels):
		ground_truth_v = ground_truth[v,...]
		prediction_v = prediction[v,...]
		lat1, lon1 = np.reshape(ground_truth_v[...,0],(-1)), np.reshape(ground_truth_v[...,1],(-1))
		lat2, lon2 = np.reshape(prediction_v[...,0],(-1)), np.reshape(prediction_v[...,1],(-1))
		x1, y1 = m(lon1, lat1)
		x2, y2 = m(lon2, lat2)
		x1= x1[~(x1==0)]
		y1 = y1[~(y1==0)]
		x2 = x2[~(x2==0)]
		y2 = y2[~(y2==0)]
		m.plot(x1, y1, 'b-',linewidth=1)
		m.plot(x2, y2,"r--",linewidth=1)
		if abs(x1[-1]-x1[0])>0:
			mean_idx=int(len(x1)/2)
			start_x = x1[mean_idx]
			start_y = y1[mean_idx]
			dy = y1[mean_idx+1]-y1[mean_idx]
			dx = x1[mean_idx+1]-x1[mean_idx]
			ax.arrow(start_x,start_y,dx,dy,width=0,head_width=0.0005,color="blue",fill='false')
	red_patch = mpatches.Patch(color='blue', label='ground truth')
	blue_patch = mpatches.Patch(color='red', label='predicted trajectory')
	ax.legend(handles=[red_patch, blue_patch])
	num_pred = len(glob.glob(plot_dir+'*.png'))
	plt.savefig(plot_dir+str(num_pred+1)+'.png')

# function to smoothen obtained domain
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# function to plot model-learned domain
def plot_domain(r,delta_rb,sequence_length,prediction_length,delta_cog,max_dist,plot_dir,model,domain_param,save_new=False,approx=False):
	if not os.path.isdir(plot_dir):
		os.makedirs(plot_dir)
	if not os.path.isdir(plot_dir+'domain_plots/'):
		os.makedirs(plot_dir+'domain_plots/')
	r=convert_tensor_to_numpy(r)
	r=np.around(r,4)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='polar')
	ax.set_theta_zero_location("N")
	if approx:
		for angle1 in range(np.shape(r)[0]):
			ang1 = delta_cog*angle1
			r12 = r[angle1,:]
			domain=r12
			theta_ticks = [math.radians(delta_rb*i) for i in np.arange(len(r12))]
			domain=np.append(domain,domain[0])
			theta=[math.radians(i) for i in np.arange(0,360,delta_rb)]
			theta=np.append(theta,math.radians(360))
			ax.plot(theta,domain,linewidth=0.5,label=str(ang1)+'$^{o}$')
		avg_domain=np.max(r,axis=0)
		avg_domain=np.append(avg_domain,avg_domain[0])
		ax.plot(theta,avg_domain,'black',linewidth=1,label="max")
		ax.legend(loc='upper left',bbox_to_anchor=(1.05,1.05),ncol=1)
		ax.set_rmax(max_dist)
		ax.set_rticks(np.arange(0,max_dist,0.5))
		ax.set_xticks(theta_ticks)
	else:
		for angle1 in range(np.shape(r)[0]):
			# cog
			ang1 = delta_cog*angle1
			r12 = r[angle1,:]
			domain = np.zeros(360)
			deg = 0
			for j in range(len(r12)):
				domain[int(deg):int(deg+delta_rb)]=r12[j]
				deg+=delta_rb	
			theta_ticks = [math.radians(delta_rb*i) for i in np.arange(len(r12))]
			theta=[math.radians(i) for i in np.arange(0,360,1)]
			#theta=np.append(theta,math.radians(360))
			ax.plot(theta,domain,linewidth=0.5,label=str(ang1)+'$^{o}$')
		ax.legend(loc='upper left',bbox_to_anchor=(1.05,1.05),ncol=1)
		ax.set_rmax(max_dist)
		ax.set_rticks(np.arange(0,max_dist,0.5))
		ax.set_xticks(theta_ticks)
	ax.set_title("Vessel Domain in nautical miles\nSequence Length: "+str(sequence_length)+" Prediction Length: "+str(prediction_length    ))
			
	if not save_new:
		plt.savefig(str(plot_dir)+'domain_plots/'+str(model)+'/'+str(sequence_length)+'_'+str(prediction_length)+'_'+str(delta_rb)+'_'+str(delta_cog)+'_'+str(domain_param)+'.png')
	else:
		num=len(glob.glob(str(plot_dir)+'domain_plots/'+str(model)+'/'+str(sequence_length)+'_'+str(prediction_length)+'_'+str(delta_rb)+'_'+str(delta_cog)+'_'+str(domain_param)+'*.png'))
		plt.savefig(str(plot_dir)+'domain_plots/'+str(model)+'/'+str(sequence_length)+'_'+str(prediction_length)+'_'+str(delta_rb)+'_'+str(delta_cog)+'_'+str(domain_param)+'_'+str(num)+'.png')
	plt.close()


# function to plot train, valid errors during training 
class Plotter(object):
	def __init__(self,plotfile,train=True,valid=True):
		if train:
			self.train = []
		if valid:
			self.valid=[]
		self.plotfile = plotfile
	def update(self,train_loss=None,valid_loss=None):
		if train_loss:
			self.train.append(train_loss)
		if valid_loss:
			self.valid.append(valid_loss)
		fig = plt.figure()
		if train_loss:
			plt.plot(self.train, 'r-', label='training ADE')
		if valid_loss:
			plt.plot(self.valid, 'b-', label="validation ADE")
		plt.legend()
		plt.ylabel("ADE")
		plt.xlabel("Epochs")
		plt.savefig(self.plotfile)

