from __future__ import print_function
import sys
import os
sys.dont_write_bytecode=True
import warnings
warnings.filterwarnings("ignore")
from data import *
from models.model1 import *
from models.model2 import *
from models.vlstm import *
from models.temporal_vlstm import *
from models.spatial_vlstm import *
from utils import *
from plotting_utils import *
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('agg')
import argparse
import random
import metrics
import glob
import matplotlib.pyplot as plt
from termcolor import colored
from torch.utils.data import DataLoader,random_split
from torch.utils.data.sampler import SubsetRandomSampler

# use cuda if available else use cpu
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set seeds for reproducibility
seed_everything()

parser=argparse.ArgumentParser(description="vessel intent modeling")

# train/test/validation modes
parser.add_argument('--train',action="store_true",help="training phase included")
parser.add_argument('--val',action="store_true",help="validation phase included")
parser.add_argument('--test',action="store_true",help="testing phase included")

# dataset parameters
parser.add_argument('--split_data',action="store_true",help="split data into train, valid, test")
parser.add_argument('--validation_split',type=float,default=0.2,help="validation split")
parser.add_argument('--test_split',type=float,default=0.2,help="test split")
parser.add_argument('--data_directory',type=str,default='data/',help="directory containing AIS data")
parser.add_argument('--save_data',action="store_true",help="save data if processed again")
parser.add_argument('--dataset_directory',type=str,default="dataset/",help="directory to save dataset")
parser.add_argument('--feature_size',type=int,default=4,help="feature size")
parser.add_argument('--output_size',type=int,default=2,help="output size")

# model parameters
parser.add_argument('--maxVessels',type=int,default=15,help="maximum number of vessels per sample")
parser.add_argument('--hidden_size',type=int,default=12,help="hidden LSTM dimension")
parser.add_argument('--sequence_length',type=int,default=10,help="sequence length")
parser.add_argument('--prediction_length',type=int,default=1,help="prediction length")
parser.add_argument('--delta_rb',type=float,default=15,help="relative bearing discretization")
parser.add_argument('--delta_cog',type=float,default=30,help="heading discretization")
parser.add_argument('--min_dist',type=float,default=0.001,help="minimum domain distance")
parser.add_argument('--domain_init',type=str,default='custom',choices=['constant','custom'],help="domain param initialization")
parser.add_argument('--param_domain',type=float,help="domain initialization parameter")

# training parameters
parser.add_argument('--epochs',type=int,default=1000,help="training epochs")
parser.add_argument('--batch_size',type=int,default=12,help="batch size")
parser.add_argument('--learning_rate',type=float,default=0.0005,help="learning rate")
parser.add_argument('--eval_batch_size',type=int,default=128,help="batch size for evaluation")
parser.add_argument('--optimizer',type=str,default='Adam',help="optimizer")
parser.add_argument('--criterion',type=str,choices=['MSE','ADE','rootADE','FDE','smoothADE'],default='ADE',help="loss criterion")
parser.add_argument('--num_workers',type=int,default=0,help="number of dataloading workers")
parser.add_argument('--pin_memory',action="store_true",default=False,help="pin memory")
parser.add_argument('--net_dir',type=str,default='trained_models/',help="directory to save models")
parser.add_argument('--gpuid',type=int,default=1,choices=[0,1,2],help="gpu id")
parser.add_argument('--use_saved',action="store_true",help="use best saved model")
parser.add_argument('--best_loss',type=float,default=1000,help="best loss value")
parser.add_argument('--scheduler',action="store_true",default=False,help="use learning rate scheduler")
parser.add_argument('--model_type',choices=['vlstm','temporal_vlstm','spatial_vlstm','sta_lstm1','sta_lstm2'],type=str,help="model to use")

# plotting parameters
parser.add_argument('--plot_trajectory_batch',action="store_true",help="plot predicted trajectory for all vessels in frame for test data")
parser.add_argument('--plot_trajectory_vessel',action="store_true",help="plot predicted trajectory for a single vessel per frame")
parser.add_argument('--plot_hardwired_attention',action="store_true",help="plot hardwired attention for test data")
parser.add_argument('--plot_soft_attention',action="store_true",help="plot soft attention for test data")
parser.add_argument('--plot_domain',action="store_true",help="plot domain on every saved model")
parser.add_argument('--plot_dir',type=str,default='plots/',help="directory for saving plots")
parser.add_argument('--log_loss',action="store_true",help="log loss values")

# performance optimization parameters
parser.add_argument('--fix_num_vessels',action="store_true",default=False,help="number of vessels per batch constant")
parser.add_argument('--transfer_weights',action="store_true",default=False,help="transfer weights from other models")
parser.add_argument('--transfer_model',type=str,default='vlstm',help="model to transfer weights from")
parser.add_argument('--transfer_dir',type=str,default='models/',help="directory where transfer model is located")
parser.add_argument('--freeze_transferred_params',action="store_true",help="do not learn transferred parameters any further")
args=parser.parse_args()

if torch.cuda.is_available() and args.gpuid:
	torch.cuda.set_device(args.gpuid)

# make directory to save data 
if not os.path.isdir(args.net_dir):
	os.makedirs(args.net_dir)

if not os.path.isdir(args.net_dir+args.model_type):
	os.makedirs(args.net_dir+args.model_type)

# file to save trained models
if args.model_type=='sta_lstm' or args.model_type=='spatial_vlstm':
	netfile = args.net_dir+args.model_type+'/hsz_'+str(args.hidden_size)+'sequence_length_'+str(args.sequence_length)+'prediction_length_'+str(args.prediction_length)+'_'+str(args.delta_rb)+'_'+str(args.delta_cog)+'_'+str(args.param_domain)+'_'+str(args.maxVessels)+'.pt'
else:
	netfile = args.net_dir+args.model_type+'/hsz_'+str(args.hidden_size)+'sequence_length_'+str(args.sequence_length)+'prediction_length_'+str(args.prediction_length)+'_'+str(args.maxVessels)+'.pt'

# print train/test attributes 
train_attrs = ['train','val','test','hidden_size','sequence_length','prediction_length','delta_rb','delta_cog','param_domain','epochs','batch_size','eval_batch_size','learning_rate','optimizer','criterion','net_dir','use_saved','scheduler','model_type','fix_num_vessels']
test_attrs=['test','eval_batch_size','model_type','fix_num_vessels','net_dir','hidden_size','sequence_length','prediction_length','criterion']

attributes = vars(args)
for item in attributes:
	if args.train:
		if str(item) in train_attrs:
			print(colored("%s : %s" %(item,attributes[item]),"blue"))
	else:
		if str(item) in test_attrs:
			print(colored("%s : %s" %(item,attributes[item]),"blue"))

# initialize model
if args.model_type=='sta_lstm1':
	net=model1(args).float().to(device)
elif args.model_type=='sta_lstm2':
	net=model2(args).float().to(device)
elif args.model_type=='vlstm':
	net=vlstm(args).to(device)
elif args.model_type=='temporal_vlstm':
	net = temporal_vlstm(args).to(device)
elif args.model_type=='spatial_vlstm':
	net=spatial_vlstm(args).to(device)

# retrieve trained model parameters if testing
if args.use_saved:
	net.load_state_dict(torch.load(netfile,map_location=device))

# loss criterion for training, testing
criterion = getattr(metrics,str(args.criterion))().to(device)

# performance optimization -- transfer learning 
if(args.train and args.transfer_weights and not args.model_type=='vlstm'):
	print("transferring learned parameters from ", args.transfer_model)
	if args.transfer_model=='vlstm':
		transfer_model = vlstm(args).float().to(device)
	elif args.transfer_model=='spatial_vlstm':
		transfer_model=spatial_vlstm(args).float().to(device)
	elif args.transfer_model=='temporal_vlstm':
		transfer_model=temporal_vlstm(args).float().to(device)
	transfer_weights(transfer_model,net)
	transfer_file = args.transfer_dir + args.transfer_model + '/hsz_'+str(args.hidden_size)+'sequence_length_'+str(args.sequence_length)+'prediction_length_'+str(args.prediction_length)+'.pt'
	transfer_model.load_state_dict(torch.load(transfer_file))

# baseline loss
best_loss=args.best_loss
print(colored("BEST LOSS VALUE: %.3f"%(best_loss),"red"))

# function to smoothen domain shape
def smooth(domain_data, box_pts):
	y_smooth = np.zeros(domain_data.shape)
	for cog in range(domain_data.shape[0]):
		y = domain_data[cog,:]
		box = np.ones(box_pts)/box_pts
		y_smooth[cog,:] = np.convolve(y,box,mode='same')
	return y_smooth
	
# function to predict per batch 
def predict(batch,return_context=False):
	sequence,target,dist_matrix,rb_matrix,cog_matrix,ip_mask, op_mask, frameVessels, maxval, minval = batch
	maxval, minval = maxval.to(device), minval.to(device)
	if args.fix_num_vessels:
		maxVessels=args.maxVessels
	else:
		maxVessels=min(frameVessels.max(), args.maxVessels)
	sequence,target,dist_matrix,rb_matrix,cog_matrix,ip_mask,op_mask=sequence[:,:maxVessels,...],target[:,:maxVessels,...],dist_matrix[:,:maxVessels,...,:maxVessels],rb_matrix[:,:maxVessels,...,:maxVessels],cog_matrix[:,:maxVessels,...,:maxVessels],ip_mask[:,:maxVessels,...],op_mask[:,:maxVessels,...]
	sequence=sequence[...,:args.feature_size]
	target=target[...,:args.output_size]
	maxval, minval = maxval[...,:args.feature_size], minval[...,:args.feature_size]
	sequence, target, dist_matrix, rb_matrix, ip_mask, op_mask = sequence.to(device),target.to(device),dist_matrix.to(device),rb_matrix.to(device),ip_mask.to(device),op_mask.to(device)
	cog_matrix=cog_matrix.to(device)
	if not (len(sequence.size())==4):
		sequence,target,dist_matrix,rb_matrix,cog_matrix,ip_mask, op_mask, maxval, minval=sequence.unsqueeze(0),target.unsqueeze(0), dist_matrix.unsqueeze(0),rb_matrix.unsqueeze(0),cog_matrix.unsqueeze(0),ip_mask.unsqueeze(0),op_mask.unsqueeze(0),maxval.unsqueeze(0),minval.unsqueeze(0)
	if (return_context):
		pred,context_vector_dict = net(sequence, dist_matrix, rb_matrix, cog_matrix,ip_mask,maxval.clone(),minval.clone(),maxVessels, return_context=True)
		pred = pred.float().to(device)
	else:
		pred = net(sequence, dist_matrix, rb_matrix,cog_matrix,ip_mask,maxval,minval,maxVessels).float().to(device)
	pred_, target_ = unnormalize(pred,maxval,minval), unnormalize(target, maxval, minval)
	target_mask = op_mask.unsqueeze(-1).expand(target.size())
	pred_mask=op_mask.unsqueeze(-1).expand(pred.size())
	pred_[pred_mask==0] = 0
	target_[target_mask==0] = 0
	loss=criterion(pred_,target_).float().to(device)
	assert(not (torch.isnan(loss).any() or torch.isinf(loss).any())),"nan/inf loss encountered"
	sequence=unnormalize(sequence,maxval,minval)
	if not(return_context):
		return loss, pred_, target_, sequence[...,:2]
	else:
		return loss, pred_, target_, sequence[...,:2], context_vector_dict

# function to test trained model
def test(testloader=None):
	net.load_state_dict(torch.load(netfile))
	net.eval()
	if (testloader==None):
		print('-- initializing test dataset ---')
		test_data=torch.load(str(args.dataset_directory)+'test/'+str(args.sequence_length)+'_'+str(args.prediction_length)+'.pt')
		test_loader=DataLoader(test_data,batch_size=args.eval_batch_size,shuffle=False)
	else:
		test_loader=testloader
	test_loss=0
	with torch.no_grad():
		for b, batch in enumerate(test_loader):
			if not (args.plot_trajectory_batch or args.plot_trajectory_vessel or args.plot_hardwired_attention or args.plot_soft_attention):
				loss = predict(batch)[0]
			else:
				loss, prediction, target, sequence = predict(batch,return_context=False)
				v_id = random.choice(np.arange(sequence.size(1)))
				if args.plot_trajectory_batch:
					print("==> plotting batch trajectory")
					plot_batch(sequence, target, prediction, args.plot_dir,loss.item(),args.model_type)	
				if args.plot_trajectory_vessel:
					print("==> plotting trajectory for vessel")
					sequence_vessel = sequence[:,v_id,...]
					target_vessel = target[:, v_id, ...]
					prediction_vessel = prediction[:, v_id, ...]
					plot_vessel(sequence_vessel, target_vessel, prediction_vessel, args.plot_dir)		
				if args.plot_hardwired_attention:
					print("==> plotting hardwired attention weights")
					mask , dist_matrix, rb_matrix = batch[5], batch[2], batch[3]
					plot_hardwired_attention(sequence, mask, dist_matrix, rb_matrix, net.hardwiredAttention.domain, args.plot_dir, v_id=v_id)	
				if args.plot_soft_attention:
					print("==> plotting soft attention weights")
			sys.stdout.write("\rtest batch: %d           test loss=%.5f" %(b+1,loss.item()))
			test_loss+=loss.item()
			if (b==3000):
				break
	test_loss/=(b+1)
	sys.stdout.write(colored("\rMEAN TEST LOSS : %.5f    \n" %(test_loss),"yellow"))

# function to load data 
def load_data(args):
	trainfile = str(args.dataset_directory)+'train/'+str(args.sequence_length)+'_'+str(args.prediction_length)+'_'+str(args.delta_rb)+'_'+str(args.delta_cog)+str(args.maxVessels)+'.pt'
	validfile=str(args.dataset_directory)+'valid/'+str(args.sequence_length)+'_'+str(args.prediction_length)+'_'+str(args.delta_rb)+'_'+str(args.delta_cog)+str(args.maxVessels)+'.pt'
	testfile=str(args.dataset_directory)+'test/'+str(args.sequence_length)+'_'+str(args.prediction_length)+'_'+str(args.delta_rb)+'_'+str(args.delta_cog)+str(args.maxVessels)+'.pt'
	if os.path.exists(trainfile) and os.path.exists(validfile) and os.path.exists(testfile) and not args.split_data:
		print("==> loading dataset from ",str(args.dataset_directory))
		traindataset, validdataset, testdataset = torch.load(trainfile), torch.load(validfile), torch.load(testfile)
	else:
		datafiles = glob.glob(args.data_directory+'*.csv')
		data = dataset(datafiles,args)
		dataset_size=len(data)
		valid_size = int(args.validation_split*dataset_size)
		test_size=int(args.test_split*dataset_size)
		train_size=dataset_size-valid_size-test_size
		traindataset,validdataset,testdataset = random_split(data,[train_size,valid_size,test_size])
		if args.save_data:
			print("==> saving data")
			torch.save(traindataset, trainfile)
			torch.save(validdataset, validfile)
			torch.save(testdataset,testfile)
	return traindataset, validdataset, testdataset

# training function 
def main():
	global best_loss
	if hasattr(net,'hardwiredAttention') and args.train and not args.use_saved:
		if args.domain_init=='constant':
			nn.init.constant_(net.hardwiredAttention.domain,args.param_domain)
		elif args.domain_init=='custom':
			domain_initialization(net.hardwiredAttention.domain,args.delta_rb,args.delta_cog,args.param_domain)		
	if ((not args.transfer_weights) or (args.transfer_weights and not args.freeze_transferred_params)):
		optimizer=getattr(torch.optim,args.optimizer)(net.parameters(),lr=args.learning_rate)
	elif (args.transfer_weights and args.freeze_transferred_params):
		params_tf = dict(transfer_model.named_parameters())
		params = dict(net.named_parameters())
		params = [params[p] for p in params if not p in params_tf.keys()]
		optimizer = getattr(torch.optim,args.optimizer)(params,lr=args.learning_rate)
		print("optimizing only parameters NOT transferred")
	if args.scheduler:
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=0.0005) 
	traindataset, validdataset, testdataset = load_data(args)
	print("==== data loaded ===== ")
	if not torch.cuda.is_available():
		exit()
	trainloader = DataLoader(traindataset,batch_size=args.batch_size,num_workers=args.num_workers,pin_memory=args.pin_memory,shuffle=True)
	validloader=DataLoader(validdataset,batch_size=args.eval_batch_size,drop_last=False,num_workers=args.num_workers,pin_memory=args.pin_memory,shuffle=False)
	testloader=DataLoader(testdataset,batch_size=args.eval_batch_size,drop_last=False,num_workers=args.num_workers,pin_memory=args.pin_memory,shuffle=False) 
	print("Training samples: %d\nValidation Samples: %d\nTesting Samples: %d\nMaximum Vessels per Frame: %d\n" %(len(traindataset),len(validdataset),len(testdataset),args.maxVessels))
	plotfile = 'plots/train_plots/'+str(args.model_type)+'/'+str(args.sequence_length)+'_'+str(args.prediction_length)+'_'+str(args.hidden_size)+'_'+str(args.param_domain)+'_'+str(args.maxVessels)+'.png'
	if args.train:
		print("Number of trainable parameters in model: %d" %(count_parameters(net)))
		numBatches=len(trainloader)
		if args.val:
			numValBatches=len(validloader)
			plotter = Plotter(plotfile)
		else:
			plotter = Plotter(plotfile,valid=False)
		print("training...")
		for epoch in range(args.epochs):
			net.train()
			sys.stdout.write("TRAINING EPOCH %d\n"%(epoch+1))
			epoch_loss=0
			for b, batch in enumerate(trainloader):
				optimizer.zero_grad()
				loss=predict(batch)[0]
				sys.stdout.write("\r epoch %d  batch %d/%d  loss %.5f      " %(epoch+1,b+1,numBatches,loss.item()))
				epoch_loss+=loss.item()
				loss.backward()
				if hasattr(net,'hardwiredAttention'):
					nn.utils.clip_grad_norm_(net.hardwiredAttention.domain,0.5)
				optimizer.step()
				if hasattr(net,'hardwiredAttention'):
				#	net.hardwiredAttention.domain.data[:,0].copy_(net.hardwiredAttention.domain.data[:,-1])
					net.hardwiredAttention.domain.data.clamp_(min=0.01,max=5)
		#	exit()
			epoch_loss/=(b+1)
			sys.stdout.write('\r' + colored('MEAN TRAINING LOSS FOR EPOCH: %.5f             ' %(epoch_loss),"red"))
			if hasattr(net,'hardwiredAttention'):
				print(colored('\nlearned safety distance-->',"red"))
				print(net.hardwiredAttention.domain.data)
			if not args.val:
				plotter.update(epoch_loss)
				if(epoch_loss<best_loss):
					best_loss=epoch_loss
					print("saving model...")
					torch.save(net.state_dict(),netfile)
			else:
				net.eval()
				valid_loss=0
				print("validation...")
				for b, batch in enumerate(validloader):
					loss = predict(batch)[0]
					valid_loss+=loss.item()
					sys.stdout.write("\r batch: %d/%d  loss: %.5f " %(b+1,numValBatches,loss.item()))
					del loss
				valid_loss/=(b+1)
				plotter.update(epoch_loss,valid_loss)
				if args.scheduler:
						scheduler.step(valid_loss)
				sys.stdout.write('\r' + colored('MEAN VALIDATION LOSS FOR EPOCH: %.5f \t \n' %(valid_loss),"red"))
				if(valid_loss<best_loss):
					best_loss=valid_loss
					print("saving model...")
				torch.save(net.state_dict(),netfile)
					
	elif args.test:
		sys.stdout.write(colored("testing","blue"))
		test(testloader)

if __name__ == '__main__':
	main()


