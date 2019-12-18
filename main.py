from __future__ import print_function
import sys
sys.dont_write_bytecode=True
import warnings
warnings.filterwarnings("ignore")

import torch
import argparse
import random
import glob
import numpy as np
from termcolor import colored
import models
from data import *
from utils import *
from geographic_utils import *
from train_utils import *

np.set_printoptions(precision=2)
torch.set_printoptions(precision=2)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything()

def evaluate_model(testdataset,net,netfile,test_batch_size,args):
	print("loading trained parameters..")
	net.load_state_dict(torch.load(netfile,map_location=device))
	criterion_mean = displacement_error(args,method="mean").float().to(device)
	criterion_ade = displacement_error(args, method="mean_squared").float().to(device)
	criterion_fde = displacement_error(args, method="final").float().to(device)
	net.eval()
	ade = float(0)
	mean_error = float(0)
	fde = float(0)
	testloader = DataLoader(testdataset,batch_size=test_batch_size,collate_fn=collate_function(),shuffle=True)
	numTest = len(testloader)
	with torch.no_grad():
		for b, batch in enumerate(testloader):
			pred, target, sequence, context_vector, vessels = predict(batch,net)
			pred, target, sequence = pred.squeeze(), target.squeeze(), sequence.squeeze()
			_,ade_batch = criterion_ade(pred,target,vessels)
			_,mean_batch = criterion_mean(pred,target,vessels)
			_,fde_batch = criterion_fde(pred,target,vessels)
			sys.stdout.write("\rbatch: {}/{} ADE: {} mean: {} FDE: {}                  \
			".format(b+1,numTest,ade_batch,mean_batch,fde_batch))
			ade+=ade_batch.item()
			mean_error+=mean_batch.item()
			fde+=fde_batch.item()
	ade/=(b+1)
	mean_error/=(b+1)
	fde/=(b+1)
	print("\rADE: %.5f     Mean Displacement Error: %.5f        FDE: %.5f" %(ade,mean_error,fde))

def train(traindataset, validdataset, testdataset, net, netfile, args):
	eps = 1e-24
	if 'dist' in args.criterion_type:
		criterion = displacement_error(args,method="mean",interaction=False).float().to(device)
	else:
		criterion = displacement_error(args,method="mean",interaction=True).float().to(device)
	best_loss = args.best_loss
	trainloader = DataLoader(traindataset,batch_size=args.batch_size,collate_fn=collate_function(),shuffle=True)
	validloader = DataLoader(validdataset,batch_size=len(validdataset),collate_fn=collate_function(),shuffle=False)
	optimizer = getattr(torch.optim,args.optimizer)(net.parameters(),lr=args.learning_rate)
	if hasattr(net, 'spatialAttention') and not args.use_saved:
		if args.criterion_type=='dist_int_error':
			nn.init.constant_(net.spatialAttention.domain.data, eps)
			net.spatialAttention.domain.requires_grad_(False)
			print("net.spatialAttention.domain.requires_grad_(False)")
		else:
			nn.init.constant_(net.spatialAttention.domain.data,args.param_domain)
	elif args.use_saved:
		print("loading trained parameters from %s" %(netfile))
		net.load_state_dict(torch.load(netfile))
	plotter = Plotter(args)
	if args.scheduler:
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=100, factor = 0.1, patience=4)
	print("Training samples: %d\nValidation Samples: %d\nTesting samples: %d" %(len(traindataset),len(validdataset),len(testdataset)))
	numBatches = len(trainloader)
	numValBatches = len(validloader)
	for epoch in range(args.epochs):
		net.train()
		epoch_loss=float(0)
		for b, batch in enumerate(trainloader):
			optimizer.zero_grad()
			prediction, target, _, _, vessels = predict(batch, net)
			loss, dist_loss = criterion(prediction,target,vessels)
			sys.stdout.write("\repoch: %d/%d training batch: %d/%d total loss: %.2f displacement error: %.2f max num vessels: %d        \r" %(epoch+1,args.epochs,b+1,numBatches,loss.item(),dist_loss.item(),vessels.max()))
			epoch_loss+=dist_loss.item()
			loss.backward()
			optimizer.step()
			if hasattr(net,'spatialAttention'):
				net.spatialAttention.domain.data.clamp_(min=0.00)
		epoch_loss/=(b+1)
		net.eval()
		valid_loss = float(0)
		with torch.no_grad():
			for b, batch in enumerate(validloader):
				prediction, target, _, _, vessels = predict(batch, net)
				loss, dist_loss = criterion(prediction,target,vessels)
				valid_loss+=dist_loss.item()
				sys.stdout.write("\repoch: %d/%d validation batch: %d/%d displacement error: %.2f\r" %(epoch+1,args.epochs,b+1,numValBatches,dist_loss.item()))
		valid_loss/=(b+1)
		if args.scheduler:
			scheduler.step(valid_loss)
		sys.stdout.write("\repoch: %d/%d mean training loss: %.3f mean validation loss: %.3f                       " \
		%(epoch+1,args.epochs,epoch_loss,valid_loss))
		plotter.update(epoch_loss,valid_loss)
		if hasattr(net, 'spatialAttention'):
			print("\nLearned safety domain parameter -->")
			print(net.spatialAttention.domain.data)
		if (valid_loss < best_loss):
			best_loss = valid_loss
			print("saving model...")
			torch.save(net.state_dict(),netfile)
			print("testing..")
			with torch.no_grad():
				evaluate_model(testdataset,net,netfile,len(testdataset),args)
		print("-"*50)
		if args.criterion_type=='dist_int_error' and epoch_loss<=args.threshold and not net.spatialAttention.domain.requires_grad:
			print("learning domain param now")
			net.spatialAttention.domain.requires_grad_(True)
			nn.init.constant_(net.spatialAttention.domain.data, args.param_domain)
			criterion = displacement_error(args,method="mean",interaction=True).float().to(device)

def main(args):
	print("Initializing model..")
	net = getattr(models,args.model)(args.sequence_length,args.prediction_length,args.feature_size,args.hidden_size,args.delta_bearing,args.delta_heading    ,args.param_domain).float().to(device)
	net_dir, data_dir = get_dirs(args)
	traindataset, validdataset,testdataset = load_data(data_dir,args)
	netfile = net_dir+'hsz_'+str(args.hidden_size)+'.pt'
	if hasattr(net, 'spatialAttention'):
		netfile = net_dir+'hsz_'+str(args.hidden_size)+'_'+str(args.criterion_type)+'.pt'
	print("Trained parameters at: %s" %(netfile))
	print("-"*50)
	if args.train:
		train(traindataset, validdataset, testdataset, net, netfile, args)
	if args.test:
		evaluate_model(testdataset,net,netfile,1,args)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--train",action="store_true",help="train model")
	parser.add_argument("--test",action="store_true",help="evaluate model")
	parser.add_argument("--zone",type=int,default=11,help="UTM zone")
	parser.add_argument("--split_data",action="store_true",help="split data into train, valid, test")
	parser.add_argument("--model",type=str,default='spatial_temporal_model',choices=['spatial_temporal_model','spatial_model','temporal_model','vanilla_lstm'],help="model type")
	parser.add_argument('--feature_size',type=int,default=2,help="feature size")
	parser.add_argument('--hidden_size',type=int,default=32,help="hidden LSTM dimension")
	parser.add_argument('--sequence_length',type=int,default=8,help="sequence length")
	parser.add_argument('--prediction_length',type=int,default=12,help="prediction length")
	parser.add_argument('--param_domain',type=float,default=4,help="parameter for domain initialization")
	parser.add_argument('--delta_bearing',type=float,default=5,help="discretization of bearing angle")
	parser.add_argument('--delta_heading',type=float,default=45,help="discretization of heading angle")
	parser.add_argument('--use_saved',action="store_true",help="train a trained model further")
	parser.add_argument('--epochs',type=int,default=2000,help="training epochs")
	parser.add_argument('--batch_size',type=int,default=32,help="batch size")
	parser.add_argument('--learning_rate',type=float,default=0.001,help="learning rate")
	parser.add_argument('--optimizer',type=str,default='Adam',help="optimizer")
	parser.add_argument('--gpuid',type=int,default=1,choices=[0,1,2],help="gpu id")
	parser.add_argument('--best_loss',type=float,default=1000,help="best loss value")
	parser.add_argument('--scheduler',action="store_true",default=False,help="use learning rate scheduler")
	parser.add_argument('--criterion_type',choices=['dist_error','dist_int_error','int_error'],type=str,help="training criterion")	
	parser.add_argument('--threshold',type=float,help="training error threshold to start learning from interaction error")
	args=parser.parse_args()
	print("-"*50)
	print("Parameters:")
	for k, v in vars(args).items():
		print(k,":",v)
	print("-"*50)
	if torch.cuda.is_available() and args.gpuid:
		torch.cuda.set_device(args.gpuid)

	print("Using {}".format(device))
	main(args)
