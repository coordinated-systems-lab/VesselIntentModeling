import torch
import numpy as np
from utils import *
from plotting_utils import *
import argparse
from model1 import *
from model2 import *
from spatial_vlstm import *
parser=argparse.ArgumentParser(description="vessel intent modeling")

parser.add_argument('--feature_size',type=int,default=4,help="feature size")
parser.add_argument('--output_size',type=int,default=2,help="output size")
parser.add_argument('--maxVessels',type=int,default=30,help="maximum number of vessels per sample")
parser.add_argument('--hidden_size',type=int,default=300,help="hidden LSTM dimension")
parser.add_argument('--sequence_length',type=int,default=10,help="sequence length")
parser.add_argument('--prediction_length',type=int,default=10,help="prediction length")
parser.add_argument('--delta_rb',type=float,default=30,help="relative bearing discretization")
parser.add_argument('--delta_cog',type=float,default=45,help="heading discretization")
parser.add_argument('--min_dist',type=float,default=0.001,help="minimum domain distance")
parser.add_argument('--model_type',default="sta_lstm",help="model to use")
parser.add_argument('--param_domain',default=2.5,type=float,help="domainparam")
parser.add_argument('--net_dir',default='models/',type=str,help="net directory")
parser.add_argument('--save_new',action="store_true",help="save separate instead of overwriting if corresponding file already exists")

args=parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
plot_dir='plots/'
max_dist=5
netfile = args.net_dir+args.model_type+'/hsz_'+str(args.hidden_size)+'sequence_length_'+str(args.sequence_length)+'prediction_length_'+str(args.prediction_length)+'_'+str(args.maxVessels)+'.pt'
if args.model_type=='sta_lstm1':
	net = model1(args).to(device)
elif args.model_type=='sta_lstm2':
	net=model2(args).to(device)
else:
	net=spatial_vlstm(args).to(device)
net.load_state_dict(torch.load(netfile))
r = net.hardwiredAttention.domain.data
plot_domain(r,args.delta_rb,args.sequence_length,args.prediction_length,args.delta_cog,max_dist,plot_dir,args.model_type,args.param_domain,args.save_new,approx=False)
