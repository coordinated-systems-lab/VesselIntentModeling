import glob
import os
import numpy as np
import pandas as pd
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--grid_size',type=float,default=0.05,help="grid size")

args=parser.parse_args()

files = glob.glob('data_processing/processed_data/*.csv')
sequence_length=10
prediction_length=5
i=0

out_dir = 'data/'

if not os.path.isdir(out_dir):
	os.makedirs(out_dir)

for f, file in enumerate(files):
	print(file)
	df_file = pd.read_csv(file, header=0, parse_dates=['BaseDateTime'])
	df_file.sort_values(['BaseDateTime'],inplace=True)
	df_file['LAT']=df_file['LAT'].round(3)
	df_file['LON']=df_file['LON'].round(3)
	df_file['SOG']=df_file['SOG'].round(3)
	latmin = int(np.floor(df_file['LAT'].min()))
	latmax = int(np.ceil(df_file['LAT'].max()))
	lonmin=int(np.floor(df_file['LON'].min()))
	lonmax=int(np.ceil(df_file['LON'].max()))
	l=latmin
	while not (l+args.grid_size)>latmax:
		l2=lonmin
		df = df_file.loc[((df_file['LAT']>=l)&(df_file['LAT']<=(l+args.grid_size)))]
		while not (l2+args.grid_size)>lonmax:
			df=df.loc[((df['LON']>=l2) & (df['LON']<=(l2+args.grid_size)))]
			timestamps = df['BaseDateTime'].unique()
			if not len(timestamps)<(sequence_length+prediction_length):
				print("Found %d timestamps in grid defined by LAT: %.2f - %.2f and LON: %.2f - %.2f" %(len(timestamps),l,l+args.grid_size,l2,l2+args.grid_size))
				df.to_csv(str(out_dir)+str(i)+'.csv',index=False)
				i+=1
			l2+=args.grid_size
		l+=args.grid_size




