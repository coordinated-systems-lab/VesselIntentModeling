from __future__ import print_function
from global_land_mask import globe
import os
import sys
sys.dont_write_bytecode=True
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import glob

if not os.path.isdir('processed_data/'):
	os.makedirs('processed_data/')

files = glob.glob('raw_data/*.csv')
files.sort()
print(files)
for f, filename in enumerate(files):
	if not 'Zone11' in filename:
		continue
	print("processing data from ", str(filename))
	out_file = filename.split("Zone")[1]
	df = pd.read_csv(filename,header=0,parse_dates=['BaseDateTime'],usecols=['MMSI','BaseDateTime','LAT','LON','SOG','Heading'])
	vessels = df['MMSI'].unique()
	df.sort_values(['BaseDateTime'],inplace=True)
	out_frame = pd.DataFrame()
	for v, vessel in enumerate(vessels):
		print("mmsi: ", vessel," ",int(v+1),"/",len(vessels))
		vessel_data = df.loc[df['MMSI']==vessel]
		vessel_data['BaseDateTime'] = pd.to_datetime(vessel_data['BaseDateTime'],format = "%Y-%m-%dT%H:%M:%S")
		vessel_data['BaseDateTime'] = vessel_data['BaseDateTime'].dt.ceil('min')
		vessel_data = vessel_data.loc[~vessel_data['BaseDateTime'].duplicated(keep='first')]
		vessel_data = vessel_data.set_index(['BaseDateTime']).resample('1min').interpolate(limit=5)
		vessel_data.reset_index('BaseDateTime',inplace=True)
		vessel_data = vessel_data.dropna(subset=['LAT','LON'])
		try:
			vessel_data.set_index(['BaseDateTime'],inplace=True)
			vessel_data['Heading']=vessel_data['Heading'].astype('int32')
			if not len(vessel_data['Heading'].unique())==1:
				if (np.int(511) in vessel_data['Heading'].values):
					vessel_data['Heading'].replace(to_replace=511, method='ffill',inplace=True)
					vessel_data['Heading'].replace(to_replace=511, method='bfill',inplace=True)
				out_frame = out_frame.append(vessel_data)
		except ValueError:
			print("invalid heading values found")
	#	out_frame = out_frame.append(vessel_data)
	out_frame.index.name='BaseDateTime'
	out_frame.sort_values(['BaseDateTime'],inplace=True)
	print("Saving processed data to processed_data/" + out_file)
	out_frame.to_csv('processed_data/'+out_file, index=True)
	print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
