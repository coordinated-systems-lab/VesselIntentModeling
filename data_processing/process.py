from __future__ import print_function
import sys
sys.dont_write_bytecode=True
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import glob

files = glob.glob('Zone01/*.csv')
for f, file in enumerate(files):
	print("processing data from ", str(file))
	df = pd.read_csv(file,header=0,parse_dates=['BaseDateTime'],usecols=['MMSI','BaseDateTime','LAT','LON','SOG','Heading'])
	vessels = df['MMSI'].unique()
	print("VESSELS: ",len(vessels))
	df.sort_values(['BaseDateTime'],inplace=True)
	out_frame = pd.DataFrame()
	for v, vessel in enumerate(vessels):
		print("mmsi: ", vessel," ",int(v+1),"/",len(vessels))
		vessel_data = df.loc[df['MMSI']==vessel]
		vessel_data['BaseDateTime'] = pd.to_datetime(vessel_data['BaseDateTime'],format = "%Y-%m-%dT%H:%M:%S")
		vessel_data['BaseDateTime'] = vessel_data['BaseDateTime'].dt.round('T')
		vessel_data = vessel_data.loc[~vessel_data['BaseDateTime'].duplicated(keep='first')]
		idx = split_index(vessel_data['BaseDateTime'])
		if not len(idx)==0:	
			for i, index in enumerate(idx):
				if i==len(idx):
					break
				elif (i==0):
					sub_frame = vessel_data.ix[:index]
				else:
					sub_frame = vessel_data.ix[prev_index:index][1:]
				prev_index=index
				upsampled_indices=pd.date_range(start=sub_frame['BaseDateTime'].values[0], end=sub_frame['BaseDateTime'].values[-1],freq='T')
				sub_frame.set_index(['BaseDateTime'],inplace=True)
				sub_frame=sub_frame.reindex(index=upsampled_indices,fill_value=np.nan)
				sub_frame = sub_frame.interpolate(method='time')
				
				try:
					sub_frame['Heading'] = sub_frame['Heading'].astype('int32')
					if not (np.int(511) in sub_frame['Heading'].values) or (not len(sub_frame['Heading'].unique())==1):
						if (np.int(511) in sub_frame['Heading'].values):
							sub_frame['Heading'].replace(to_replace=511, method='ffill',inplace=True)
							sub_frame['Heading'].replace(to_replace=511, method='bfill',inplace=True)
						out_frame = out_frame.append(sub_frame)
				except ValueError:
					print("invalid heading values found")
		else:
			try:
				vessel_data.set_index(['BaseDateTime'],inplace=True)
				vessel_data['Heading']=vessel_data['Heading'].astype('int32')
				if not (np.int(511) in vessel_data['Heading'].values) or not len(vessel_data['Heading'].unique())==1:
					if (np.int(511) in vessel_data['Heading'].values):
						vessel_data['Heading'].replace(to_replace=511, method='ffill',inplace=True)
						vessel_data['Heading'].replace(to_replace=511, method='ffill',inplace=True)
					out_frame.append(vessel_data)
			except ValueError:
				print("invalid heading values found")
	out_frame.index.name='BaseDateTime'
	out_frame.sort_values(['BaseDateTime'],inplace=True)
	out_frame.to_csv('processed_data/'+str(f+1)+'.csv',index=True)
	print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
