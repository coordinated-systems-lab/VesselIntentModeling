import glob
import os
import sys
import numpy as np
import pandas as pd
import random
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
#from mpl_toolkits.basemap import Basemap
import argparse
from global_land_mask import globe
from termcolor import colored
import utm

def read_file(filename):
	df = pd.read_csv(filename,header=0, parse_dates=['BaseDateTime'])
	df.sort_values(['BaseDateTime'],inplace=True)
	return df

def lat_lon_range(df_file):
	min_lat, max_lat = df_file['LAT'].min(),df_file['LAT'].max()
	min_lon, max_lon = df_file['LON'].min(), df_file['LON'].max()
	min_lat, max_lat, min_lon, max_lon = int(np.floor(min_lat)), int(np.ceil(max_lat)), int(np.floor(min_lon)), int(np.ceil(max_lon))
	return min_lat, max_lat, min_lon, max_lon

def ocean_mask(df):
	df['is_ocean'] = globe.is_ocean(df['LAT'],df['LON'])
	df = df[df['is_ocean']==True]
	df = df.drop(['is_ocean'],axis=1)
	return df

def x_y_range(df):
	min_x, max_x = df['x'].min(), df['x'].max()
	min_y, max_y = df['y'].min(), df['y'].max()
	return min_x, max_x, min_y, max_y

def convert_to_utm(df):
	utm_conv = utm.from_latlon(df['LAT'].values, df['LON'].values)
	df['x'] = utm_conv[0]
	df['y'] = utm_conv[1]
	df['zone_1'] = utm_conv[2]
	df['zone_2'] = utm_conv[3]	
	return df

def plot_trajectory_df(df,filtered=True):
	fig, ax = plt.subplots()
	m = Basemap(llcrnrlat=24, llcrnrlon=-125, urcrnrlat=36, urcrnrlon=-111)
	#m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 1500, verbose= True)
	m.bluemarble()
	for v in df['MMSI'].unique():
		df_v = df.loc[df['MMSI']==v]
		vlat, vlon = df_v['LAT'], df_v['LON']
		x, y = m(vlon.values, vlat.values)
		plt.plot(x,y)
	if filtered:
		plt.title("Zone 11 AIS Samples - Filtered")
		plt.savefig("filtered.png")
	else:
		plt.title("Raw Zone 11 AIS samples")
		plt.savefig("raw.png")

def get_grids(filename, grid_size, out_dir,zone):
	i = 0
	print(colored("Processing file: {}".format(filename),"yellow"))
	df = read_file(filename)
	print("Number of Vessels: ",len(df['MMSI'].unique()))
	print(df['LAT'].min(),df['LAT'].max(),df['LON'].min(),df['LON'].max())
	print("Setting LAT, LON range")
	df = df.loc[(df['LAT']>=32)&(df['LON']>=-118)&(df['LAT']<=33)&(df['LON']<=-117)]
	print("Number of Vessels: ",len(df['MMSI'].unique()))
	print("Setting Speed Range")
	df = df.loc[(abs(df['SOG'])<=22)]	
	print("Number of Vessels: ",len(df['MMSI'].unique()))
	print("Removing erroneous AIS messages from land coordinates")
	df = ocean_mask(df)
	print("Removing moored/anchored vessels")
	groups = df.groupby(['MMSI'])
	df = groups.filter(lambda x: abs(x['SOG']).max()>1.0)
	print("Number of Vessels: ",len(df['MMSI'].unique()))
	assert(not(df.empty))
	print("Removing timestamps with records from <=3 vessels")
	groups = df.groupby(['BaseDateTime'])
	df = groups.filter(lambda x: len(x['MMSI'])>3)
	print("Number of Vessels: ",len(df['MMSI'].unique()))
	groups = df.groupby(['BaseDateTime'])
	df = groups.filter(lambda x: abs(x['SOG'].max())>1.0)
	print("Number of Vessels: ",len(df['MMSI'].unique()))
	min_lat, max_lat, min_lon, max_lon = lat_lon_range(df)
	print(min_lat, max_lat, min_lon, max_lon)
	l = min_lat
	min_lat, max_lat, min_lon, max_lon = lat_lon_range(df)
	while not l>=max_lat:
		l2=min_lon
		df_grid = df.loc[((df['LAT']>=l)&(df['LAT']<=(l+grid_size)))]
		while not (l2)>=max_lon:
			df_grid_=df_grid.loc[((df_grid['LON']>=l2) & (df_grid['LON']<=(l2+grid_size)))]
			groups = df_grid_.groupby(['BaseDateTime'])
			df_grid_ = groups.filter(lambda x: len(x['MMSI'])>2)
			if not df_grid_.empty:
				vessels =df_grid_['MMSI'].unique()
				timestamps = df_grid_['BaseDateTime'].unique()
				sys.stdout.write("\rGrid: LAT: %.2f - %.2f and LON: %.2f - %.2f Vessels: %d Timestamps: %d" %(l,l+grid_size,l2,l2+grid_size,len(vessels),len(timestamps)))
				if len(timestamps)>=1000 and len(vessels)>3:
					print(colored("\nFound %d timestamps in grid defined by LAT: %.2f - %.2f and LON: %.2f - %.2f" %(len(timestamps),df_grid_['LAT'].min(),df_grid_['LAT'].max(), df_grid_['LON'].min(), df_grid_['LON'].max()), "blue")) 
					df_grid_.to_csv(str(out_dir)+str(i)+'.csv',index=False)
					i+=1
			l2+=0.1
		l+=0.1
		

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--grid_size",type=float,default=0.2,help="grid size")
	parser.add_argument("--zone",type=int,default=11,help="zone")
	parser.add_argument("--utm",action="store_true",help="UTM coordinates")
	args=parser.parse_args()
	out_dir = "data/%02d/"%(args.zone)
	if not os.path.isdir(out_dir):
		os.makedirs(out_dir)
	in_file = "processed_data/%02d.csv"%(args.zone)
	get_grids(in_file, args.grid_size, out_dir,args.zone)
