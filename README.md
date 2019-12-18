## Step 1: Download Data

To download data run:

```bash
sh download_data.sh
```

This downloads AIS Data from January 2017 from  https://marinecadastre.gov/data/ and saves it in data_processing/raw_data/. 

## Step 2: Preprocess the data

The raw data contains duplicate MMSIs, missing/invalid heading values, etc. that need to be removed. Further, all the vessels transmit AIS data at different frequencies, but for feeding the data into our model, we need to resample all data to 1 minute intervals. For doing this, run:

```bash
python preprocess_data.py --zone
```

Every file contains data corresponding to a Zone. Each vessel is associated with its trajectory over timestamps, i.e. latitude and longitude positions and speed and heading values. Each UTM Zone spans 6&deg; of longitude and 8&deg; of latitude (approximately grid squares of 100 km). For simplicity, we (optionally) split each zone into smaller grids. For example:

```bash
python grid.py --zone=11 --grid_size=0.05 
```

## Step 3: Train a model 

There are four models to choose from: a vanilla LSTM , a spatially attentive LSTM, a temporally attentive LSTM, a spatially and temporally attentive LSTM. 

To train a new model with our best hyper-parameters, run:

```bash
sh scripts/train.sh <model_type> 
```

## Step 4: Test a trained model

To test a trained model, run:

```bash
sh scripts/test.sh <model_type> 
```

