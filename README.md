## Step 1: Download Data

To download data run:

```bash
cd data_processing/
sh download_data.sh
```

This downloads AIS Data from January 2017 from  https://marinecadastre.gov/data/ and saves it in data_processing/raw_data/. 

## Step 2: Preprocess the data

The raw data contains duplicate MMSIs, missing/invalid heading values, etc. that need to be removed. Further, all the vessels transmit AIS data at different frequencies, but for feeding the data into our model, we need to resample all data to 1 minute intervals. For doing this, run:

```bash
cd data_processing/
python preprocess_data.py
```

Every file contains data corresponding to a Zone. Each vessel is associated with its trajectory over timestamps, i.e. latitude and longitude positions and speed and heading values. Each UTM Zone spans 6&deg; of longitude and 8&deg; of latitude (approximately grid squares of 100 km). For simplicity, we (optionally) split each zone into smaller grids. 

```bash
python grid.py --grid_size=0.05
# python grid.py <grid_size> 
```

## Step 3: Train a model 

There are five models to choose from: a vanilla LSTM , a spatially attentive LSTM, a temporally attentive LSTM, a spatially and temporally attentive LSTM (I) and a spatially and temporally attentive LSTM (2). 

To train a new model, run:

```bash
sh scripts/train.sh sta_lstm1 128 10 5
# sh scripts/train.sh <model_type> <hidden_size> <sequence_length> <prediction_length>
```

## Step 4: Test a trained model

To test a trained model, run:

```bash
sh scripts/test.sh sta_lstm1 128 10 5 ADE
# sh scripts/train.sh <model_type> <hidden_size> <sequence_length> <prediction_length> <criterion>
```

