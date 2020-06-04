# A Spatially and Temporally Attentive Joint Trajectory Prediction Framework for Modeling Vessel Intent

Ships, or vessels, often sail in and out of cluttered environments over the course of their trajectories. Safe navigation in such cluttered scenarios requires an accurate estimation of the intent of neighboring vessels and their effect on the self and vice-versa well into the future. We propose a spatially and temporally attentive LSTM-based encoder-decoder model that is able to predict future trajectories <em>jointly</em> for all ships in the frame. 

For three agents in a frame as shown below, the trajectory of the <em> red </red?> agent is influenced by that of the other two. The spatial attention mechanism hence weighs the hidden states of these neighbors based on their influence and uses the weighted sum at the next time step. 

<p align="center">
<img src = https://github.com/coordinated-systems-lab/VesselIntentModeling/blob/master/img/spatial_influence.png width="300" height = "300">
</p>

## Spatial Attention Mechanism

To model the spatial influence of neighbors on a vessel of interest and incorporate the influence on the vessel's trajectory, we introduce a <em> spatial attention mechanism </em>. 

<p align="center">
<img src = https://github.com/coordinated-systems-lab/VesselIntentModeling/blob/master/img/spatial_attention_mechanism.png width="400" height = "300">
</p>

## Temporal Attention Mechanism

In the decoder, we also interleave a <em> temporal attention mechanism </em> with the spatial attention mechanism, to enable the model to inform prediction using previously observed spatial situations. 

<p align="center">
<img src = https://github.com/coordinated-systems-lab/VesselIntentModeling/blob/master/img/decoder_method.png width="450" height="300"> 
</p>

## Spatial Influence

We define a trainable parameter, called <em>domain</em> in our spatial attention mechanism. For an agent attempting to navigate safely in a crowded environment, the agent’s domain can be defined as the safe space surrounding the agent, the intrusion of which by any neighboring agent would cause both to have a direct impact on each other’s future intent. 

On training on AIS Data (https://marinecadastre.gov/ais/) from January 2017, our model infers the <em>ship domain </em> as: 

<p align="center">
<img src = https://github.com/coordinated-systems-lab/VesselIntentModeling/blob/master/img/domain.png width="400" height="500">
</p>

Below is an example of the spatial influence computed by our model for 2 nearly similar scenarios. The size of the blue circle is directly proportional to the model inferred spatial influence of that vessel on the neighbor. 

<p align="center">
<img src = https://github.com/coordinated-systems-lab/VesselIntentModeling/blob/master/img/spatial_attn_1.gif width="400"  height="300"> <img src = https://github.com/coordinated-systems-lab/VesselIntentModeling/blob/master/img/spatial_attn_2.gif width="400" height="300"> 
</p>

## Implementation Details

### Step 1: Download Data

To download data run:

```bash
sh download_data.sh
```

This downloads AIS Data from January 2017 from  https://marinecadastre.gov/data/ and saves it in data_processing/raw_data/. 

### Step 2: Preprocess the data

The raw data contains duplicate MMSIs, missing/invalid heading values, etc. that need to be removed. Further, all the vessels transmit AIS data at different frequencies, but for feeding the data into our model, we need to resample all data to 1 minute intervals. For doing this, run:

```bash
python preprocess_data.py --zone
```

Every file contains data corresponding to a Zone. Each vessel is associated with its trajectory over timestamps, i.e. latitude and longitude positions and speed and heading values. Each UTM Zone spans 6&deg; of longitude and 8&deg; of latitude (approximately grid squares of 100 km). For simplicity, we (optionally) split each zone into smaller grids. For example:

```bash
python grid.py --zone=11 --grid_size=0.05 
```

### Step 3: Train a model 

There are four models to choose from: a vanilla LSTM , a spatially attentive LSTM, a temporally attentive LSTM, a spatially and temporally attentive LSTM. 

To train a new model with our best hyper-parameters, run:

```bash
sh scripts/train.sh <model_type> 
```

### Step 4: Test a trained model

To test a trained model, run:

```bash
sh scripts/test.sh <model_type> 
```

