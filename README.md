# The Augmented PEMS-BAY Dataset

This repository contains an augmented version of the PEMS-BAY traffic dataset provided by the [DCRNN authors](https://github.com/liyaguang/DCRNN). In addition to the files in the original dataset, this augmented dataset includes traffic occupancy data (for the same sensors and time period), the spatiotemporal extents of congestion instances, and additional sensor metadata. 

The following sections describe the files in the dataset and the step of generating training/validation/testing data for traffic prediction models such as [DCRNN](https://github.com/liyaguang/DCRNN), [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet) and [STAWnet](https://github.com/CYBruce/STAWnet).

## 1. Data Files
The files in the dataset are in two directories, namely, `traffic_data` and `sensor_graph`.

### 1.1 Traffic Data
The `traffic_data` folder contains the following files:

- `speed.csv`: This file contains traffic speed data (in miles per hour), averaged over 5-minute time steps, for 325 sensors in the San Francisco Bay Area for the first 6 months of 2017. The index column contains the sensor IDS and the header row contains the timestamps. This data is equivalent to the speed data provided in the [PEMS-BAY dataset](https://github.com/liyaguang/DCRNN) except that missing speed values in the source data from [CalTrans PeMS](https://pems.dot.ca.gov/) are not replaced with zeros, but left blank.

- `occupancy.csv`: This file contains traffic occupancy data (percentage expressed as a decimal), averaged over 5-minute time steps, for the same sensors and time period. The index column contains the sensor IDS and the header row contains the timestamps.

- `congestion_blocks.csv`: A *congestion block* represents an instance of congestion along a freeway or road corridor. It has a spatial extent (the most upstream sensor affected to most downstream sensor affected) and a temporal extent (start time to end time). The file `congestion_blocks.csv` contains data on congestion blocks in the PEMS-BAY test set used for evaluating traffic prediction models. It covers the period from 2017-05-25 17:50 to 2017-06-30 23:55.

- `congestion_blocks_mask.csv`: This file stores a binary matrix where each row corresponds to a sensor and each column a time step. A cell with a value of 1 indicates that the corresponding sensor was part of a congestion block at the corresponding time step. The index column in the file contains the sensor IDs and the header row contains the timestamps.

### 1.2 Sensor Graph
The `sensor_graph` folder contains the following files:

- `adj_mx_bay.pkl`: The adjacency matrix of the sensor graph, provided in the PEMS-BAY dataset.

- `distances_bay_2017.csv`: This file contains the distances between sensors, provided in the PEMS-BAY dataset. It is used by the script `generate_adjacency_matrix.py` to create the adjacency matrix.

- `graph_sensor_ids.txt`: The list of sensor IDs in the dataset. It is used by the script `generate_adjacency_matrix.py` to create the adjacency matrix.

- `sensor_metadata.csv`: This file contains the attributes of each sensor, such as the freeway on which the sensor lies, the freeway's direction, the postmile along the freeway, latitude, longitude, length of the road segment covered by the sensor and the number of lanes.

- `freeway_sensor_sequence.csv`: This file provides the ordered sequence of sensors along each of the two directions of each freeway.

## 2. Generating Data for Traffic Prediction Models
Run the below command to generate the data in .npz format for training/validating/testing traffic prediction models.
```
python generate_data_for_models.py`
```
The generated files (`train.npz`, `val.npz`, and `test.npz`) are placed in `data/traffic_data/npz_files`. These files contain the following 3 input features in the given order: speed, occupancy and time-of-day.