## Neural Random Forests

Relevant paper:
[Neural Random Forests](https://arxiv.org/abs/1604.07143).

## Requirements
This code is based on python3 and uses tensorflow 1.3.0.

First, let's make sure you have all packages needed:
```
pip3 install -r requirements.txt
```


## Quick Start
For a quick start, let's download the [mpg](https://archive.ics.uci.edu/ml/datasets/auto+mpg) dataset from the UCI Machine Learning Repository (30KB):
```
cd datasets/data/mpg_data
sh download.sh
```

To run different Neural Random Forest models on the mpg dataset, execute this (takes ~2min) from the repository root directory:
```
python3 main.py mpg
```

## Other Datasets
To run the model on a new dataset, you must write a data loader function and add an option to `data_loader.py`.
For inspiration, check out the data loaders in `preprocessing/` which are for other datasets used in the [paper](https://arxiv.org/abs/1604.07143) . 

The data loader functions all return a pair _(X, Y)_, where _X_  is an input matrix of size `[# samples, # features]`, and _Y_  is a vector of regression outputs with size `[# samples]`.
