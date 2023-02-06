# PrimeNet : Pre-Training for Irregular Multivariate Time Series
This is the official PyTorch implementation of the [AAAI 2023](https://aaai.org/Conferences/AAAI-23/) paper **PrimeNet : Pre-Training for Irregular Multivariate Time Series** BY 

![alt text](https://github.com/ranakroychowdhury/PrimeNet/blob/main/setup.png)



## Datasets

### Activity
Run `preprocess/download_activity.sh` to download and place the data under the proper directory.

Run `preprocess/preprocess_activity.py` to preprocess the data. 


### MIMIC-III
Follow the instruction from [interp-net](https://github.com/mlds-lab/interp-net) to download and preprocess the data.


### Appliances Energy
[Download](https://zenodo.org/record/3902637) the dataset.

Run `preprocess/preprocess_ae.py` to preprocess the data.


### PhysioNet
Run `preprocess/preprocess_physionet.py` to download and preprocess the data.


The data directory structure should be the same as that in data.zip. Extract data.zip to run experiments with a sample toy dataset.
`data/
  |--pretrain/
        |--X_train.pt
        |--X_val.pt
  |--finetune/
        |--X_train.pt
        |--y_train.pt
        |--X_val.pt
        |--y_val.pt
        |--X_test.pt
        |--y_test.pt`


## Pre-training



## Fine-tuning and Evaluation



## Reference
