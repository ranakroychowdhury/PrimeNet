# PrimeNet : Pre-Training for Irregular Multivariate Time Series
This is the official PyTorch implementation of the [AAAI 2023](https://aaai.org/Conferences/AAAI-23/) paper [PrimeNet: Pre-Training for Irregular Multivariate Time Series](https://drive.google.com/file/d/1FSy9F_qO-eaNhKYWbCwpBCogC1ZNo8nd/view?usp=sharing).

![alt text](https://github.com/ranakroychowdhury/PrimeNet/blob/main/setup.png)

## Quick Start
```
git clone https://github.com/ranakroychowdhury/PrimeNet.git
```

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


The data directory structure should be the same as that in `data.zip`. Extract `data.zip` to run experiments with a sample toy dataset.  
> data/  
>> pretrain/  
>>> X_train.pt  
>>> X_val.pt 
>>>  
>> finetune/ 
>>> X_train.pt  
>>> y_train.pt  
>>> X_val.pt  
>>> y_val.pt  
>>> X_test.pt  
>>> y_test.pt


## Pre-training

Run `pretrain.sh` to run pretraining experiments on a dataset. The pretrained model is saved under `./models/` and the pretraining results are stored under `./results/`. The arguments for pretraining are explained in `pretrain.py`.

```
sh pretrain.sh
```


## Fine-tuning and Evaluation

Run `finetune.sh` to run finetuning experiments on a dataset. The pretrained model saved during the pretraining experiment under `./models/` is used for finetuning. The name of the pretrained model to use is added as an argument in the finetuning command. The finetuning results are stored under `./results/`. The arguments for finetuning are explained in `finetune.py`.

```
sh finetune.sh
```


## Reference
