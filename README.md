## Perceptual Computing

This is team Procyon's repository for the assignment of Perceptual Computing.
We participated in [Understanding Clouds from Satellite Images](https://www.kaggle.com/c/understanding_cloud_organization).

### Descriptions

#### Files
- dataset.py : Define Dataset Class
- loss.py : Define Loss Class
- train.py : Execute training
- trainer.py : Define Trainer Class
- utils.py : Define Functions to Use

#### Directories
- dataset/ : Containing Dataset
- notebook/ : To Try Some Scripts

### Environments
If you use `pipenv`, run these commands to set the environment other than `torch` and `torchvision`.
```
pipenv shell
```

I used `torch==1.4.0+cu100 torchvision==0.5.0+cu100`.  
You should install torch and torchvision for your environment like GPU.

### Download Dataset
You can download dataset with kaggle API. Before downloading it, you should move to `dataset` directory
```
cd dataset
kaggle competitions download -c understanding_cloud_organization
```

### Cautions
When you can use GPU, this scripts use multiple GPUs automatically.
