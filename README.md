# MSCFL

Official Pytorch Implementation of Overcoming System Heterogeneity and Data Shift in Clustered Federated Learning

## Environment

numpy==1.21.5

pandas==0.24.2

Pillow==11.2.1

scikit_learn==0.22.1

scipy==1.4.1

torch==1.12.1+cu116

torchvision==0.13.1+cu116

## Training

```python
python MSCFL.py --dataset fmnist --first_train_ep 10 --epochs 5 --logdir "./logs/" --total_rounds 20
```

This repository offers different clustered federated learning framework of FL+HC, IFCA, one-shot, nxor, FlexCFL, MSCFL algorithm.  More experiment settings are in the miscellaneous.py 
