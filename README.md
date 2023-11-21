# STFed
An implementation for our paper: **A Self-Training Dual-Network Denoising Framework for Federated Recommendation**

## dependencies
- pytorch>=1.8.2 (CUDA version)
- tqdm

## ML1M
To specify the model (base recommendation model) and mode (baseline or relabeling method), use the following command.
```
python train_federated.py --dataset ml-1m --mode baseline_noisy --model ncf --lr 0.0005 --num_clients 600
```

## Acknowledgment
Thanks to the [NCF implementation under the federated setting](https://github.com/omarmoo5/Federated-Neural-Collaborative-Filtering#readme).
