# GREEN-DR
The official source code of "Graph Residual Re-ranking Network for Diabetic Retinopathy Detection"

# Requirments
- Pytorch >= 1.1.0
- Python == 3.6

# Train
- To train the baseline models (without GCN module), run:
```
python baseline_train.py --arch se_resnext50_32x4d --gpus 0 --train_dataset aptos2019 --epoch 30 -b 32 --lr 1e-3 --min_lr ie-5
```
- To train the GREEN model, run:
```
python train.py --arch se_resnext50_32x4d --gpus 0 --train_dataset aptos2019 --epoch 60 -b 128 --lr 2e-2 --min_lr ie-5
```
