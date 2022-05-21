Taylor Series Hyperbolic Layers
==================================================

## 1. Overview

This repository is based on the [GraphZoo](https://github.com/reddy-lab/GraphZoo) framework:

## 2. Dataset 
The dataset for the experiments can be downloaded from [here](https://data.world/reddy-lab/ptse-hyperbolic-networks)
The citation datasets are part of the graph_data folder.

graph_data
- data
  - cora
  - pubmed
  - citeseer
The repository expects the dataset to be put in a data/ folder, so remove the top-level graph_data/

## 3. Setup
```pip install -r requirements.txt```

## 4. Run Experiments

Sample run command:
```python train.py --task lp --dataset cora --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold TSPoincareBall --log-freq 5 --cuda 0 --c None```

### Models:
1. TGCN: ```--model HGCN --manifold TSPoincareBall```
2. TGAT: ```--model HGAT --manifold TSPoincareBall```

### Tasks:
1. Node Classification: ```--task nc```
2. Link Prediction: ```--task lp```

### Datasets:
1. Cora: ```--dataset cora```
2. Pubmed: ```--dataset pubmed```
3. Citeseer: ```--dataset citeseer```

