import torch
from torch_geometric.datasets import Planetoid
import torch_geometric

dataset = Planetoid(root='./Datasets/Cora', name='Cora')

print('PyTorch Geometric:', torch_geometric.__version__)
print('PyTorch:', torch.__version__)