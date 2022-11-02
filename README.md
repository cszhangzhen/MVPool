# MVPool
Hierarchical Multi-View Graph Pooling with Structure Learning ([paper](https://ieeexplore.ieee.org/abstract/document/9460814)).

This is a PyTorch implementation of the MVPool algorithm, which is accepted by TKDE. The proposed MVPool conducts pooling operation via mulit-view information. Then, a structure learning layer is stacked on the pooling operation, which aims to learn a refined graph structure that can best preserve the essential topological information. It's a general operator that can be used in various architectures, including node-level representation learning and graph-level representation learning.

## Requirements
* python3.6
* pytorch==1.3.0
* torch-scatter==1.4.0
* torch-sparse==0.4.3
* torch-cluster==1.4.5
* torch-geometric==1.3.2

Note:
An older version of torch-sparse is needed, lower than 0.4.4. This code repository is heavily built on [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric), which is a Geometric Deep Learning Extension Library for PyTorch. Please refer [here](https://pytorch-geometric.readthedocs.io/en/latest/) for how to install and utilize the library.

## Node Classification Datasets
The input contains:
* x, the feature vectors of the labeled training instances
* y, the one-hot labels of the labeled training instances
* allx, the feature vectors of both labeled and unlabeled training instances (a superset of x)
* graph, a dict in the format {index: [index_of_neighbor_nodes]}.

Let n be the number of both labeled and unlabeled training instances. These n instances should be indexed from 0 to n - 1 in graph with the same order as in allx.

In addition to x, y, allx, and graph as described above, the preprocessed datasets also include:
* tx, the feature vectors of the test instances
* ty, the one-hot labels of the test instances
* test.index, the indices of test instances in graph, for the inductive setting
* ally, the labels for instances in allx.

The indices of test instances in graph for the transductive setting are from #x to #x + #tx - 1, with the same order as in tx.

You can use cPickle.load(open(filename)) to load the numpy/scipy objects x, y, tx, ty, allx, ally, and graph. test.index is stored as a text file. More details can be found at [here](https://github.com/kimiyoung/planetoid).

### Node Classification

![](https://github.com/cszhangzhen/MVPool/blob/main/fig/node-classification.png)

Just execuate the following command for node classification task:
```
python main_node_classification.py
```
### Parameter settings for node classification
| Datasets      | lr        | weight_decay   | batch_size      | pool_ratio     | lambda  | net_layers |
| ------------- | --------- | -------------- | -------- 	   | --------       | -------- | ---------- |
| Cora      | 0.01     | 0.01     	 | Full            | 0.5/0.5/0.8/0.5            | 0.9      | 4			| 
| Citeseer  | 0.005     | 0.1          | Full             | 0.7            | 0.0      | 	1		|
| Pubmed	    | 0.01     | 0.001          | Full             | 0.05/0.6/0.5/0.9            | 1.0      | 4			|
| CS          | 0.01		| 0.01          | Full             | 0.05/0.5/0.5/0.5            | 0.0      | 4			|
| Physics            | 0.01    | 0.01          | Full              | 0.05/0.8/0.8/0.8            | 0.0      | 4          |


## Graph Classification Datasets
Graph classification benchmarks are publicly available at [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).

This folder contains the following comma separated text files (replace DS by the name of the dataset):

**n = total number of nodes**

**m = total number of edges**

**N = number of graphs**

**(1) DS_A.txt (m lines)** 

*sparse (block diagonal) adjacency matrix for all graphs, each line corresponds to (row, col) resp. (node_id, node_id)*

**(2) DS_graph_indicator.txt (n lines)**

*column vector of graph identifiers for all nodes of all graphs, the value in the i-th line is the graph_id of the node with node_id i*

**(3) DS_graph_labels.txt (N lines)** 

*class labels for all graphs in the dataset, the value in the i-th line is the class label of the graph with graph_id i*

**(4) DS_node_labels.txt (n lines)**

*column vector of node labels, the value in the i-th line corresponds to the node with node_id i*

There are OPTIONAL files if the respective information is available:

**(5) DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)**

*labels for the edges in DS_A_sparse.txt* 

**(6) DS_edge_attributes.txt (m lines; same size as DS_A.txt)**

*attributes for the edges in DS_A.txt* 

**(7) DS_node_attributes.txt (n lines)** 

*matrix of node attributes, the comma seperated values in the i-th line is the attribute vector of the node with node_id i*

**(8) DS_graph_attributes.txt (N lines)** 

*regression values for all graphs in the dataset, the value in the i-th line is the attribute of the graph with graph_id i*


### Run Graph Classification

![](https://github.com/cszhangzhen/MVPool/blob/main/fig/graph-classification.png)

Just execuate the following command for graph classification task:
```
python main_graph_classification.py
```

## Citing
If you find MVPool useful for your research, please consider citing the following paper:
```
@article{zhang2021hierarchical,
  title={Hierarchical Multi-View Graph Pooling with Structure Learning},
  author={Zhang, Zhen and Bu, Jiajun and Ester, Martin and Zhang, Jianfeng and Li, Zhao and Yao, Chengwei and Huifen, Dai and Yu, Zhi and Wang, Can},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2021},
  publisher={IEEE}
}
``` 
