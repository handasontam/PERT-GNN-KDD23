

### Requirements
  * python >= 3.9
  * numpy >= 1.22.3
  * pandas >= 1.4.3
  * scikit-learn >= 1.1.1
  * tqdm
  * torch_geometric >= 2.0.2
  * torch >= 1.10.0
  * pandas >= 1.4.3


Steps to reproduce:
1. Follow https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2021 to fetch the data.
2. Put the MSCallGraph and MSResource directory under `data/`
3. `mkdir processed`
4. Run `python preprocess.py`
5. Run `python pert_gnn.py` (It takes 10+hrs to generate the data_list for the first time)
