

### Requirements
  * 200G+ Disk space

Steps to reproduce:
1. Create conda environemnt: `conda env create -f environment.yml` and activate it: `conda activate pert-gnn`
2. Follow https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2021 to fetch the data.
3. Put the MSCallGraph and MSResource directory under `data/`
4. `mkdir processed`
5. Run `python preprocess.py`
6. Run `python pert_gnn.py` (It takes 10+hrs to generate the data_list for the first time)
