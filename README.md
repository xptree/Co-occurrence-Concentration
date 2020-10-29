# Co-occurrence-Concentration
NeurIPS 2020: A Matrix Chernoff Bound for Markov Chains and its Application to Co-occurrence Matrices

### Install Dependency

```
pip install torch dgl pandas numpy scipy networkx seaborn matplotlib
```

### Run Experiments

We report four experiments in our submission. 
One can use the following commands to reproduce them.
Each command will save a pdf file showing the convergence rate and a csv file for detailed statistics.

For BlogCatalog dataset, please download it from `http://leitang.net/code/social-dimension/data/blogcatalog.mat` manually.

```
python barbel.py
python winning_streak.py
python blogcatalog.py
python random_graph.py
```
