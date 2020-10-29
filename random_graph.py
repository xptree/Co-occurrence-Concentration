# coding: utf-8
import scipy.sparse as sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg
import numpy as np
import dgl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx

def empirical_regular_markov_chain(G, pi, a, steps=10000, repeat=2):
    n = G.number_of_nodes()
    window = a.shape[0]
    assert steps > window
    np.random.seed(0)
    seeds = np.random.choice(list(range(n)), size=repeat, replace=True, p=pi).tolist()
    trajectory = dgl.contrib.sampling.random_walk(
                g=G, seeds=seeds, num_traces=1, num_hops=steps-1
            )
    Cs = []
    for k in range(repeat):
        C = np.zeros((n, n), dtype=np.float64)
        seq = trajectory[k][0]
        for i in range(steps - window):
            u = seq[i]
            for r in range(1, window+1):
                v = seq[i+r]
                C[u, v] += a[r-1]/2
                C[v, u] += a[r-1]/2
        C /= len(seq) - window
        Cs.append(C)
    return Cs


def expected(P, pi, a):
    n = P.shape[0]
    S = np.zeros_like(P, dtype=np.float64)
    P_power = sparse.identity(n)
    for i in range(a.shape[0]):
        P_power = P_power.dot(P)
        X = sparse.diags(pi).dot(P_power)
        S += a[i] / 2 * (X + X.T)
    return S

def create_gnp(n, p, a=None):
    nx_gnp = nx.generators.gnp_random_graph(n, p, seed=0, directed=False)
    assert nx.is_connected(nx_gnp)
    gnp = dgl.DGLGraph()
    gnp.from_networkx(nx_gnp.to_directed())
    A = gnp.adjacency_matrix_scipy(transpose=False, return_edge_ids=False)
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    ev, evec = linalg.eigsh(sparse.identity(n) - L, which="LM", k=2)
    pi = ((d_rt ** 2) / vol).astype(np.float64)
    P = sparse.diags(d_rt ** -1).dot(sparse.identity(n) - L).dot(sparse.diags(d_rt)).astype(np.float64)
    EC = expected(P, pi, a)
    return gnp, pi, EC

data = []
for T in [2, 4, 8]:
    a = np.ones(T, dtype=np.float64) / T
    gnp, pi, EC_gnp = create_gnp(100, 0.1, a)
    for power in range(1, 8):
        steps = 10 ** power
        repeat = 64
        C_gnp = empirical_regular_markov_chain(gnp, pi, a, steps=steps, repeat=repeat)
        for k in range(repeat):
            diff = C_gnp[k]-EC_gnp
            ev = linalg.eigsh(diff, which="LM", k=1, return_eigenvectors=False)[0]
            data.append([steps, np.abs(ev), T])

df_gnp = pd.DataFrame(data, columns=["L", "error", "T"], copy=True)
print(df_gnp)
f, ax = plt.subplots(figsize=(10, 10))
ax.set(yscale="log")
sns.boxplot(x="L", y="error", hue='T', data=df_gnp, linewidth=1)

plt.xlabel("L", fontsize=25)
plt.ylabel("error", fontsize=25)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:3], labels[:3], title="T",
          handletextpad=1, columnspacing=1,
          loc="upper right", ncol=3, frameon=True, fontsize=20, title_fontsize=20)
plt.savefig("exp_random_graph.pdf", format="pdf", bbox="tight")
df_gnp.to_csv("exp_random_graph.csv", sep=",")
