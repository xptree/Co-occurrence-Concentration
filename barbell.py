# coding: utf-8
import scipy.sparse as sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg
import numpy as np
import dgl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def empirical_regular_markov_chain(G, pi, a, steps=10000, repeat=2):
    n = G.number_of_nodes()
    window = a.shape[0]
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
        C /= steps - window
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

def create_barbell(n_clique, n_path,  a):
    n = n_clique * 2 + n_path
    BB = dgl.DGLGraph()
    BB.add_nodes(n)
    BB.add_edge(n_clique-1, n_clique)
    BB.add_edge(n_clique, n_clique-1)
    for i in range(n_path):
        BB.add_edge(n_clique+i, n_clique+i+1)
        BB.add_edge(n_clique+i+1, n_clique+i)
    for i in range(n_clique):
        for j in range(n_clique):
            if i != j:
                BB.add_edge(i, j)
                BB.add_edge(n_clique+n_path+i, n_clique+n_path+j)
    A = BB.adjacency_matrix_scipy(transpose=False, return_edge_ids=False)
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    ev, evec = linalg.eigsh(sparse.identity(n) - L, which="LM", k=2)
    pi = ((d_rt ** 2) / vol).astype(np.float64)
    P = sparse.diags(d_rt ** -1).dot(sparse.identity(n) - L).dot(sparse.diags(d_rt)).astype(np.float64)
    EC = expected(P, pi, a)
    return BB, pi, EC

T = 2
a = np.ones(T, dtype=np.float32) / T
bb, pi, EC_bb = create_barbell(33, 34, a)
data = []
for power in range(1, 8):
    steps = 10 ** power
    repeat = 64
    C_bb = empirical_regular_markov_chain(bb, pi, a, steps=steps, repeat=repeat)
    for k in range(repeat):
        diff = C_bb[k]-EC_bb
        ev = linalg.eigsh(diff, which="LM", k=1, return_eigenvectors=False)[0]
        data.append((steps, np.abs(ev), 33))
bb, pi, EC_bb = create_barbell(50, 0, a)
for power in range(1, 8):
    steps = 10 ** power
    repeat = 64
    C_bb = empirical_regular_markov_chain(bb, pi, a, steps=steps, repeat=repeat)
    for k in range(repeat):
        diff = C_bb[k]-EC_bb
        ev = linalg.eigsh(diff, which="LM", k=1, return_eigenvectors=False)[0]
        data.append((steps, np.abs(ev), 50))

df = pd.DataFrame(data, columns=["L", "error", "clique"], copy=True)
print(df)

f, ax = plt.subplots(figsize=(10, 10))
ax.set(yscale="log")
sns.boxplot(x="L", y="error", hue='clique', data=df)
sns.swarmplot(x="L", y="error", hue='clique', data=df,
              size=2, color=".3", linewidth=0, dodge=True)

plt.xlabel("L", fontsize=25)
plt.ylabel("error", fontsize=25)
# Improve the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title="size of clique",
          handletextpad=1, columnspacing=1,
          loc="upper right", ncol=2, frameon=True, fontsize=20, title_fontsize=20)
plt.savefig("exp_barbell.pdf", format="pdf", bbox="tight")
df.to_csv("exp_barbell.csv", sep=",")
