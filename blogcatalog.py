# coding: utf-8
import scipy.io
import scipy.sparse as sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg
import numpy as np
import dgl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_adjacency_matrix(file, variable_name="network"):
    data = scipy.io.loadmat(file)
    #print("loading mat file %s" % file)
    return data[variable_name]

def expected(A, window):
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    ev, evec = linalg.eigsh(sparse.identity(n) - L, which="LM", k=2)
    # P= D_inv A
    pi = (d_rt ** 2) / vol
    gamma = 1 - (ev[1] if ev[0] == 1.0 else np.abs(ev[0]))
    tau = np.log(8 / (np.min(pi))) / gamma
    P = sparse.diags(d_rt ** -1).dot(sparse.identity(n) - L).dot(sparse.diags(d_rt))
    S = np.zeros_like(P)
    P_power = sparse.identity(n)
    for i in range(window):
        P_power = P_power.dot(P)
        X = sparse.diags(pi).dot(P_power)
        S += X + X.T
    S = S / window / 2.0
    return S.todense(), tau

def empirical(A, window, steps=10000, repeat=2):
    n = A.shape[0]
    g = dgl.DGLGraph()
    g.from_scipy_sparse_matrix(A)

    vol = float(A.sum())
    _, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    pi = (d_rt ** 2) / vol

    seeds = np.random.choice(list(range(n)), size=repeat, replace=True, p=pi).tolist()
    trajectory = dgl.contrib.sampling.random_walk(
                g=g, seeds=seeds, num_traces=1, num_hops=steps-1
            ).tolist()
    Cs = []
    for k in range(repeat):
        C = np.zeros((n, n), dtype=np.float)
        seq = trajectory[k][0]
        for i in range(len(seq) - window):
            u = seq[i]
            for r in range(1, window+1):
                v = seq[i+r]
                C[u, v] += 1./window/2.0
                C[v, u] += 1./window/2.0
        C /= len(seq) - window
        Cs.append(C)
    return Cs

data_248 = []
A = load_adjacency_matrix("blogcatalog.mat")
for window in [2, 4, 8]:
    EC, s = expected(A, window)
    for power in range(1, 8):
        steps = 10 ** power
        repeat = 64
        Cs = empirical(A, window=window, steps=steps, repeat=repeat)
        for k in range(repeat):
            diff = Cs[k]-EC
            ev = linalg.eigsh(diff, which="LM", k=1, return_eigenvectors=False)[0]
            data_248.append([steps, np.abs(ev), window])
            #print(steps, np.abs(ev))

df_blog2 = pd.DataFrame(data_248, columns=["L", "error", "T"], copy=True)
print(df_blog2)
f, ax = plt.subplots(figsize=(10, 10))
ax.set(yscale="log")
sns.boxplot(x="L", y="error", hue='T', data=df_blog2, linewidth=1)

plt.xlabel("L", fontsize=25)
plt.ylabel("error", fontsize=25)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:3], labels[:3], title="T",
          handletextpad=1, columnspacing=1,
          loc="upper right", ncol=3, frameon=True, fontsize=20, title_fontsize=20)
plt.savefig("exp_blog.pdf", format="pdf", bbox="tight")
df_blog2.to_csv("exp_blog.csv", sep=",")
