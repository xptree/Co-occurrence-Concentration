# coding: utf-8
import scipy.sparse as sparse
from scipy.sparse import linalg
import numpy as np
import dgl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def empirical_regular_markov_chain(G, pi, a, steps=10000, repeat=2):
    n = G.number_of_nodes()
    window = a.shape[0]
    seeds = np.random.choice(list(range(n)), size=repeat, replace=True, p=pi).tolist()
    trajectory = dgl.contrib.sampling.random_walk(
                g=G, seeds=seeds, num_traces=1, num_hops=steps
            ).tolist()
    Cs = []
    for k in range(repeat):
        C = np.zeros((n, n), dtype=np.float32)
        seq = trajectory[k][0]
        for i in range(len(seq) - window):
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
    S = np.zeros_like(P)
    P_power = sparse.identity(n)
    for i in range(a.shape[0]):
        P_power = P_power.dot(P)
        X = sparse.diags(pi).dot(P_power)
        S += a[i] / 2 * (X + X.T)
    return S

def create_winning_streak_chains(n, a):
    WSC = dgl.DGLGraph()
    WSC.add_nodes(n+1)
    pi = np.zeros(n+1)
    P = sparse.dok_matrix((n+1, n+1), dtype=np.float32)
    for i in range(n+1):
        j = i if i == n else i+1
        WSC.add_edge(i, j)
        WSC.add_edge(i, 0)
        pi[i] = 1. / (2 ** j)
        P[i, 0] = 0.5
        P[i, j] = 0.5
    assert pi.sum() == 1.0
    P = P.tocsr()
    EC = expected(P, pi, a)
    return WSC, pi, EC

T = 2
a = np.ones(T, dtype=np.float64) / T
WSC, pi, EC_WSC = create_winning_streak_chains(100, a)
x_WSC = []
y_WSC = []
for power in range(1, 8):
    steps = 10 ** power
    repeat = 64
    C_WSC = empirical_regular_markov_chain(WSC, pi, a, steps=steps, repeat=repeat)
    for k in range(repeat):
        diff = C_WSC[k]-EC_WSC
        ev = linalg.eigsh(diff, which="LM", k=1, return_eigenvectors=False)[0]
        x_WSC.append(steps)
        y_WSC.append(np.abs(ev))

T = 2
a_50 = np.ones(T, dtype=np.float64) / T
WSC_50, pi_50, EC_WSC_50 = create_winning_streak_chains(50, a_50)

x_WSC_50 = []
y_WSC_50 = []
for power in range(1, 8):
    steps = 10 ** power
    repeat = 64
    C_WSC_50 = empirical_regular_markov_chain(WSC_50, pi_50, a_50, steps=steps, repeat=repeat)
    for k in range(repeat):
        diff = C_WSC_50[k]-EC_WSC_50
        ev = linalg.eigsh(diff, which="LM", k=1, return_eigenvectors=False)[0]
        x_WSC_50.append(steps)
        y_WSC_50.append(np.abs(ev))

data = []
for step, error in zip(*(x_WSC, y_WSC)):
    data.append([step, error, 100])
for step, error in zip(*(x_WSC_50, y_WSC_50)):
    data.append([step, error, 50])

df = pd.DataFrame(data, columns=["L", "error", "n"], copy=True)
print(df)

f, ax = plt.subplots(figsize=(10, 10))
ax.set(yscale="log")
sns.boxplot(x="L", y="error", hue='n', data=df)
sns.swarmplot(x="L", y="error", hue='n', data=df,
              size=2, color=".3", linewidth=0, dodge=True)

plt.xlabel("L", fontsize=25)
plt.ylabel("error", fontsize=25)
# Improve the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title="size of state space, n",
          handletextpad=1, columnspacing=1,
          loc="upper right", ncol=2, frameon=True, fontsize=20, title_fontsize=20)
plt.savefig("exp_winning_streak.pdf", format="pdf", bbox="tight")
df.to_csv("exp_winning_streak.csv", sep=",")
