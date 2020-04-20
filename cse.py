import numpy as np
import matplotlib.pyplot as plt
import tqdm

def cse(D):
    n, m = D.shape
    assert n == m
    # symmetrise
    Ds = (D + D.T) / 2
    # centralise
    Q = np.eye(n) - np.ones((n, n))/n
    Dc = Q @ Ds @ Q
    # decompose
    Sc = -Dc / 2
    # make PSD
    wmin, *_ = np.linalg.eigvalsh(Sc)
    Sp = Sc - wmin*np.eye(n)
    # compute embedding
    w, v = np.linalg.eigh(Sp)
    d = np.sum(w > 1e-10)
    X = np.sqrt(w[-d:]) * v[:, -d:]
    return X

def embed_and_plot_graph(A, filename="embedding.png", label=False):
    n, _ = A.shape
    print("graph:")
    print(A)
    print("computing distances from adjacency matrix...")
    # floyd-warshall
    D = np.ones((n, n), dtype=float)
    D[np.where(1-A)] = np.inf
    D[np.diag_indices(n)] = 0
    for k in tqdm.trange(n):
        for i in range(n):
            for j in range(n):
                D[i, j] = min(D[i, j], D[i, k] + D[k, j])
    print(D)
    # remove infinities
    D = np.clip(D, 0, np.log(n))
    print("computing embedding...")
    X = cse(D)[:,-2:]
    print(X.T.round(2))
    print("creating and saving plot...")
    plt.scatter(*X.T, color="red")
    ij = np.where(np.triu(A) == 1)
    for i, j in zip(*ij):
        a = X[i]
        b = X[j]
        plt.plot([a[0], b[0]], [a[1], b[1]], color="red", alpha=0.3)
    if label:
        for i in range(n):
            plt.annotate(str(i+1), X[i])
    plt.savefig(filename)
    print("saved @", filename)


# test
if __name__ == "__main__":
    n = 150
    p = 0.02
    np.random.seed(42)
    G = np.zeros((n, n))
    G[np.triu_indices(n, k=1)] = np.random.random(n*(n-1)//2)
    G += G.T
    G = G < p
    embed_and_plot_graph(G)
